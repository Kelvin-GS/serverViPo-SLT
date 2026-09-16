import argparse
import os
import os.path as osp
import shutil
import json
import tqdm
import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, CLIPVisionModel

import sys
sys.path.append('./')

from preprocess.SP_FT.s2wrapper import forward as multiscale_forward
from preprocess.SP_FT.helpers import read_video, get_img_list


_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)


class ViTFeatureReader(object):
    def __init__(
        self,
        model_name='openai/clip-vit-large-patch14',
        cache_dir=None,
        device='cuda:0',
        s2_mode='s2wrapping',
        scales=[1, 2],
        nth_layer=-1
    ):
        self.s2_mode = s2_mode
        self.device = device
        self.scales = scales
        self.nth_layer = nth_layer

        self.model = CLIPVisionModel.from_pretrained(
            model_name,
            output_hidden_states=True,
            cache_dir=cache_dir
        ).to(device).eval()

        self.image_processor = AutoImageProcessor.from_pretrained(
            model_name
        )

    @torch.no_grad()
    def forward_features(self, inputs):
        outputs = self.model(inputs).hidden_states
        outputs = outputs[self.nth_layer]
        print(outputs.shape, " =============")
        return outputs

    @torch.no_grad()
    def get_feats(self, video):
        inputs = self.image_processor(
            list(video),
            return_tensors="pt"
        ).to(self.device).pixel_values

        if self.s2_mode == "s2wrapping":
            outputs = multiscale_forward(
                self.forward_features,
                inputs,
                scales=self.scales,
                num_prefix_token=1
            )
        else:
            outputs = self.forward_features(inputs)

        return outputs[:, 0]


# -------------------------------------------------------------------------
# Checkpoint helpers
# -------------------------------------------------------------------------

def atomic_save_npy(path, array):
    """
    Atomically write a numpy array.

    The array is first written to a temporary file and then moved into
    place with os.replace(). This prevents an interrupted write from
    leaving a file that looks like a completed checkpoint/output.
    """
    tmp_path = f"{path}.tmp"

    try:
        with open(tmp_path, "wb") as f:
            np.save(f, array)
            f.flush()
            os.fsync(f.fileno())

        os.replace(tmp_path, path)

    except Exception:
        if osp.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise


def atomic_save_json(path, data):
    """
    Atomically write checkpoint metadata.
    """
    tmp_path = f"{path}.tmp"

    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())

        os.replace(tmp_path, path)

    except Exception:
        if osp.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise


def get_checkpoint_dir(save_path, fileid):
    """
    Internal checkpoint directory for one sample.

    This does not alter the expected final output structure.
    """
    return osp.join(
        save_path,
        ".vit_checkpoints",
        str(fileid)
    )


def clear_checkpoint(checkpoint_dir):
    """
    Remove the checkpoint for a sample after its final output has been
    successfully written.
    """
    if osp.isdir(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)


def prepare_checkpoint(checkpoint_dir, metadata):
    """
    Create a checkpoint directory and establish/validate its metadata.

    If an existing checkpoint was generated with incompatible extraction
    parameters, it is discarded rather than mixed with the current run.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    metadata_path = osp.join(
        checkpoint_dir,
        "metadata.json"
    )

    if osp.exists(metadata_path):
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                existing_metadata = json.load(f)

            if existing_metadata != metadata:
                clear_checkpoint(checkpoint_dir)
                os.makedirs(checkpoint_dir, exist_ok=True)

        except (OSError, json.JSONDecodeError):
            clear_checkpoint(checkpoint_dir)
            os.makedirs(checkpoint_dir, exist_ok=True)

    if not osp.exists(metadata_path):
        atomic_save_json(
            metadata_path,
            metadata
        )


def get_batch_checkpoint_path(checkpoint_dir, batch_index):
    """
    Path for a single extracted batch.
    """
    return osp.join(
        checkpoint_dir,
        f"batch_{batch_index:08d}.npy"
    )


def remove_incomplete_batches(checkpoint_dir, first_missing_batch):
    """
    Remove any later batch checkpoints after the first missing batch.

    This ensures that only one contiguous sequence of completed batches
    is ever considered resumable.
    """
    if not osp.isdir(checkpoint_dir):
        return

    for filename in os.listdir(checkpoint_dir):
        if not filename.startswith("batch_"):
            continue

        if not filename.endswith(".npy"):
            continue

        try:
            batch_index = int(filename[6:-4])
        except ValueError:
            continue

        if batch_index >= first_missing_batch:
            path = osp.join(
                checkpoint_dir,
                filename
            )

            try:
                os.remove(path)
            except OSError:
                pass


def find_resume_batch(checkpoint_dir, num_batches):
    """
    Return the first batch that has not been successfully checkpointed.

    Only a contiguous sequence beginning with batch 0 is accepted.
    """
    for batch_index in range(num_batches):
        batch_path = get_batch_checkpoint_path(
            checkpoint_dir,
            batch_index
        )

        if not osp.exists(batch_path):
            remove_incomplete_batches(
                checkpoint_dir,
                batch_index
            )
            return batch_index

    return num_batches


def load_checkpointed_features(checkpoint_dir, num_batches):
    """
    Load completed batch features in their original extraction order.
    """
    features = []

    for batch_index in range(num_batches):
        batch_path = get_batch_checkpoint_path(
            checkpoint_dir,
            batch_index
        )

        if not osp.exists(batch_path):
            raise RuntimeError(
                f"Missing checkpoint batch {batch_index} in "
                f"{checkpoint_dir}"
            )

        features.append(
            np.load(batch_path)
        )

    return features


def extract_features_with_resume(
    videos,
    reader,
    batch_size,
    checkpoint_dir,
    checkpoint_metadata
):
    """
    Perform the same batched feature extraction as the original code,
    while checkpointing each completed batch.

    On restart:
        - previously completed batches are loaded;
        - extraction resumes from the first missing batch;
        - completed batches are not recomputed.
    """
    num_batches = (
        len(videos) + batch_size - 1
    ) // batch_size

    prepare_checkpoint(
        checkpoint_dir,
        checkpoint_metadata
    )

    start_batch = find_resume_batch(
        checkpoint_dir,
        num_batches
    )

    # All batches were already completed before the interruption.
    if start_batch == num_batches:
        video_feats = load_checkpointed_features(
            checkpoint_dir,
            num_batches
        )

        return np.concatenate(
            video_feats,
            axis=0
        )

    # Restore the completed prefix.
    if start_batch > 0:
        video_feats = load_checkpointed_features(
            checkpoint_dir,
            start_batch
        )
    else:
        video_feats = []

    # Continue from the first missing batch.
    for j in range(start_batch, num_batches):
        video_batch = videos[
            j * batch_size:
            min((j + 1) * batch_size, len(videos))
        ]

        # This is the original feature extraction operation.
        feats = reader.get_feats(
            video_batch
        ).cpu().numpy()

        # Persist the completed batch immediately.
        batch_path = get_batch_checkpoint_path(
            checkpoint_dir,
            j
        )

        atomic_save_npy(
            batch_path,
            feats
        )

        video_feats.append(feats)

    # Preserve the original concatenation behavior.
    return np.concatenate(
        video_feats,
        axis=0
    )


# -------------------------------------------------------------------------
# Argument parser
# -------------------------------------------------------------------------

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--anno_root',
        help='location of tsv files',
        required=True
    )

    parser.add_argument(
        '--video_root',
        help='location of tsv files',
        required=True
    )

    parser.add_argument(
        '--device',
        help='device to use',
        default='cuda:0'
    )

    parser.add_argument(
        '--s2_mode',
        default=''
    )

    parser.add_argument(
        '--scales',
        nargs='+',
        type=int,
        help='List of scales',
        default=[]
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=32
    )

    parser.add_argument(
        '--nth_layer',
        type=int,
        default=-1
    )

    parser.add_argument(
        '--cache_dir',
        help='cache dir for model',
        default=None
    )

    parser.add_argument(
        '--save_dir',
        help='where to save the output',
        required=True
    )

    parser.add_argument(
        '--model_name',
        help='ViT model name',
        default='openai/clip-vit-large-patch14'
    )

    return parser


# -------------------------------------------------------------------------
# Data iterator
# -------------------------------------------------------------------------

def get_iterator(args, mode, save_mode=None):
    batch_size = args.batch_size

    data = np.load(
        os.path.join(
            args.anno_root,
            f'{mode}_info.npy'
        ),
        allow_pickle=True
    ).item()

    num = len(data) - 1

    ds_name = osp.split(
        args.anno_root
    )[-1]

    reader = ViTFeatureReader(
        args.model_name,
        device=args.device,
        s2_mode=args.s2_mode,
        scales=args.scales,
        nth_layer=args.nth_layer,
        cache_dir=args.cache_dir
    )

    # save_mode is the actual directory used by main().
    # This fixes the case where, for example, How2Sign uses "val"
    # as the source mode but writes results under "dev".
    if save_mode is None:
        save_mode = mode

    model_name = os.path.split(
        args.model_name
    )[-1]

    output_dir = osp.join(
        args.save_dir,
        f'{model_name}_feat_{ds_name}',
        save_mode
    )

    def iterate():
        for i in range(num):
            fname = data[i]['folder']
            fileid = data[i]['fileid']

            # This is the same final output path used by main().
            target = osp.join(
                output_dir,
                f'{fileid}.npy'
            )

            # Completed final output -> skip.
            if osp.exists(target):
                continue

            print(ds_name)

            # -------------------------------------------------------------
            # Phoenix2014-T / CSL-Daily
            # -------------------------------------------------------------
            if (
                ds_name == 'phoenix2014-T'
                or ds_name == 'CSL-Daily'
            ):
                image_list = get_img_list(
                    ds_name,
                    args.video_root,
                    fname
                )

                videos = [
                    Image.open(image).convert('RGB')
                    for image in image_list
                ]

                # Nothing has changed about how batches are extracted.
                checkpoint_dir = get_checkpoint_dir(
                    output_dir,
                    fileid
                )

                checkpoint_metadata = {
                    'fileid': str(fileid),
                    'dataset': str(ds_name),
                    'mode': str(mode),
                    'save_mode': str(save_mode),
                    'model_name': str(args.model_name),
                    'batch_size': int(args.batch_size),
                    's2_mode': str(args.s2_mode),
                    'scales': list(args.scales),
                    'nth_layer': int(args.nth_layer),
                    'num_videos': int(len(videos))
                }

                if len(videos) == 0:
                    # Preserve original behavior as closely as possible.
                    yield (
                        [],
                        fileid,
                        None
                    )
                    continue

                feats = extract_features_with_resume(
                    videos=videos,
                    reader=reader,
                    batch_size=batch_size,
                    checkpoint_dir=checkpoint_dir,
                    checkpoint_metadata=checkpoint_metadata
                )

                yield (
                    feats,
                    fileid,
                    None
                )

            # -------------------------------------------------------------
            # Other datasets
            # -------------------------------------------------------------
            else:
                if ds_name == 'How2Sign':
                    start_time = data[i][
                        'original_info'
                    ][
                        'START_REALIGNED'
                    ]

                    end_time = data[i][
                        'original_info'
                    ][
                        'END_REALIGNED'
                    ]

                    videos = read_video(
                        fname,
                        start_time=start_time,
                        end_time=end_time
                    )

                # This branch retains the original dataset behavior:
                # datasets other than the explicitly supported video
                # datasets must provide a `videos` variable in the same
                # way as the original implementation expected.
                if len(videos) > 0:
                    checkpoint_dir = get_checkpoint_dir(
                        output_dir,
                        fileid
                    )

                    checkpoint_metadata = {
                        'fileid': str(fileid),
                        'dataset': str(ds_name),
                        'mode': str(mode),
                        'save_mode': str(save_mode),
                        'model_name': str(args.model_name),
                        'batch_size': int(args.batch_size),
                        's2_mode': str(args.s2_mode),
                        'scales': list(args.scales),
                        'nth_layer': int(args.nth_layer),
                        'num_videos': int(len(videos)),
                        'start_time': str(start_time),
                        'end_time': str(end_time)
                    }

                    feats = extract_features_with_resume(
                        videos=videos,
                        reader=reader,
                        batch_size=batch_size,
                        checkpoint_dir=checkpoint_dir,
                        checkpoint_metadata=checkpoint_metadata
                    )

                    yield (
                        feats,
                        fileid,
                        str(start_time)
                    )

                else:
                    yield (
                        [],
                        fileid,
                        str(start_time)
                    )

    return iterate, num


# -------------------------------------------------------------------------
# Progress helpers
# -------------------------------------------------------------------------

def count_completed_outputs(save_path):
    """
    Count final feature files that already exist for this mode.

    These files are the authoritative completion markers used by the
    resume logic. Checkpoint files are intentionally not counted here.
    """
    if not osp.isdir(save_path):
        return 0

    return sum(
        1
        for filename in os.listdir(save_path)
        if filename.endswith('.npy')
    )


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main():
    parser = get_parser()
    args = parser.parse_args()

    mode = [
        "dev",
        "test",
        "train"
    ]

    for m in mode:
        ds_name = osp.split(
            args.anno_root
        )[-1]

        model_name = os.path.split(
            args.model_name
        )[-1]

        fname = (
            f'{model_name}_feat_{ds_name}'
        )

        save_path = osp.join(
            args.save_dir,
            fname,
            m
        )

        os.makedirs(
            save_path,
            exist_ok=True
        )

        # Preserve the original source-mode mapping.
        if ds_name == 'How2Sign':
            if m == 'dev':
                _m = 'val'
            else:
                _m = m

        elif ds_name == 'NIASL2021':
            if m == 'dev':
                _m = 'validation'
            else:
                _m = m

        else:
            _m = m

        # IMPORTANT:
        # `_m` tells the iterator which annotation file to read,
        # while `m` remains the actual output directory.
        generator, num = get_iterator(
            args,
            _m,
            save_mode=m
        )

        iterator = generator()

        # The iterator intentionally skips samples whose final .npy file
        # already exists. Because of that, letting tqdm automatically count
        # yielded samples makes the progress bar misleading after a resume:
        # it would display, for example, 1/7096 even when 3332 files were
        # already completed.
        #
        # Initialize the bar from the number of existing final outputs and
        # update it only after a new final output has been written
        # successfully. This preserves the extraction/resume logic while
        # making the displayed progress reflect actual completed work.
        completed = count_completed_outputs(save_path)

        if completed > num:
            completed = num

        pbar = tqdm.tqdm(
            total=num,
            initial=completed,
            desc=f"{m}",
            unit="sample",
            dynamic_ncols=True,
            bar_format=(
                "{l_bar}{bar}| {n_fmt}/{total_fmt} "
                "[{elapsed}<{remaining}, {rate_fmt}]"
            )
        )

        try:
            for vit_feat in iterator:
                feats, fileid, st = vit_feat

                # Preserve the original postfix logic exactly.
                #
                # NOTE:
                # The original code calculated `postfix` but did not include
                # it in the filename. We deliberately keep that behavior so
                # the externally expected output filename remains unchanged.
                postfix = ""

                if args.s2_mode != "":
                    postfix = f"_{args.s2_mode}"

                if len(args.scales) == 3:
                    postfix = f'{postfix}_large'

                if st is not None:
                    postfix = f'_{st}{postfix}'

                final_path = osp.join(
                    save_path,
                    f'{fileid}.npy'
                )

                # Atomically publish the final output.
                #
                # The final output is only considered complete once this
                # operation succeeds.
                atomic_save_npy(
                    final_path,
                    feats
                )

                # Only remove checkpoints after the final output exists.
                checkpoint_dir = get_checkpoint_dir(
                    save_path,
                    fileid
                )

                clear_checkpoint(
                    checkpoint_dir
                )

                # Update progress only after the final output and checkpoint
                # cleanup have succeeded. If extraction or saving fails, the
                # progress bar therefore does not falsely claim completion.
                pbar.update(1)

        finally:
            pbar.close()


if __name__ == "__main__":
    main()
