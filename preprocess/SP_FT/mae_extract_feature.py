import os
import json
import shutil
import numpy as np
import torch
import argparse
import tqdm
import os.path as osp
from PIL import Image
from transformers import VideoMAEModel, VideoMAEImageProcessor

import sys
sys.path.append('./')

from preprocess.SP_FT.helpers import sliding_window_for_list, read_video, get_img_list

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True


class VideoMAEFeatureReader(object):
    def __init__(
        self,
        model_name='MCG-NJU/videomae-large',
        cache_dir=None,
        device='cuda:0',
        overlap_size=0,
        nth_layer=-1
    ):
        self.device = device
        self.overlap_size = overlap_size
        self.nth_layer = nth_layer

        self.image_processor = VideoMAEImageProcessor.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )
        self.model = VideoMAEModel.from_pretrained(model_name).to(self.device).eval()

    @torch.no_grad()
    def get_feats(self, video):
        inputs = self.image_processor(
            images=video,
            return_tensors="pt"
        ).to(self.device)

        outputs = self.model(
            **inputs,
            output_hidden_states=True
        ).hidden_states

        outputs = outputs[self.nth_layer]
        outputs = outputs[:, 0]

        return outputs


def atomic_save_npy(path, array):
    """
    Write a .npy file atomically.

    The file is first written to a temporary path, flushed to disk,
    and then atomically renamed to the requested path. This prevents
    interrupted writes from leaving apparently valid-but-corrupt
    checkpoint/output files.
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
    Atomic JSON write used for checkpoint metadata.
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
    Internal checkpoint location.

    This does not change the expected final output structure.
    Checkpoints live in a hidden directory alongside the final outputs.
    """
    return osp.join(save_path, ".mae_checkpoints", str(fileid))


def clear_checkpoint(checkpoint_dir):
    """
    Remove a completed sample's internal checkpoint.
    """
    if osp.isdir(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)


def prepare_checkpoint(checkpoint_dir, metadata):
    """
    Create or validate a checkpoint directory.

    The metadata prevents accidentally resuming an old checkpoint
    generated with incompatible extraction parameters.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    metadata_path = osp.join(checkpoint_dir, "metadata.json")

    if osp.exists(metadata_path):
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                old_metadata = json.load(f)

            if old_metadata != metadata:
                # The checkpoint was produced using different extraction
                # parameters. Discard it rather than mixing feature sets.
                clear_checkpoint(checkpoint_dir)
                os.makedirs(checkpoint_dir, exist_ok=True)

        except (OSError, json.JSONDecodeError):
            # Corrupt metadata means the checkpoint cannot safely be trusted.
            clear_checkpoint(checkpoint_dir)
            os.makedirs(checkpoint_dir, exist_ok=True)

    if not osp.exists(metadata_path):
        atomic_save_json(metadata_path, metadata)


def get_batch_checkpoint_path(checkpoint_dir, batch_index):
    return osp.join(
        checkpoint_dir,
        f"batch_{batch_index:08d}.npy"
    )


def remove_incomplete_batches(checkpoint_dir, start_batch):
    """
    Remove any batches after the first missing batch.

    This guarantees that a checkpoint can never contain a discontinuous
    feature sequence.
    """
    if not osp.isdir(checkpoint_dir):
        return

    for filename in os.listdir(checkpoint_dir):
        if not filename.startswith("batch_") or not filename.endswith(".npy"):
            continue

        try:
            batch_index = int(filename[6:-4])
        except ValueError:
            continue

        if batch_index >= start_batch:
            path = osp.join(checkpoint_dir, filename)
            try:
                os.remove(path)
            except OSError:
                pass


def find_resume_batch(checkpoint_dir, num_batches):
    """
    Find the first batch that has not been successfully checkpointed.

    Only a contiguous sequence beginning at batch 0 is considered complete.
    """
    for batch_index in range(num_batches):
        batch_path = get_batch_checkpoint_path(
            checkpoint_dir,
            batch_index
        )

        if not osp.exists(batch_path):
            # Remove any later files so they cannot be mistakenly included.
            remove_incomplete_batches(checkpoint_dir, batch_index)
            return batch_index

    return num_batches


def load_checkpointed_features(checkpoint_dir, num_batches):
    """
    Load all completed batch features in their original extraction order.
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

        features.append(np.load(batch_path))

    return features


def extract_features_with_resume(
    videos,
    reader,
    batch_size,
    checkpoint_dir,
    checkpoint_metadata
):
    """
    Perform the exact same batched VideoMAE extraction as before, but
    persist every completed batch immediately.

    On restart:
      - existing completed batches are loaded;
      - extraction resumes at the first missing batch;
      - completed batches are never recomputed.

    Returns:
        np.ndarray containing the same concatenated feature matrix that
        the original implementation would have produced.
    """
    num_batches = (len(videos) + batch_size - 1) // batch_size

    prepare_checkpoint(
        checkpoint_dir,
        checkpoint_metadata
    )

    start_batch = find_resume_batch(
        checkpoint_dir,
        num_batches
    )

    # Everything is already extracted for this sample.
    if start_batch == num_batches:
        video_feats = load_checkpointed_features(
            checkpoint_dir,
            num_batches
        )
        return np.concatenate(video_feats, axis=0)

    # Load only the already-completed batches.
    video_feats = load_checkpointed_features(
        checkpoint_dir,
        start_batch
    ) if start_batch > 0 else []

    # Continue from the first missing batch.
    for j in range(start_batch, num_batches):
        video_batch = videos[
            j * batch_size:min((j + 1) * batch_size, len(videos))
        ]

        feats = reader.get_feats(video_batch).cpu().numpy()

        batch_path = get_batch_checkpoint_path(
            checkpoint_dir,
            j
        )

        # Persist the batch BEFORE moving on to the next one.
        atomic_save_npy(
            batch_path,
            feats
        )

        video_feats.append(feats)

    return np.concatenate(video_feats, axis=0)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--anno_root', help='location of tsv files', required=True)
    parser.add_argument('--video_root', help='location of tsv files', required=True)
    parser.add_argument('--save_dir', help='where to save the output', required=True)
    parser.add_argument(
        '--model_name',
        help='ViT model name',
        default='MCG-NJU/videomae-large'
    )
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', help='device to use', default='cpu')
    parser.add_argument('--overlap_size', type=int, default=8)
    parser.add_argument('--mode', nargs='+', type=str)
    parser.add_argument('--nth_layer', type=int, default=-1)
    parser.add_argument('--cache_dir', help='cache dir for model', default=None)
    return parser


def get_iterator(args, mode, save_mode=None):
    batch_size = args.batch_size

    data = np.load(
        os.path.join(args.anno_root, f'{mode}_info.npy'),
        allow_pickle=True
    ).item()

    num = len(data) - 1
    ds_name = osp.split(args.anno_root)[-1]

    # `save_mode` is the actual output directory used by main().
    # For normal datasets mode == save_mode.
    # For How2Sign/NIASL2021 dev, mode maps to val/validation while
    # the output is still written to dev.
    if save_mode is None:
        save_mode = mode

    reader = VideoMAEFeatureReader(
        args.model_name,
        device=args.device,
        overlap_size=args.overlap_size,
        nth_layer=args.nth_layer,
        cache_dir=args.cache_dir
    )

    output_dir = osp.join(
        args.save_dir,
        f'mae_feat_{ds_name}',
        save_mode
    )

    def iterate():
        for i in range(num):
            fname = data[i]['folder']
            fileid = data[i]['fileid']

            # This is the actual expected final output path.
            target = osp.join(
                output_dir,
                f'{fileid}.npy'
            )

            # Existing completed sample -> skip it.
            if osp.exists(target):
                continue

            checkpoint_dir = get_checkpoint_dir(
                output_dir,
                fileid
            )

            if ds_name == 'phoenix2014-T' or ds_name == 'CSL-Daily':
                image_list = get_img_list(
                    ds_name,
                    args.video_root,
                    fname
                )

                if len(image_list) < 16:
                    len_diff = 16 - len(image_list)
                    image_list.extend(
                        [image_list[-1]] * (16 - len(image_list))
                    )

                image_list_chunks = sliding_window_for_list(
                    image_list,
                    window_size=16,
                    overlap_size=args.overlap_size
                )

                videos = []

                for image_list in image_list_chunks:
                    videos.append(
                        [
                            Image.open(image).convert('RGB')
                            for image in image_list
                        ]
                    )

                checkpoint_metadata = {
                    "fileid": str(fileid),
                    "dataset": ds_name,
                    "mode": str(mode),
                    "save_mode": str(save_mode),
                    "model_name": args.model_name,
                    "batch_size": args.batch_size,
                    "overlap_size": args.overlap_size,
                    "nth_layer": args.nth_layer,
                    "num_videos": len(videos),
                }

                feats = extract_features_with_resume(
                    videos=videos,
                    reader=reader,
                    batch_size=batch_size,
                    checkpoint_dir=checkpoint_dir,
                    checkpoint_metadata=checkpoint_metadata
                )

                # Preserve the original generator output.
                yield feats, fileid, None

            else:
                if ds_name == 'How2Sign':
                    start_time = data[i]['original_info']['START_REALIGNED']
                    end_time = data[i]['original_info']['END_REALIGNED']

                    videos = read_video(
                        fname,
                        start_time=start_time,
                        end_time=end_time
                    )

                    if len(videos) > 0:
                        if len(videos) < 16:
                            len_diff = 16 - len(videos)
                            videos.extend(
                                [videos[-1]] * (16 - len(videos))
                            )

                        videos = sliding_window_for_list(
                            videos,
                            window_size=16,
                            overlap_size=args.overlap_size
                        )

                        checkpoint_metadata = {
                            "fileid": str(fileid),
                            "dataset": ds_name,
                            "mode": str(mode),
                            "save_mode": str(save_mode),
                            "model_name": args.model_name,
                            "batch_size": args.batch_size,
                            "overlap_size": args.overlap_size,
                            "nth_layer": args.nth_layer,
                            "num_videos": len(videos),
                            "start_time": str(start_time),
                            "end_time": str(end_time),
                        }

                        feats = extract_features_with_resume(
                            videos=videos,
                            reader=reader,
                            batch_size=batch_size,
                            checkpoint_dir=checkpoint_dir,
                            checkpoint_metadata=checkpoint_metadata
                        )

                        yield feats, fileid, str(start_time)

                    else:
                        # Preserve original behavior for empty videos.
                        yield [], fileid, str(start_time)

    return iterate, num



def count_completed_outputs(save_path):
    """
    Count final feature files that already exist.

    Checkpoints are stored under .mae_checkpoints and therefore are not
    counted. Only completed final .npy outputs contribute to progress.
    """
    if not osp.isdir(save_path):
        return 0

    return sum(
        1
        for filename in os.listdir(save_path)
        if filename.endswith(".npy") and osp.isfile(osp.join(save_path, filename))
    )


def main():
    parser = get_parser()
    args = parser.parse_args()

    mode = ["dev", "test", "train"]

    for m in mode:
        ds_name = osp.split(args.anno_root)[-1]
        fname = f'mae_feat_{ds_name}'

        save_path = osp.join(
            args.save_dir,
            fname,
            m
        )

        os.makedirs(
            save_path,
            exist_ok=True
        )

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

        # `save_mode=m` ensures that the resume check and checkpoint
        # location correspond to the SAME final output directory.
        generator, num = get_iterator(
            args,
            _m,
            save_mode=m
        )

        # Initialize progress from final .npy files that already exist.
        # This makes the progress bar represent total dataset progress
        # across multiple runs rather than only this invocation.
        completed = count_completed_outputs(save_path)
        completed = min(completed, num)

        iterator = generator()

        with tqdm.tqdm(
            iterator,
            total=num,
            initial=completed,
            desc=f"{m}",
            unit="sample"
        ) as progress:
            for vit_feat in progress:
                feats, id, st = vit_feat

            # Preserve the original output naming/procedure.
            # Note that the original `postfix` variable was not actually
            # incorporated into the filename, so that behavior is kept.
            postfix = f'_overlap-{args.overlap_size}'

            if st is not None:
                postfix = f'_{st}{postfix}'

            final_path = osp.join(
                save_path,
                f'{id}.npy'
            )

            # Atomically publish the completed final feature file.
            atomic_save_npy(
                final_path,
                feats
            )

            # Only after the final output exists do we delete checkpoints.
            checkpoint_dir = get_checkpoint_dir(
                save_path,
                id
            )

            clear_checkpoint(
                checkpoint_dir
            )

            # Advance progress only after the final output has been
            # successfully written and its checkpoint has been cleared.
            progress.update(1)


if __name__ == "__main__":
    main()