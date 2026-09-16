import argparse
import os
import os.path as osp
import glob
import tqdm
import torch
import numpy as np
from PIL import Image
from transformers import VivitImageProcessor, VivitModel


import sys
sys.path.append('./')

from preprocess.SP_FT.s2wrapper import forward as multiscale_forward
from preprocess.SP_FT.helpers import read_video, get_img_list

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)


class ViViT_Reader(object):
    def __init__(
        self, 
        model_name='google/vivit-b-16x2-kinetics400', 
        cache_dir=None,
        device='cuda:0'
    ):
        self.device = device
        self.model = VivitModel.from_pretrained(model_name, cache_dir=cache_dir).to(device).eval()
        self.image_processor = VivitImageProcessor.from_pretrained(model_name)

    @torch.no_grad()
    def get_feats(self, video, clip_len=32):
        """
        video: list of PIL.Image or numpy arrays [T, H, W, 3]
        Returns: [num_frames, hidden_dim]
        """
        # convert all frames to PIL if not already
        video = [Image.fromarray(v) if isinstance(v, np.ndarray) else v for v in video]

        # duplicate last frame if video is too short
        total_frames = len(video)
        if total_frames < clip_len:
            video += [video[-1]] * (clip_len - total_frames)
            total_frames = len(video)

        # sample equidistant frames, largest multiple of clip_len
        max_frames = (total_frames // clip_len) * clip_len  # largest multiple of clip_len
        tqdm.tqdm.write(f"T: {total_frames} | max_frames: {max_frames}")
        
        # equidistant indices
        indices = np.linspace(0, total_frames - 1, num=max_frames, dtype=int)
        video = [video[i] for i in indices]

        # process frames
        pixel_values = self.image_processor(video, return_tensors="pt").pixel_values  # [T, C, H, W]

        # ensure batch dimension
        if pixel_values.dim() == 4:
            pixel_values = pixel_values.unsqueeze(0)  # [1, T, C, H, W]

        pixel_values = pixel_values.to(self.device)

        all_feats = []
        T = pixel_values.shape[1]
        for start in range(0, T, clip_len):
            end = start + clip_len
            clip = pixel_values[:, start:end]  # [1, clip_len, C, H, W]
            outputs = self.model(clip)
            last_hidden_states = outputs.last_hidden_state  # [1, tokens, hidden]

            # remove CLS
            last_hidden_states = last_hidden_states[:, 1:, :]

            # reshape: [1, frames, patches, hidden]
            num_tokens = last_hidden_states.shape[1]
            num_patches = 196
            num_frames = num_tokens // num_patches
            last_hidden_states = last_hidden_states.view(1, num_frames, num_patches, -1)

            # mean over patches
            last_hidden_states = last_hidden_states.mean(dim=2)  # [1, frames, hidden]

            all_feats.append(last_hidden_states)

        # concatenate along frames and flatten batch -> [total_frames, hidden]
        all_feats = torch.cat(all_feats, dim=1).view(-1, last_hidden_states.shape[-1])
        return all_feats.cpu().numpy()




def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--anno_root', help='location of tsv files', required=True)
    parser.add_argument('--video_root', help='location of tsv files', required=True)
    parser.add_argument('--device', help='device to use', default='cuda:0')
    parser.add_argument('--s2_mode', default='')
    parser.add_argument('--scales', nargs='+', type=int, help='List of scales', default=[])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--nth_layer', type=int, default=-1)
    parser.add_argument('--cache_dir', help='cache dir for model', default=None)
    
    parser.add_argument('--save_dir', help='where to save the output', required=True)
    parser.add_argument('--model_name', help='ViT model name', default='openai/clip-vit-large-patch14')

    return parser

def get_iterator(args, mode):
    batch_size = args.batch_size
    
    data = np.load(os.path.join(args.anno_root, f'{mode}_info.npy'), allow_pickle=True).item()
    num = len(data) - 1
    ds_name = osp.split(args.anno_root)[-1]
    reader = ViViT_Reader(
        args.model_name, 
        cache_dir=args.cache_dir
    )
    
    def iterate():
        for i in range(num):
            fname = data[i]['folder']

            # if os.path.exists(os.path.join("/home/ahmedubc/projects/aip-lsigal/ahmedubc/datasets/dino_sp/sp/dinov2-giant-imagenet1k-1-layer_feat_phoenix2014-T/train",data[i]['fileid']+"_s2wrapping.npy")):
            #     yield None, None, None

            if ds_name == 'phoenix2014-T' or ds_name == 'CSL-Daily':
                image_list = get_img_list(ds_name, args.video_root, fname)
                videos = [Image.open(image).convert('RGB') for image in image_list]
                
                video_feats = []
                video_feats = reader.get_feats(videos)
                
                yield video_feats, data[i]['fileid'], None
            
            else:
                if ds_name == 'How2Sign':
                    start_time, end_time = data[i]['original_info']['START_REALIGNED'], data[i]['original_info']['END_REALIGNED']
                    videos = read_video(fname, start_time=start_time, end_time=end_time)
            
                if len(videos) > 0:
                    video_feats = []
                    for j in range(0, len(videos), batch_size):
                        video_batch = videos[j:min(j + batch_size, len(videos))]
                        feats = reader.get_feats(video_batch).cpu().numpy()
                        video_feats.append(feats)
                    yield np.concatenate(video_feats, axis=0), data[i]['fileid'], str(start_time)
                else:
                    yield [], data[i]['fileid'], str(start_time)
    
    return iterate, num


def main():
    mode = ["dev", "test", "train"]
    # mode = ["train"]

    for m in mode:
        parser = get_parser()
        args = parser.parse_args()

        ds_name = osp.split(args.anno_root)[-1]
        _model_name = os.path.split(args.model_name)[-1]
        fname = f'{_model_name}_feat_{ds_name}'
        
        os.makedirs(osp.join(args.save_dir, fname, m), exist_ok=True)
    
        if ds_name == 'How2Sign':
            if m == 'dev': _m = 'val'
            else: _m = m
        elif ds_name == 'NIASL2021':
            if m == 'dev': _m = 'validation' 
        else:
            _m = m

        generator, num = get_iterator(args, _m)
        iterator = generator()

        for vit_feat in tqdm.tqdm(iterator, total=num):
            feats, id, st = vit_feat
            if feats is None:
                print("ops")
                continue
            save_path = osp.join(args.save_dir, fname, m)

            tqdm.tqdm.write(str(feats.shape))
            
            postfix = ""
            if args.s2_mode != "":
                postfix = f"_{args.s2_mode}"
            if len(args.scales) == 3:
                postfix = f'{postfix}_large'
            if st is not None:
                postfix = f'_{st}{postfix}'

            tqdm.tqdm.write(f"{feats.shape}------------------------\n\n")
            np.save(osp.join(save_path, f'{id}.npy'), feats)


if __name__ == "__main__":
    main()