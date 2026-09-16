import os
import torch
import pickle
import numpy as np
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

class PrecomputedFeatureDataset(Dataset):
    def __init__(self, 
        feature_dir, sp_dir, pose_dir, mo_dir,
        dataset="phoenix2014-T", mode="train",
        include_sp=True, include_pose=True, include_mo=True,
        logger=None
    ):
        self.dataset = dataset
        self.feature_dir = feature_dir
        self.sp_dir = sp_dir
        self.pose_dir = pose_dir
        self.mo_dir = mo_dir

        if not os.path.exists(self.mo_dir):
            self.mo_dir = mo_dir.replace("/" + mode, "")

        self.mode = mode
        self.include_sp = include_sp
        self.include_pose = include_pose
        self.include_mo = include_mo

        assert self.include_sp or self.include_pose or self.include_mo, "At least one feature must be included."
        if logger:
            logger(f"FEATURES: include_sp: {include_sp}, include_pose: {include_pose}, include_mo: {include_mo}")

        self.sp_suffix = "_s2wrapping"
        self.mo_suffix = "_overlap-8"

        self.files = [
            fname.split(".")[0]
            for fname in os.listdir(feature_dir)
        ]
        self.filter_missing()
        if not self.files:
            raise ValueError(f"No .pkl files found in directory: {feature_dir}")

        self.anno_path = f"./preprocess/{dataset}/{mode}_info_ml.npy"
        self.prep_annots(np.load(self.anno_path, allow_pickle=True).item())

        self._pose_init(os.path.join(self.pose_dir, f"{dataset}.{mode}"), mode)


    def _pose_init(self, path, phase):
        self.clip_len = 400
        self.max_length = 300

        self.tmin = 0.5 if phase == 'train' else 1
        self.tmax = 1.5 if phase == 'train' else 1

        self.w = 512 if "csl" in self.dataset else 210
        self.h = 512 if "csl" in self.dataset else 260

        with open(path, "rb") as f:
            self.pose_dict = pickle.load(f)

            if self.dataset == "CSL-Daily":
                new_pose_dict = {}
                for key in self.pose_dict.keys():
                    new_key = f"{phase}/{key}"
                    new_pose_dict[new_key] = self.pose_dict[key]
                self.pose_dict = new_pose_dict
            

    def filter_missing(self):
        to_remove = []
        for f in self.files:
            # if not os.path.exists(os.path.join(self.sp_dir, f + f"{self.sp_suffix}.npy")):
            if not os.path.exists(os.path.join(self.sp_dir, f + f".npy")):
                to_remove.append(f)

        for f in to_remove:
            self.files.remove(f)


    def prep_annots(self, annots):
        if "prefix" in annots: del annots["prefix"]

        langs = ["en", "fr", "es"]
        self.translations, self.text, self.gloss = {}, {}, {}
        for i in range(len(annots)):
            self.translations[annots[i]["fileid"]] = {lang: annots[i][f"{lang}_text"] for lang in langs if f"{lang}_text" in annots[i]}
            self.text[annots[i]["fileid"]] = annots[i]["text"] if "text" in annots[i] else ""
            self.gloss[annots[i]["fileid"]] = annots[i]["gloss"] if "gloss" in annots[i] else annots[i]["label"]

        to_remove = []
        for key in self.translations.keys():
            if key not in self.files:
                to_remove.append(key)

        for key in to_remove:
            del self.translations[key]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # sp_ft_path = os.path.join(self.sp_dir, self.files[idx] + f"{self.sp_suffix}.npy")
        sp_ft_path = os.path.join(self.sp_dir, self.files[idx] + f".npy")
        mo_ft_path = os.path.join(self.mo_dir, self.files[idx] + f".npy")

        sp_ft, pose_ft, mo_ft = None, None, None

        sp_ft = torch.tensor(np.load(sp_ft_path), dtype=torch.float32) if self.include_sp else None
        pose_ft = self.pose_dict[f"{self.mode}/{self.files[idx].replace('.pkl', '')}"]["keypoint"].permute(2, 0, 1).to(torch.float32) if self.include_pose else None
        mo_ft = torch.tensor(np.load(mo_ft_path), dtype=torch.float32) if self.include_mo else None

        gloss = self.gloss[self.files[idx].replace(".pkl", "")]
        text = self.text[self.files[idx].replace(".pkl", "")]

        # icl_text = "\n".join([
        #     f'{self.translations[self.files[idx].replace(".pkl", "")]["en"]}={text}',
        #     f'{self.translations[self.files[idx].replace(".pkl", "")]["fr"]}={text}',
        #     f'{self.translations[self.files[idx].replace(".pkl", "")]["es"]}={text}',
        # ])
        icl_text = "\n".join([
            f'{self.translations[self.files[idx].replace(".pkl", "")]["fr"]}={text}',
            f'{self.translations[self.files[idx].replace(".pkl", "")]["es"]}={text}',
            f'{self.translations[self.files[idx].replace(".pkl", "")]["en"]}={text}',
        ])

        return sp_ft, pose_ft, mo_ft, gloss, text, icl_text

    def rotate_points(self, points, angle):
        center = [0, 0]
        points_centered = points - center
        rotation_matrix = np.array([[np.cos(angle), np.sin(angle)],
                                    [-np.sin(angle), np.cos(angle)]])

        points_rotated = np.dot(points_centered, rotation_matrix.T)
        points_transformed = points_rotated + center

        return points_transformed

    def augment_preprocess_inputs(self, is_train, keypoints=None):
        if is_train == 'train':
            keypoints[:, 0, :, :] /= self.w
            keypoints[:, 1, :, :] = self.h - keypoints[:, 1, :, :]
            keypoints[:, 1, :, :] /= self.h
            keypoints[:, :2, :, :] = (keypoints[:, :2, :, :] - 0.5) / 0.5
            keypoints[:, :2, :, :] = self.random_move(
                keypoints[:, :2, :, :].permute(0, 2, 3, 1).numpy()).permute(0, 3, 1, 2)
        else:
            keypoints[:, 0, :, :] /= self.w
            keypoints[:, 1, :, :] = self.h - keypoints[:, 1, :, :]
            keypoints[:, 1, :, :] /= self.h
            keypoints[:, :2, :, :] = (keypoints[:, :2, :, :] - 0.5) / 0.5
        return keypoints

    def get_selected_index(self, vlen):
        if self.tmin == 1 and self.tmax == 1:
            if vlen <= self.clip_len:
                frame_index = np.arange(vlen)
                valid_len = vlen
            else:
                sequence = np.arange(vlen)
                an = (vlen - self.clip_len) // 2
                en = vlen - self.clip_len - an
                frame_index = sequence[an: -en]
                valid_len = self.clip_len

            if (valid_len % 4) != 0:
                valid_len -= (valid_len % 4)
                frame_index = frame_index[:valid_len]

            assert len(frame_index) == valid_len, (frame_index, valid_len)
            return frame_index, valid_len
            
        min_len = min(int(self.tmin * vlen), self.clip_len)
        max_len = min(self.clip_len, int(self.tmax * vlen))
        selected_len = np.random.randint(min_len, max_len + 1)
        if (selected_len % 4) != 0:
            selected_len += (4 - (selected_len % 4))
        if selected_len <= vlen:
            selected_index = sorted(np.random.permutation(np.arange(vlen))[:selected_len])
        else:
            copied_index = np.random.randint(0, vlen, selected_len - vlen)
            selected_index = sorted(np.concatenate([np.arange(vlen), copied_index]))

        if selected_len <= self.clip_len:
            frame_index = selected_index
            valid_len = selected_len
        else:
            assert False, (vlen, selected_len, min_len, max_len)
        assert len(frame_index) == valid_len, (frame_index, valid_len)
        return frame_index, valid_len

    def random_move(self, data_numpy):
        degrees = np.random.uniform(-15, 15)
        theta = np.radians(degrees)
        p = np.random.uniform(0, 1)
        if p >= 0.5:
            data_numpy = self.rotate_points(data_numpy, theta)
        return torch.from_numpy(data_numpy)


    def pose_collate(self, batch):
        keypoint_batch, src_length_batch = [], []
        for keypoint_sample, length in batch:
            index, valid_len = self.get_selected_index(length)
            if keypoint_sample is not None:
                keypoint_batch.append(torch.stack([keypoint_sample[:, i, :] for i in index], dim=1))
            src_length_batch.append(valid_len)

        max_length = max(src_length_batch)
        padded_sgn_keypoints = []
        for keypoints, len_ in zip(keypoint_batch, src_length_batch):
            if len_ < max_length:
                padding = keypoints[:, -1, :].unsqueeze(1)
                padding = torch.tile(padding, [1, max_length - len_, 1])
                padded_keypoint = torch.cat([keypoints, padding], dim=1)
                padded_sgn_keypoints.append(padded_keypoint)
            else:
                padded_sgn_keypoints.append(keypoints)

        keypoints = torch.stack(padded_sgn_keypoints, dim=0)
        keypoints = self.augment_preprocess_inputs(self.mode, keypoints)
        src_length_batch = torch.tensor(src_length_batch)
        new_src_lengths = (((src_length_batch - 1) / 2) + 1).long()
        new_src_lengths = (((new_src_lengths - 1) / 2) + 1).long()
        max_len = max(new_src_lengths)
        mask = torch.zeros(new_src_lengths.shape[0], 1, max_len)
        for i in range(new_src_lengths.shape[0]):
            mask[i, :, :new_src_lengths[i]] = 1
        mask = mask.to(torch.bool)
        src_input = {}

        src_input['keypoint'] = keypoints
        src_input['mask'] = mask
        return src_input

    
    def collate_fn(self, batch):
        sp_ft, pose_ft, mo_ft, glosses, texts, icl_text = zip(*batch)  # unpack the batch

        pose_prepped = self.pose_collate([(p, p.shape[1]) for p in pose_ft if p is not None]) if pose_ft[0] is not None else None

        sp_lengths = torch.tensor([t.shape[0] for t in sp_ft], dtype=torch.long) if sp_ft[0] is not None else None
        pose_lengths = torch.tensor([t.shape[1] for t in pose_prepped["keypoint"]], dtype=torch.long) if pose_ft[0] is not None else None
        mo_lengths = torch.tensor([t.shape[0] for t in mo_ft], dtype=torch.long) if mo_ft[0] is not None else None

        padded_sp_ft = pad_sequence(sp_ft, batch_first=True) if sp_ft[0] is not None else None
        padded_mo_ft = pad_sequence(mo_ft, batch_first=True) if mo_ft[0] is not None else None

        return padded_sp_ft, pose_prepped, padded_mo_ft, glosses, texts, icl_text, sp_lengths, pose_lengths, mo_lengths


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    
    mode="dev"
    dataset = PrecomputedFeatureDataset(
        f"/home/ahmedubc/projects/aip-lsigal/ahmedubc/datasets/phoenix2014T/fullFrame-256x256px/{mode}",
        f"/home/ahmedubc/projects/aip-lsigal/ahmedubc/datasets/features2/{mode}",
        f"/home/ahmedubc/projects/aip-lsigal/ahmedubc/datasets/pose/",
        f"/home/ahmedubc/projects/aip-lsigal/ahmedubc/datasets/spamo/pomo/mae_feat_Phoenix14T/{mode}",
        f"/home/ahmedubc/projects/aip-lsigal/ahmedubc/datasets/spamo/posp/clip-vit-large-patch14_feat_Phoenix14T/{mode}",
        mode=mode
    )
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=dataset.collate_fn)

    for sp_features, pose_features, mo_features, glosses, texts, icl_text, sp_lengths, pose_lengths, mo_lengths in dataloader:
        print(sp_features.shape)
        print(pose_features["keypoint"].shape)
        print(mo_features.shape)
        print(glosses)
        print(texts)
        print(icl_text[0])
        print(sp_lengths)
        print(pose_lengths)
        print(mo_lengths)

        break