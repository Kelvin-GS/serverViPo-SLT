import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import sys
sys.dont_write_bytecode = True

import torch.distributed as dist
from tqdm import tqdm

import time
import yaml
import torch
import importlib
import faulthandler
import numpy as np
import shutil
import inspect
from utils.logger import Logger
from utils.parameters import get_parser
from transformers import get_linear_schedule_with_warmup

faulthandler.enable()
from datetime import datetime
from utils.evaluate import evaluate_results


main_logger, pred_logger = None, None


def import_class(name):
    components = name.rsplit('.', 1)
    mod = importlib.import_module(components[0])
    mod = getattr(mod, components[1])
    return mod

def init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def setup_work_dir(arg, WORK_DIR_PATH):
    global main_logger
    global pred_logger

    arg.optimizer_args['num_epoch'] = arg.num_epoch
    
    main_logger = Logger(os.path.join(WORK_DIR_PATH, 'log.txt'))
    pred_logger = Logger(os.path.join(WORK_DIR_PATH, 'preds.txt'))

    arg.llm_args["llm"] = arg.llm
    arg.llm_args["lang"] = arg.dataset_info["language"]
    arg.llm_args["include_sp"] = arg.include_sp
    arg.llm_args["include_pose"] = arg.include_pose
    arg.llm_args["include_mo"] = arg.include_mo
    arg.llm_args["sp_hidden_size"] = arg.sp_hidden_size
    arg.llm_args["pose_hidden_size"] = arg.pose_hidden_size
    arg.llm_args["mo_hidden_size"] = arg.mo_hidden_size
    arg.llm_args["contrastive"] = arg.contrastive
    arg.llm_args["include_ctc"] = arg.include_ctc
    arg.llm_args["dataset"] = arg.dataset
    arg.llm_args["gloss_dict"] = np.load(arg.dataset_info['dict_path'], allow_pickle=True).item()
    arg.llm_args["num_heads"] = arg.temporal_args["num_heads"]
    arg.llm_args["num_layers"] = arg.temporal_args["num_layers"]
    with open(arg.dataset_info["pose_encoder_config_path"], 'r') as f:
        arg.llm_args["pose_encoder_cfg"] = yaml.safe_load(f)["model"]["RecognitionNetwork"]

    ft_dataset = import_class(arg.feeder_ft)

    return arg

def setup_data(arg):
    steps_info = {}
    if arg.dataset == 'CSL':
        dataset_list = zip(["train", "dev"], [True, False])
    elif 'phoenix' in arg.dataset:
        dataset_list = zip(["test"], [False]) 
    elif arg.dataset == 'CSL-Daily':
        dataset_list = zip(["test"], [False])

    data_loader = {}
    ft_dataset = import_class(arg.feeder_ft)
    for idx, (mode, train_flag) in enumerate(dataset_list):
        dataset = ft_dataset(
            os.path.join(arg.dataset_info["feature_dir"], mode), 
            os.path.join(arg.dataset_info["sp_path"], mode), 
            arg.dataset_info["pose_path"], 
            os.path.join(arg.dataset_info["mo_path"], mode), 
            dataset=arg.dataset,
            mode=mode,
            include_sp=arg.include_sp,
            include_pose=arg.include_pose,
            include_mo=arg.include_mo,
        )

        data_loader[mode] = torch.utils.data.DataLoader(
            dataset,
            batch_size=arg.test_batch_size,
            drop_last=train_flag,
            num_workers=0,
            collate_fn=dataset.collate_fn,
            worker_init_fn=init_fn,
        )

        steps_info[mode] = (len(data_loader[mode]) * arg.num_epoch)

    return data_loader, steps_info

def setup_model_comps(arg, data_loader, steps_info, WORK_DIR_PATH):
    model_class = import_class(arg.model) 
    model = model_class(arg.llm_args).to("cuda")

    return model


def main(arg, WORK_DIR_PATH):
    global main_logger
    global pred_logger

    torch.manual_seed(42)
    np.random.seed(42)


    arg = setup_work_dir(arg, WORK_DIR_PATH)
    data_loader, steps_info = setup_data(arg)
    model = setup_model_comps(arg, data_loader, steps_info, WORK_DIR_PATH)

    test_loader = data_loader["test"]

    warmup, patience = False, arg.patience
    
    main_logger("\n\nTESTING --------------------------------\n")

    w = torch.load(WORK_DIR_PATH + '/best_model.pt')
    new_w = {}
    for k, v in w.items():
        if k.startswith('module.'):
            new_w[k[7:]] = v
        else:
            new_w[k] = v
    w = new_w
    msg = model.load_state_dict(w)
    print(msg)

    gen_strings, ref_strings = [], []
    model.eval()

    for i, batch in enumerate(tqdm(test_loader)):
        sp_features, pose_features, rp_features, glosses, texts, icl_text, sp_lengths, pose_lengths, rp_lengths = batch

        sp_features = sp_features.to("cuda") if sp_features is not None else None
        if arg.include_pose:
            pose_features["keypoint"] = pose_features["keypoint"].to("cuda") if pose_features is not None else None
            pose_features["mask"] = pose_features["mask"].to("cuda") if pose_features is not None else None
        rp_features = rp_features.to("cuda") if rp_features is not None else None
        sp_lengths = sp_lengths.to("cuda") if sp_lengths is not None else None
        pose_lengths = pose_lengths.to("cuda") if pose_lengths is not None else None
        rp_lengths = rp_lengths.to("cuda") if rp_lengths is not None else None

        with torch.no_grad():
            gen_str, ref_str = model(
                sp_features, pose_features, rp_features,
                sp_lengths, pose_lengths, rp_lengths,
                glosses, texts, icl_text, warmup=warmup
            )
        gen_strings.extend(gen_str)
        ref_strings.extend(ref_str)

    pred_logger("=" * 10 + f"Generated and Reference Strings" + "=" * 10)
    for i in range(len(gen_strings)):
        pred_logger(f"Generated: {gen_strings[i]}")
        pred_logger(f"Reference: {ref_strings[i]}")
        pred_logger("-" * 50)
    pred_logger("\n")

    scores = evaluate_results(gen_strings, ref_strings)
    main_logger(f"Test Scores > B1: {scores['BLEU-1']:.4f}, B2: {scores['BLEU-2']:.4f}, B3: {scores['BLEU-3']:.4f}, B4: {scores['BLEU-4']:.4f}, RG-L: {scores['ROUGE-L_F1']:.4f}")


if __name__ == '__main__':
    sparser = get_parser()
    p = sparser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            try:
                default_arg = yaml.load(f, Loader=yaml.FullLoader)
            except AttributeError:
                default_arg = yaml.load(f)

        key = vars(p).keys()
        sparser.set_defaults(**default_arg)

    args = sparser.parse_args()
    with open(f"./configs/{args.dataset}.yaml", 'r') as f:
        args.dataset_info = yaml.load(f, Loader=yaml.FullLoader)

    WORK_DIR_PATH = args.work_dir
    if not os.path.exists(WORK_DIR_PATH):
        print(f"Work directory {WORK_DIR_PATH} does not exist.")
        exit()
    if not WORK_DIR_PATH.endswith('/'): WORK_DIR_PATH = WORK_DIR_PATH + '/'

    args.num_warmup_epochs = 0 if not args.contrastive else args.num_warmup_epochs
    assert args.include_sp or args.include_pose or args.include_mo, "At least one modality should be included."

    print(args.dataset)

    main(args, WORK_DIR_PATH)