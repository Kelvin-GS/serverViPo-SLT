import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import sys
sys.dont_write_bytecode = True

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler
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

# util functions ========================================================================================
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def import_class(name):
    components = name.rsplit('.', 1)
    mod = importlib.import_module(components[0])
    mod = getattr(mod, components[1])
    return mod

def init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

# util functions ========================================================================================


# main functions ========================================================================================

def setup_work_dir(arg, is_main, WORK_DIR_PATH):
    global main_logger
    global pred_logger

    arg.optimizer_args['num_epoch'] = arg.num_epoch
    
    if is_main:
        main_logger = Logger(os.path.join(WORK_DIR_PATH, 'log.txt'))
        pred_logger = Logger(os.path.join(WORK_DIR_PATH, 'preds.txt'))

        for key, value in vars(arg).items():
            main_logger(f"{key}: {value}")

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
    arg.llm_args["gloss_dict"] = np.load(arg.dataset_info['dict_path'], allow_pickle=True).item()
    arg.llm_args["dataset"] = arg.dataset
    arg.llm_args["num_heads"] = arg.temporal_args["num_heads"]
    arg.llm_args["num_layers"] = arg.temporal_args["num_layers"]
    with open(arg.dataset_info["pose_encoder_config_path"], 'r') as f:
        arg.llm_args["pose_encoder_cfg"] = yaml.safe_load(f)["model"]["RecognitionNetwork"]

    ft_dataset = import_class(arg.feeder_ft)
    shutil.copy2(inspect.getfile(ft_dataset), WORK_DIR_PATH)
    shutil.copy2("./main.py", WORK_DIR_PATH)

    return arg

def setup_data(arg, world_size, rank, is_main):
    steps_info = {}
    if arg.dataset == 'CSL':
        dataset_list = zip(["train", "dev"], [True, False])
    elif 'phoenix' in arg.dataset:
        dataset_list = zip(["train", "dev", "test"], [True, False, False]) 
    elif arg.dataset == 'CSL-Daily':
        dataset_list = zip(["train", "dev", "test"], [True, False, False, False])

    data_loader, sampler = {}, {}
    ft_dataset = import_class(arg.feeder_ft)
    for idx, (mode, train_flag) in enumerate(dataset_list):
        dataset = ft_dataset(
            os.path.join(arg.dataset_info["feature_dir"], mode), 
            os.path.join(arg.dataset_info["sp_path"], mode), 
            arg.dataset_info["pose_path"], 
            os.path.join(arg.dataset_info["mo_path"], mode), 
            mode=mode,
            include_sp=arg.include_sp,
            include_pose=arg.include_pose,
            include_mo=arg.include_mo,
            dataset=arg.dataset
        )
        sampler[mode] = DistributedSampler(dataset) if train_flag else None

        batch_size = arg.batch_size 
        batch_size = (batch_size // world_size) if mode == "train" else arg.test_batch_size
        data_loader[mode] = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            drop_last=train_flag,
            num_workers=0,
            collate_fn=dataset.collate_fn,
            worker_init_fn=init_fn,
            sampler=sampler[mode]
        )

        steps_info[mode] = (len(data_loader[mode]) * arg.num_epoch)

    return data_loader, steps_info, sampler

def setup_model_comps(arg, rank, world_size, data_loader, steps_info, is_main, WORK_DIR_PATH):
    model_class = import_class(arg.model) 
    model = model_class(arg.llm_args).to(torch.device(f"cuda:{rank}"))
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    shutil.copy2(inspect.getfile(model_class), WORK_DIR_PATH)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=arg.optimizer_args['base_lr']*world_size,
        eps=1e-8, 
        weight_decay=0.01, 
        betas=(0.9, 0.98)
    )

    if arg.score_dependent_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max' if arg.monitor_bleu else 'min',
            factor=arg.optimizer_args['gamma'],
            patience=arg.scheduler_patience,
        )
        if is_main: main_logger("Using score-dependent learning rate scheduler.")

    else:
        scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=steps_info["train"],
        )
        if is_main: main_logger("Using cosine learning rate scheduler.")

    return model, optimizer, scheduler

# main functions ========================================================================================



def main(rank, world_size, arg, WORK_DIR_PATH):
    global main_logger
    global pred_logger

    torch.manual_seed(42)
    np.random.seed(42)

    is_main = rank == 0
    setup(rank, world_size)

    arg = setup_work_dir(arg, is_main, WORK_DIR_PATH)
    data_loader, steps_info, sampler = setup_data(arg, world_size, rank, is_main)
    model, optimizer, scheduler = setup_model_comps(arg, rank, world_size, data_loader, steps_info, is_main, WORK_DIR_PATH)

    if is_main:
        main_logger("\n")
        main_logger("=" * 20 + " STARTING " + "=" * 20)
        main_logger(model)

    train_loader = data_loader["train"]
    val_loader = data_loader["dev"]
    test_loader = data_loader["test"]

    warmup, patience = False, arg.patience
    least_loss, best_b4 = 1000, 0
    sampler = sampler["train"]
    dist.barrier()
    for epoch in range(arg.num_epoch + arg.num_warmup_epochs):
        sampler.set_epoch(epoch)

        total_loss = 0.0
        start_time = time.time()
        if epoch < arg.num_warmup_epochs and not warmup:
            warmup = True
            if is_main:main_logger("WARMUP ---------------------------------")
            model.module.freeze_all_parameters()
        elif epoch == arg.num_warmup_epochs:
            warmup = False
            if is_main:main_logger("FULL -----------------------------------")
            model.module.unfreeze_lora_parameters()

        model.train()
        # for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{arg.num_epoch + arg.num_warmup_epochs}", ncols=100)):
        for i, batch in enumerate(train_loader):
            sp_features, pose_features, mo_features, glosses, texts, icl_text, sp_lengths, pose_lengths, mo_lengths = batch

            sp_features = sp_features.to(rank) if sp_features is not None else None
            if arg.include_pose:
                pose_features["keypoint"] = pose_features["keypoint"].to(rank) if pose_features is not None else None
                pose_features["mask"] = pose_features["mask"].to(rank) if pose_features is not None else None
            mo_features = mo_features.to(rank) if mo_features is not None else None
            sp_lengths = sp_lengths.to(rank) if sp_lengths is not None else None
            pose_lengths = pose_lengths.to(rank) if pose_lengths is not None else None
            mo_lengths = mo_lengths.to(rank) if mo_lengths is not None else None

            loss = model(
                sp_features, pose_features, mo_features,
                sp_lengths, pose_lengths, mo_lengths,
                glosses, texts, icl_text, warmup=warmup
            )

            loss = loss / arg.accumulate_grad_batches
            total_loss += loss.item()

            loss.backward()

            # update only every arg.accumulate_grad_batches iterations
            if ((i + 1) % arg.accumulate_grad_batches == 0) or (i == len(train_loader) - 1):
                optimizer.step()
                optimizer.zero_grad()
                
                if not arg.score_dependent_scheduler and not warmup:
                    scheduler.step()

        if is_main: 
            main_logger(f"Epoch {(epoch + 1)-arg.num_warmup_epochs} - Loss: {total_loss / len(train_loader):.4f}")

        scores = None
        if not warmup:
            gen_strings, ref_strings = [], []
            model.eval()
            # for i, batch in enumerate(tqdm(val_loader, desc="Validation", ncols=100)):
            for i, batch in enumerate(val_loader):
                sp_features, pose_features, mo_features, glosses, texts, icl_text, sp_lengths, pose_lengths, mo_lengths = batch

                sp_features = sp_features.to(rank) if sp_features is not None else None
                if arg.include_pose:
                    pose_features["keypoint"] = pose_features["keypoint"].to(rank) if pose_features is not None else None
                    pose_features["mask"] = pose_features["mask"].to(rank) if pose_features is not None else None
                mo_features = mo_features.to(rank) if mo_features is not None else None
                sp_lengths = sp_lengths.to(rank) if sp_lengths is not None else None
                pose_lengths = pose_lengths.to(rank) if pose_lengths is not None else None
                mo_lengths = mo_lengths.to(rank) if mo_lengths is not None else None

                with torch.no_grad():
                    gen_str, ref_str = model(
                        sp_features, pose_features, mo_features,
                        sp_lengths, pose_lengths, mo_lengths,
                        glosses, texts, icl_text, warmup=warmup
                    )
                gen_strings.extend(gen_str)
                ref_strings.extend(ref_str)

            if is_main:
                pred_logger("=" * 10 + f"EPOCH {(epoch + 1)-arg.num_warmup_epochs} - Generated and Reference Strings" + "=" * 10)
                for i in range(min(5, len(gen_strings))):
                    pred_logger(f"Generated: {gen_strings[i]}")
                    pred_logger(f"Reference: {ref_strings[i]}")
                    pred_logger("-" * 50)
                pred_logger("\n")

            scores = evaluate_results(gen_strings, ref_strings)
            if is_main: main_logger(f"Validation Scores > B1: {scores['BLEU-1']:.4f}, B2: {scores['BLEU-2']:.4f}, B3: {scores['BLEU-3']:.4f}, B4: {scores['BLEU-4']:.4f}, RG-L: {scores['ROUGE-L_F1']:.4f}")

            if arg.monitor_bleu:
                if scores['BLEU-4'] > best_b4:
                    best_b4 = scores['BLEU-4']
                    patience = arg.patience
                    if is_main: main_logger(f"New best model found at epoch {epoch + 1} with BLEU-4 {best_b4:.4f}. Saving model...")
                    torch.save(model.state_dict(), WORK_DIR_PATH + '/best_model.pt')
                elif epoch > arg.num_warmup_epochs + arg.epochs_before_lr_decay:
                    patience -= 1
                    if patience <= 0:
                        if is_main: main_logger("Early stopping triggered. No improvement in validation BLEU-4.")
                        break

                if arg.score_dependent_scheduler:
                    scheduler.step(scores['BLEU-4'])

            else:
                if least_loss > total_loss/len(train_loader):
                    least_loss = total_loss/len(train_loader)
                    patience = arg.patience
                    if is_main: main_logger(f"New best model found at epoch {epoch + 1} with loss {least_loss:.4f}. Saving model...")
                    torch.save(model.state_dict(), WORK_DIR_PATH + '/best_model.pt')
                elif epoch > arg.num_warmup_epochs + arg.epochs_before_lr_decay:
                    patience -= 1
                    if patience <= 0:
                        if is_main: main_logger("Early stopping triggered. No improvement in validation loss.")
                        break

                if arg.score_dependent_scheduler:
                    scheduler.step(total_loss / len(train_loader))
        

        current_lr = optimizer.param_groups[0]['lr']
        if is_main: main_logger(f"Learning Rate: {current_lr} - Patience: [{patience}/{arg.patience}]")
        elapsed_time_mins = (time.time() - start_time) / 60
        if is_main: main_logger(f"Elapsed Time: {elapsed_time_mins:.2f} mins")
        if is_main: main_logger("\n")

    cleanup()

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

    WORK_DIR_PATH = args.work_dir # + '_' + datetime.now().strftime("D%d_H%H_M%M_S%S")
    if not WORK_DIR_PATH.endswith('/'): WORK_DIR_PATH = WORK_DIR_PATH + '/'
    os.makedirs(WORK_DIR_PATH, exist_ok=True)

    args.num_warmup_epochs = 0 if not args.contrastive else args.num_warmup_epochs
    assert args.include_sp or args.include_pose or args.include_mo, "At least one modality should be included."

    if args.mode == "train":
        shutil.copy2(p.config, WORK_DIR_PATH)
        shutil.copy2(f"./configs/{args.dataset}.yaml", WORK_DIR_PATH)
        shutil.copy2(f"./modules/pose_encoder/model.py", WORK_DIR_PATH)
        shutil.copy2(f"./modules/ff.py", WORK_DIR_PATH)

    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(main, args=(world_size, args, WORK_DIR_PATH), nprocs=world_size, join=True)