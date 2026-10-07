import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import sys
sys.dont_write_bytecode = True

import yaml
import torch
import importlib
import numpy as np
from tqdm import tqdm

from utils.parameters import get_parser
from utils.evaluate import evaluate_results


def import_class(name):
    components = name.rsplit('.', 1)
    mod = importlib.import_module(components[0])
    mod = getattr(mod, components[1])
    return mod


def build_llm_args(args):
    args.llm_args["llm"] = args.llm
    args.llm_args["lang"] = args.dataset_info["language"]
    args.llm_args["include_sp"] = args.include_sp
    args.llm_args["include_pose"] = args.include_pose
    args.llm_args["include_mo"] = args.include_mo
    args.llm_args["sp_hidden_size"] = args.sp_hidden_size
    args.llm_args["pose_hidden_size"] = args.pose_hidden_size
    args.llm_args["mo_hidden_size"] = args.mo_hidden_size
    args.llm_args["contrastive"] = args.contrastive
    args.llm_args["include_ctc"] = args.include_ctc
    args.llm_args["llm_precision"] = getattr(args, "llm_precision", "fp32")
    args.llm_args["gradient_checkpointing"] = getattr(args, "gradient_checkpointing", False)
    args.llm_args["ctc_weight"] = getattr(args, "ctc_weight", 1.0)
    args.llm_args["dataset"] = args.dataset
    args.llm_args["gloss_dict"] = np.load(args.dataset_info['dict_path'], allow_pickle=True).item()
    args.llm_args["num_heads"] = args.temporal_args["num_heads"]
    args.llm_args["num_layers"] = args.temporal_args["num_layers"]
    with open(args.dataset_info["pose_encoder_config_path"], 'r') as f:
        args.llm_args["pose_encoder_cfg"] = yaml.safe_load(f)["model"]["RecognitionNetwork"]
    return args


def build_loader(args, mode="test"):
    ft_dataset = import_class(args.feeder_ft)
    dataset = ft_dataset(
        os.path.join(args.dataset_info["feature_dir"], mode),
        os.path.join(args.dataset_info["sp_path"], mode),
        args.dataset_info["pose_path"],
        os.path.join(args.dataset_info["mo_path"], mode),
        mode=mode,
        include_sp=args.include_sp,
        include_pose=args.include_pose,
        include_mo=args.include_mo,
        dataset=args.dataset,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        collate_fn=dataset.collate_fn,
    )
    return loader


def load_model(args, work_dir):
    model_class = import_class(args.model)
    model = model_class(args.llm_args).to("cuda")
    ckpt_path = os.path.join(work_dir, "best_model.pt")
    state = torch.load(ckpt_path, map_location="cpu")
    clean = {}
    for k, v in state.items():
        clean[k[7:] if k.startswith("module.") else k] = v
    msg = model.load_state_dict(clean)
    print(f"Loaded {ckpt_path}: {msg}")
    model.eval()
    return model


def install_ablation(model, mode, token_log=None):
    original = model._prepare_visual_input

    def ablated(sp_features, pose_features, mo_features,
                sp_lengths, pose_lengths, mo_lengths, glosses):
        visual_tokens, visual_lengths, slr_loss = original(
            sp_features, pose_features, mo_features,
            sp_lengths, pose_lengths, mo_lengths, glosses)
        if token_log is not None:
            token_log.extend(list(visual_lengths))
        bs = visual_tokens.size(0)
        if mode == "zero":
            visual_tokens = torch.zeros_like(visual_tokens)
        elif mode == "shuffle" and bs > 1:
            perm = torch.randperm(bs, device=visual_tokens.device)
            visual_tokens = visual_tokens[perm]
            visual_lengths = [visual_lengths[i] for i in perm.tolist()]
        return visual_tokens, visual_lengths, slr_loss

    model._prepare_visual_input = ablated


@torch.no_grad()
def evaluate(model, loader, max_batches=None):
    gen_strings, ref_strings = [], []
    for i, batch in enumerate(tqdm(loader)):
        sp_features, pose_features, mo_features, glosses, texts, english_texts, sp_lengths, pose_lengths, mo_lengths = batch

        sp_features = sp_features.to("cuda") if sp_features is not None else None
        if pose_features is not None:
            pose_features["keypoint"] = pose_features["keypoint"].to("cuda")
            pose_features["mask"] = pose_features["mask"].to("cuda")
        mo_features = mo_features.to("cuda") if mo_features is not None else None
        sp_lengths = sp_lengths.to("cuda") if sp_lengths is not None else None
        pose_lengths = pose_lengths.to("cuda") if pose_lengths is not None else None
        mo_lengths = mo_lengths.to("cuda") if mo_lengths is not None else None

        gen_str, ref_str = model(
            sp_features, pose_features, mo_features,
            sp_lengths, pose_lengths, mo_lengths,
            glosses, texts, english_texts, warmup=False
        )
        gen_strings.extend(gen_str)
        ref_strings.extend(ref_str)

        if max_batches is not None and i + 1 >= max_batches:
            break

    scores = evaluate_results(gen_strings, ref_strings)
    return scores, gen_strings, ref_strings


def main():
    sparser = get_parser()
    sparser.add_argument("--max-batches", type=int, default=None)
    sparser.add_argument("--ablation-modes", nargs="+", default=["none", "zero", "shuffle"])
    sparser.add_argument("--split", type=str, default="test")

    p = sparser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.load(f, Loader=yaml.FullLoader)
        sparser.set_defaults(**default_arg)

    args = sparser.parse_args()
    with open(f"./configs/{args.dataset}.yaml", 'r') as f:
        args.dataset_info = yaml.load(f, Loader=yaml.FullLoader)

    args.num_warmup_epochs = 0 if not args.contrastive else args.num_warmup_epochs
    args = build_llm_args(args)

    torch.manual_seed(42)
    np.random.seed(42)

    loader = build_loader(args, args.split)
    model = load_model(args, args.work_dir)

    out_path = os.path.join(args.work_dir, "ablation.txt")
    lines = []

    def emit(text):
        print(text)
        lines.append(text)

    emit("=" * 60)
    emit(f"VISUAL ABLATION | split={args.split} | include_sp={args.include_sp} "
         f"include_mo={args.include_mo} include_pose={args.include_pose}")
    emit(f"max_batches={args.max_batches} | ckpt={os.path.join(args.work_dir, 'best_model.pt')}")
    emit("=" * 60)

    original_visual_input = model._prepare_visual_input
    token_log = []
    for mode in args.ablation_modes:
        torch.manual_seed(42)
        model._prepare_visual_input = original_visual_input
        if mode != "none":
            install_ablation(model, mode)
        elif not token_log:
            install_ablation(model, mode, token_log)

        scores, gen_strings, ref_strings = evaluate(model, loader, args.max_batches)
        emit("")
        emit(f"--- MODE: {mode} ({len(gen_strings)} samples) ---")
        emit(f"B1: {scores['BLEU-1']:.4f}, B2: {scores['BLEU-2']:.4f}, "
             f"B3: {scores['BLEU-3']:.4f}, B4: {scores['BLEU-4']:.4f}, "
             f"RG-L: {scores['ROUGE-L_F1']:.4f}")
        for i in range(min(5, len(gen_strings))):
            emit(f"  Generated: {gen_strings[i]}")
            emit(f"  Reference: {ref_strings[i]}")
        with open(out_path, "w") as f:
            f.write("\n".join(lines) + "\n")

    if token_log:
        arr = np.array(token_log)
        emit("")
        emit(f"Visual tokens per sample: min={arr.min()}, max={arr.max()}, "
             f"mean={arr.mean():.2f}, median={np.median(arr):.1f}")

    emit("")
    emit("=" * 60)
    emit("INTERPRETATION")
    emit("  B4 unchanged between 'none' and 'zero'/'shuffle' -> encoder output is IGNORED")
    emit("  B4 drops sharply under 'zero'/'shuffle'              -> encoder output IS USED")
    emit("=" * 60)

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nWrote {out_path}")


if __name__ == '__main__':
    main()
