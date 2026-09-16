import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.tconv import TemporalConv
from modules.ff import IMTM_CMTM
from modules.pose_encoder.model import PoseEncoder
from torch.distributed.nn.functional import all_gather
from utils.helpers import safe_derangement, clip_loss

from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, T5ForConditionalGeneration, T5Config
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast, AutoModelForSeq2SeqLM
from peft import LoraConfig, get_peft_model, TaskType

import torch
import torch.nn as nn
import math

class EDM(nn.Module):
    def __init__(self, arg):
        super(EDM, self).__init__()
        self.arg = arg

        self.lora_r = arg["lora_r"]
        self.lora_alpha = arg["lora_alpha"]
        self.lora_dropout = arg["lora_dropout"]
        self.max_txt_len = arg["max_txt_len"]
        self.prompt = f'{arg["prompt"]} {arg["lang"]}'
        self.tuning_mode = arg["tuning_mode"]
        self.include_sp = arg["include_sp"]
        self.include_pose = arg["include_pose"]
        self.include_mo = arg["include_mo"]
        self.sp_hidden_size = arg["sp_hidden_size"]
        self.pose_hidden_size = arg["pose_hidden_size"]
        self.mo_hidden_size = arg["mo_hidden_size"]
        self.gloss_dict = arg["gloss_dict"]
        self.contrastive = arg["contrastive"]
        self.num_classes = len(self.gloss_dict) + 1
        self.pose_encoder_cfg = arg["pose_encoder_cfg"]
        self.hidden_size = arg["llm_hidden_size"]
        self.include_ctc = arg["include_ctc"]
        self.llm_name = arg["llm"]
        self.dataset = arg["dataset"]
        self.bart_lang_heads = {"CSL-Daily": "zh_CN", "phoenix2014-T": "de_DE"}

        self.gloss_dict = {k:v[0] for k,v in self.gloss_dict.items()}
        self._prepare_llm(arg["llm"])

        if arg["apply_lora"]:
            self._apply_lora()

        print("PROMPT:", self.prompt)

        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params / 1_000_000:.2f}M, Trainable parameters: {trainable_params / 1_000_000:.2f}M")

    def _prepare_llm(self, model: str) -> None:
        if "bart" in model.lower():
            self.llm = MBartForConditionalGeneration.from_pretrained(model)
            self.tokenizer = MBart50TokenizerFast.from_pretrained(
                model, 
                cache_dir="./data/models",
                max_length=self.max_txt_len,
                src_lang="en_XX",
                tgt_lang=self.bart_lang_heads[self.dataset],
            )
        else:
            config = T5Config.from_pretrained(model, cache_dir="./data/models")
            config.output_hidden_states = True   # ensures encoder hidden states
            config.return_dict = True
            self.llm = T5ForConditionalGeneration.from_pretrained(model, config=config)
            self.tokenizer = AutoTokenizer.from_pretrained(
                model, 
                cache_dir="./data/models",
                max_length=self.max_txt_len,
            )

        self.sp_proj = nn.Linear(self.sp_hidden_size, self.hidden_size) if self.include_sp else None
        self.pose_proj = nn.Linear(self.pose_hidden_size, self.hidden_size) if self.include_pose else None
        self.mo_proj = nn.Linear(self.mo_hidden_size, self.hidden_size) if self.include_mo else None
        self.fusion_proj = nn.Sequential(
            nn.Linear(self.hidden_size, self.llm.config.d_model),
            nn.GELU(),
            nn.Linear(self.llm.config.d_model, self.llm.config.d_model),
        )
        if self.include_pose:
            self.pose_encoder = PoseEncoder(cfg=self.pose_encoder_cfg)

            w_path = f"./ckpt/{self.dataset}.pth"
            weights = torch.load(w_path, map_location='cpu')["model"]

            msg = self.pose_encoder.load_state_dict(weights, strict=False)
            print(f"Pose encoder weights loaded from {w_path}, msg: {msg}")

            for param in self.pose_encoder.parameters():
                param.requires_grad = False

        # self.temporal_encoder = TemporalConv(self.hidden_size, self.hidden_size, conv_type=2)
        self.temporal_encoder = IMTM_CMTM(self.hidden_size, self.hidden_size, conv_type=2, num_heads=self.arg["num_heads"], num_layers=self.arg["num_layers"])
        self.logit_scale = nn.Parameter(torch.tensor(2.6592))

        if self.include_ctc:
            self.clf_sp = nn.Linear(self.hidden_size, self.num_classes) if self.include_sp else None
            self.clf_mo = nn.Linear(self.hidden_size, self.num_classes) if self.include_mo else None
            self.clf_pose = nn.Linear(self.hidden_size, self.num_classes) if self.include_pose else None

            self.pre_llm_sp = nn.Sequential(
                nn.Linear(self.num_classes, self.hidden_size),
                nn.GELU(),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.GELU(),
                nn.Linear(self.hidden_size, self.hidden_size),
            ) if self.include_sp else None
            self.pre_llm_mo = nn.Sequential(
                nn.Linear(self.num_classes, self.hidden_size),
                nn.GELU(),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.GELU(),
                nn.Linear(self.hidden_size, self.hidden_size),
            ) if self.include_mo else None
            self.pre_llm_pose = nn.Sequential(
                nn.Linear(self.num_classes, self.hidden_size),
                nn.GELU(),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.GELU(),
                nn.Linear(self.hidden_size, self.hidden_size),
            ) if self.include_pose else None

            self.ctc_loss = torch.nn.CTCLoss(reduction='none', zero_infinity=False)

    def freeze_all_parameters(self):
        for param in self.llm.parameters():
            param.requires_grad = False

        print("All parameters are frozen for warmup.")

    def unfreeze_lora_parameters(self):
        if self.arg["apply_lora"]:
            for name, param in self.llm.named_parameters():
                if "lora" in name:
                    param.requires_grad = True
        else:
            for param in self.llm.parameters():
                param.requires_grad = True

        print("LoRA parameters are unfrozen for training.")
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params / 1_000_000:.2f}M, Trainable parameters: {trainable_params / 1_000_000:.2f}M")
        print("\n\n")

    def _apply_lora(self) -> None:
        lora_config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"] if "bart" in self.llm_name.lower() else ["q", "v"],
            lora_dropout=self.lora_dropout,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM
        )
        self.llm = get_peft_model(self.llm, lora_config)
        print("LoRA adapter applied to model.")

    def create_mask(self, seq_lengths: list[int], device="cpu"):
        lengths = torch.tensor(seq_lengths, dtype=torch.int32, device=device)
        max_len = lengths.max().item()
        range_row = torch.arange(max_len, dtype=torch.int32, device=device).expand(len(lengths), -1)
        lengths = lengths.unsqueeze(1)
        mask = range_row < lengths  # shape: (batch_size, max_len)
        return mask

    def prep_cont_tensors(self, visual_ft, labels):
        output_tokens = self.tokenizer(
            labels,
            padding="longest",
            return_tensors="pt",
        ).to(visual_ft.device)
        
        if "bart" in self.llm_name.lower():
            text_embeds = self.llm.model.model.encoder.embed_tokens(output_tokens.input_ids)
        else:
            text_embeds = self.llm.encoder.embed_tokens(output_tokens.input_ids)
        
        image_embeds = visual_ft.mean(1)  # global pooling
        text_embeds = text_embeds.mean(1)  # global pooling
        
        image_embeds = F.normalize(image_embeds, dim=-1)
        text_embeds = F.normalize(text_embeds, dim=-1)

        return image_embeds, text_embeds

    def calc_ctc_loss(self, logits, gloss_labels, ft_lgt, label_lgt):
        return self.ctc_loss(
            logits.permute(1,0,2).contiguous().log_softmax(-1),
            gloss_labels,
            ft_lgt,
            label_lgt
        ).mean()


    def _prepare_visual_input(self, sp_features, pose_features, mo_features, sp_lengths, pose_lengths, mo_lengths, glosses):
        bs = sp_features.shape[0] if self.include_sp else mo_features.shape[0]

        if self.include_sp:
            sp_features = self.sp_proj(sp_features)

        if self.include_mo:
            mo_features = self.mo_proj(mo_features)
            
        if self.include_pose:
            pose_features = self.pose_encoder(pose_features)
            pose_lengths = torch.tensor([pose_features.shape[1]]*pose_features.shape[0], dtype=torch.long, device=pose_features.device)
            pose_features = self.pose_proj(pose_features)

        slr_loss, mlp_sp, mlp_mo, mlp_pose = 0, 0, 0, 0
        if self.include_ctc:
            logits_sp = self.clf_sp(sp_features) if self.include_sp else None
            logits_mo = self.clf_mo(mo_features) if self.include_mo else None
            logits_pose = self.clf_pose(pose_features) if self.include_pose else None

            if self.include_ctc:
                glosses = [gloss.split(" ") for gloss in glosses]
                gloss_labels = []
                label_lengths = []
                for gloss in glosses:
                    label = [self.gloss_dict[g] for g in gloss]
                    gloss_labels.extend(label)
                    label_lengths.append(len(label))

                gloss_labels = torch.LongTensor(gloss_labels)
                label_lengths = torch.LongTensor(label_lengths)

                slr_loss += self.calc_ctc_loss(logits_sp, gloss_labels, sp_lengths, label_lengths) if logits_sp is not None else 0
                slr_loss += self.calc_ctc_loss(logits_mo, gloss_labels, mo_lengths, label_lengths) if logits_mo is not None else 0
                slr_loss += self.calc_ctc_loss(logits_pose, gloss_labels, pose_lengths, label_lengths) if logits_pose is not None else 0

            mlp_sp = self.pre_llm_sp(logits_sp) if self.include_sp else None
            mlp_mo = self.pre_llm_mo(logits_mo) if self.include_mo else None
            mlp_pose = self.pre_llm_pose(logits_pose) if self.include_pose else None

        joint_visual, visual_lengths = [], []
        for i in range(bs):
            parts = []

            if self.include_ctc:
                if self.include_sp:
                    parts.append(sp_features[i, :sp_lengths[i]] + mlp_sp[i, :sp_lengths[i]])
                if self.include_pose:
                    parts.append(pose_features[i, :pose_lengths[i]] + mlp_pose[i, :pose_lengths[i]])
                if self.include_mo:
                    parts.append(mo_features[i, :mo_lengths[i]] + mlp_mo[i, :mo_lengths[i]])
            else:
                if self.include_sp:
                    parts.append(sp_features[i, :sp_lengths[i]])
                if self.include_pose:
                    parts.append(pose_features[i, :pose_lengths[i]])
                if self.include_mo:
                    parts.append(mo_features[i, :mo_lengths[i]])

            vis_tokens = torch.cat(parts, dim=0)
            joint_visual.append(vis_tokens)
            visual_lengths.append(vis_tokens.size(0))  # store actual length

        joint_visual = pad_sequence(joint_visual, batch_first=True)  # (B, T, D)
        visual_conv_outputs = self.temporal_encoder(
            joint_visual.permute(0, 2, 1), 
            torch.tensor(visual_lengths, device=joint_visual.device)
        )

        visual_outputs = visual_conv_outputs['visual_feat'].permute(1, 0, 2)
        visual_lengths = visual_conv_outputs['feat_len'].to(torch.int).tolist()  # final lengths after encoder
        visual_tokens = self.fusion_proj(visual_outputs)

        return visual_tokens, visual_lengths, slr_loss


    def _prepare_joint_input(self, visual_tokens, visual_lengths, prompts):
        bs = visual_tokens.size(0)

        input_tokens = self.tokenizer(
            prompts, 
            padding="longest", 
            truncation=True, 
            return_tensors="pt"
        ).to(visual_tokens.device)

        with torch.no_grad():
            if "bart" in self.llm_name.lower():
                prompt_embeds = self.llm.model.model.encoder.embed_tokens(input_tokens.input_ids)
            else:
                prompt_embeds = self.llm.encoder.embed_tokens(input_tokens.input_ids)
            
        prompt_lengths = input_tokens.attention_mask.sum(dim=1).tolist()

        joint_embeds, visual_embeds, new_lengths = [], [], []
        for i in range(bs):
            vis = visual_tokens[i, :visual_lengths[i], :]
            prm = prompt_embeds[i, :prompt_lengths[i], :]

            if "bart" in self.llm_name.lower():
                combined = torch.cat([vis], dim=0)
            else:
                combined = torch.cat([vis, prm], dim=0)
            joint_embeds.append(combined)
            visual_embeds.append(vis)

            if "bart" in self.llm_name.lower():
                new_lengths.append(visual_lengths[i])
            else:
                new_lengths.append(visual_lengths[i] + prompt_lengths[i])
            
        joint_embeds = pad_sequence(joint_embeds, batch_first=True)
        visual_embeds = pad_sequence(visual_embeds, batch_first=True)
        joint_mask = self.create_mask(new_lengths, device=visual_tokens.device)

        return joint_embeds, visual_embeds, joint_mask


    def forward(
        self, 
        sp_features, pose_features, mo_features, 
        sp_lengths, pose_lengths, mo_lengths,
        glosses, texts, icl_text, warmup=False
    ):
        bs = sp_features.shape[0] if self.include_sp else mo_features.shape[0]

        # english_texts = [icl_text[i].split('=')[0] for i in range(bs)]
        english_texts = [icl_text[i].split('=')[-2].split("\n")[1] for i in range(bs)]
        icl_text = safe_derangement(icl_text)
        prompts = [
            f"{self.prompt}\n\nExamples:\n{icl_text[i]}" 
            for i in range(bs)
        ]

        visual_tokens, visual_lengths, slr_loss = self._prepare_visual_input(
            sp_features, pose_features, mo_features,
            sp_lengths, pose_lengths, mo_lengths, glosses
        )

        joint_embeds, visual_embeds, joint_mask = self._prepare_joint_input(
            visual_tokens, visual_lengths, prompts
        )
        
        output_tokens = self.tokenizer(
            texts, 
            padding="longest", 
            truncation=True, 
            return_tensors="pt"
        ).to(joint_embeds.device)

        targets = output_tokens.input_ids.masked_fill(
            output_tokens.input_ids == self.tokenizer.pad_token_id, -100
        )

        if torch.isnan(visual_embeds).any():
            # replace nan with zeros
            for i in range(visual_embeds.size(0)):
                if torch.isnan(visual_embeds[i]).any():
                    visual_embeds[i] = torch.where(
                        torch.isnan(visual_embeds[i]), 
                        torch.zeros_like(visual_embeds[i]), 
                        visual_embeds[i]
                    )

        if self.training:
            loss = 0
            if self.contrastive:
                local_img, local_txt = self.prep_cont_tensors(visual_embeds, english_texts)

                global_img = all_gather(local_img)
                global_txt = all_gather(local_txt)

                if isinstance(global_img, (tuple, list)):
                    global_img = torch.cat(global_img, dim=0)
                if isinstance(global_txt, (tuple, list)):
                    global_txt = torch.cat(global_txt, dim=0)

                logit_scale = self.logit_scale.exp()
                logits_per_text = torch.matmul(global_txt, global_img.t()) * logit_scale

                loss = clip_loss(logits_per_text)

            if not warmup or not self.contrastive:
                outputs = self.llm(
                    inputs_embeds=joint_embeds,
                    attention_mask=joint_mask,
                    decoder_attention_mask=output_tokens.attention_mask,
                    labels=targets,
                )
                loss += outputs.loss
            
            return loss + slr_loss

        else:
            generated = self.llm.generate(
                inputs_embeds=joint_embeds,
                attention_mask=joint_mask,
                num_beams=5,
                max_length=self.max_txt_len,
                do_sample=False
            )

            if hasattr(generated, "sequences"):
                generated = generated.sequences

            generated_strings = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
            generated_strings = [gen.lower() for gen in generated_strings]

            reference_strings = self.tokenizer.batch_decode(output_tokens.input_ids, skip_special_tokens=True)
            reference_strings = [ref.lower() for ref in reference_strings]

            return generated_strings, reference_strings
