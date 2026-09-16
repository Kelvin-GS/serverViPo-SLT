import pdb
import math
import copy
import torch
import collections
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (T, B, C)
        return x + self.pe[:x.size(0)].unsqueeze(1)

class TemporalConv(nn.Module):
    def __init__(self, input_size, hidden_size, conv_type=2, use_bn=False, num_classes=-1, padding="none"):
        super(TemporalConv, self).__init__()
        self.use_bn = use_bn
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.conv_type = conv_type

        if self.conv_type == 0:
            self.kernel_size = ['K3']
        elif self.conv_type == 1:
            self.kernel_size = ['K5', "P2"]
        elif self.conv_type == 2:
            self.kernel_size = ['K5', "P2", 'K5', "P2"]
        elif self.conv_type == 3:
            self.kernel_size = ['K5', "P2", 'K5', "P2", "K5", "P2"]
        elif self.conv_type == -2:
            self.kernel_size = ['K5', "P2", 'K5', "P2"]

        modules = []
        for layer_idx, ks in enumerate(self.kernel_size):
            input_sz = self.input_size if layer_idx == 0 or self.conv_type == 6 and layer_idx == 1 or self.conv_type == 7 and layer_idx == 1 or self.conv_type == 8 and layer_idx == 2 else self.hidden_size
            if ks[0] == 'P':
                modules.append(nn.MaxPool1d(kernel_size=int(ks[1]), ceil_mode=False))
            elif ks[0] == 'K':
                modules.append(
                    nn.Conv1d(input_sz, self.hidden_size, kernel_size=int(ks[1]), stride=1, padding=0)
                )
                modules.append(nn.BatchNorm1d(self.hidden_size))
                modules.append(nn.ReLU(inplace=True))
        self.temporal_conv = nn.Sequential(*modules)

        if self.num_classes != -1:
            self.fc = nn.Linear(self.hidden_size, self.num_classes)

    def update_lgt(self, lgt):
        feat_len = copy.deepcopy(lgt)
        for ks in self.kernel_size:
            if ks[0] == 'P':
                feat_len = torch.div(feat_len, 2)
            else:
                feat_len -= int(ks[1]) - 1
        return feat_len

    def forward(self, frame_feat, lgt):
        visual_feat = self.temporal_conv(frame_feat)
        lgt = self.update_lgt(lgt)
        logits = None if self.num_classes == -1 \
            else self.fc(visual_feat.transpose(1, 2)).transpose(1, 2)
        return {
            "visual_feat": visual_feat.permute(2, 0, 1),
            "conv_logits": logits.permute(2, 0, 1) if logits is not None else None,
            "feat_len": lgt.cpu(),
        }


class AttentionLayer(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        self.attn = nn.MultiheadAttention(
            d_model, num_heads, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, key_padding_mask=None):
        # x: (T, B, C)
        attn_out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)
        x = self.norm(x + attn_out)
        return x


class Attention(nn.Module):
    def __init__(self, d_model, num_heads, num_layers=4, slow_stride=8):
        super().__init__()
        self.slow_stride = slow_stride

        self.layers = nn.ModuleList([
            AttentionLayer(d_model, num_heads)
            for _ in range(num_layers)
        ])

    def forward(self, x, key_padding_mask=None):
        # x: (T, B, C)
        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)
        return x


class IMTM_CMTM(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        conv_type=2,
        slow_stride=4,
        num_classes=-1,
        num_heads=4,
        num_layers=4,
    ):
        super().__init__()

        # 1. Temporal Conv frontend
        self.temporal_conv = TemporalConv(
            input_size=input_size,
            hidden_size=hidden_size,
            conv_type=conv_type,
            num_classes=num_classes,
        )

        # 2. Positional Encoding
        self.pos_enc = PositionalEncoding(hidden_size)

        # 3. Slow–Fast Attention
        self.attn = Attention(
            d_model=hidden_size,
            num_heads=num_heads,
            slow_stride=slow_stride,
            num_layers=num_layers,
        )

    def create_mask(self, seq_lengths: list[int], device="cpu"):
        lengths = torch.tensor(seq_lengths, dtype=torch.int32, device=device)
        max_len = lengths.max().item()
        range_row = torch.arange(max_len, dtype=torch.int32, device=device).expand(len(lengths), -1)
        lengths = lengths.unsqueeze(1)
        mask = range_row < lengths  # shape: (batch_size, max_len)
        return mask

    def forward(self, frame_feat, lgt):
        """
        frame_feat: (B, C, T)
        lgt: (B,)
        """

        out = self.temporal_conv(frame_feat, lgt)
        x = out["visual_feat"].permute(1, 0, 2).clone()
        lgt = out["feat_len"]

        x = self.pos_enc(x)
        mask = self.create_mask(lgt.tolist(), device=x.device)
        x = self.attn(x, key_padding_mask=~mask) + out["visual_feat"].permute(1, 0, 2)

        return {
            "visual_feat": x.permute(1, 0, 2),
            "feat_len": lgt,
            "conv_logits": out["conv_logits"],
        }


if __name__ == "__main__":
    batch_size = 2
    input_size = 512
    hidden_size = 256
    seq_len = 32

    model = IMTM_CMTM(
        input_size=input_size,
        hidden_size=hidden_size,
        conv_type=2,
        num_heads=4,
        slow_stride=4,
        num_classes=-1,
    )

    frame_feat = torch.randn(batch_size, input_size, seq_len)  # (B, C, T)
    lgt = torch.tensor([seq_len] * batch_size)  # (B,)

    out = model(frame_feat, lgt)
    print("Visual feature shape:", out["visual_feat"].shape)
    print("Feature lengths:", out["feat_len"])