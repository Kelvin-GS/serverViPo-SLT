import torch
import torch.nn as nn
import math
import numpy as np
from modules.pose_encoder.vh import VisualHead
# from vh import VisualHead



class PositionalEncoding(nn.Module):

    def __init__(self, channel, joint_num, time_len, domain):
        super(PositionalEncoding, self).__init__()
        self.joint_num = joint_num
        self.time_len = time_len

        self.domain = domain

        if domain == "temporal":
            # temporal embedding
            pos_list = []
            for t in range(self.time_len):
                for j_id in range(self.joint_num):
                    pos_list.append(t)
        elif domain == "spatial":
            # spatial embedding
            pos_list = []
            for t in range(self.time_len):
                for j_id in range(self.joint_num):
                    pos_list.append(j_id)

        position = torch.from_numpy(np.array(pos_list)).unsqueeze(1).float()
        pe = torch.zeros(self.time_len * self.joint_num, channel)

        div_term = torch.exp(torch.arange(0, channel, 2).float() *
                             -(math.log(10000.0) / channel))  # channel//2
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.view(time_len, joint_num, channel).permute(2, 0, 1).unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # nctv
        x = x + self.pe[:, :, :x.size(2)]
        return x


class STAttentionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, inter_channels, num_subset=2, num_node=27, num_frame=400,
                 kernel_size=1, stride=1, t_kernel=3, glo_reg_s=True, att_s=True, glo_reg_t=False, att_t=False,
                 use_temporal_att=False, use_spatial_att=True, attentiondrop=0., use_pes=True, use_pet=False):
        super(STAttentionBlock, self).__init__()
        self.inter_channels = inter_channels
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.num_subset = num_subset
        self.glo_reg_s = glo_reg_s
        self.att_s = att_s
        self.glo_reg_t = glo_reg_t
        self.att_t = att_t
        self.use_pes = use_pes
        self.use_pet = use_pet

        pad = int((kernel_size - 1) / 2)
        self.use_spatial_att = use_spatial_att
        if use_spatial_att:
            atts = torch.zeros((1, num_subset, num_node, num_node))
            self.register_buffer('atts', atts)
            self.pes = PositionalEncoding(in_channels, num_node, num_frame, 'spatial')
            self.ff_nets = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 1, 1, padding=0, bias=True),
                nn.BatchNorm2d(out_channels),
            )
            if att_s:
                self.in_nets = nn.Conv2d(in_channels, 2 * num_subset * inter_channels, 1, bias=True)
                self.alphas = nn.Parameter(torch.ones(1, num_subset, 1, 1), requires_grad=True)
            if glo_reg_s:
                self.attention0s = nn.Parameter(torch.ones(1, num_subset, num_node, num_node) / num_node,
                                                requires_grad=True)

            self.out_nets = nn.Sequential(
                nn.Conv2d(in_channels * num_subset, out_channels, 1, bias=True),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.out_nets = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, (1, 3), padding=(0, 1), bias=True, stride=1),
                nn.BatchNorm2d(out_channels),
            )
        padd = int(t_kernel / 2)
        self.out_nett = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, (t_kernel, 1), padding=(padd, 0), bias=True, stride=(stride, 1)),
            nn.BatchNorm2d(out_channels),
        )

        if in_channels != out_channels or stride != 1:
            if use_spatial_att:
                self.downs1 = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1, bias=True),
                    nn.BatchNorm2d(out_channels),
                )
            self.downs2 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=True),
                nn.BatchNorm2d(out_channels),
            )
            if use_temporal_att:
                self.downt1 = nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, 1, 1, bias=True),
                    nn.BatchNorm2d(out_channels),
                )
            self.downt2 = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, (kernel_size, 1), (stride, 1), padding=(pad, 0), bias=True),
                nn.BatchNorm2d(out_channels),
            )
        else:
            if use_spatial_att:
                self.downs1 = lambda x: x
            self.downs2 = lambda x: x
            self.downt2 = lambda x: x
        self.tan = nn.Tanh()
        self.relu = nn.LeakyReLU(0.1)
        self.drop = nn.Dropout(attentiondrop)

    def forward(self, x):
        N, C, T, V = x.size()
        if self.use_spatial_att:
            attention = self.atts
            if self.use_pes:
                y = self.pes(x)
            else:
                y = x
            if self.att_s:
                q, k = torch.chunk(self.in_nets(y).view(N, 2 * self.num_subset, self.inter_channels, T, V), 2,
                                   dim=1)  # nctv -> n num_subset c'tv [4,16,t,v]
                attention = attention + self.tan(torch.einsum('nsctu,nsctv->nsuv', [q, k]) / (self.inter_channels * T)) * self.alphas
            if self.glo_reg_s:
                attention = attention + self.attention0s.repeat(N, 1, 1, 1)
            attention = self.drop(attention)
            y = torch.einsum('nctu,nsuv->nsctv', [x, attention]).contiguous() \
                .view(N, self.num_subset * self.in_channels, T, V)
            y = self.out_nets(y)  # nctv
            y = self.relu(self.downs1(x) + y)
            y = self.ff_nets(y)
            y = self.relu(self.downs2(x) + y)
        else:
            y = self.out_nets(x)
            y = self.relu(self.downs2(x) + y)
        z = self.out_nett(y)
        z = self.relu(self.downt2(y) + z)

        return z


class DSTA(nn.Module):
    def __init__(self, num_frame=400,
                 num_subset=6, dropout=0.1,
                 cfg=None,
                 num_channel=2, glo_reg_s=True, att_s=True, glo_reg_t=False, att_t=False,
                 use_temporal_att=False, use_spatial_att=True, attentiondrop=0.1, use_pet=False,
                 use_pes=True, mode='SLR'):
        super(DSTA, self).__init__()
        self.mode = mode
        self.cfg = cfg
        config = self.cfg['net']
        self.out_channels = config[-1][1]
        in_channels = config[0][0]
        self.num_frame = num_frame
        param = {
            'num_subset': num_subset,
            'glo_reg_s': glo_reg_s,
            'att_s': att_s,
            'glo_reg_t': glo_reg_t,
            'att_t': att_t,
            'use_spatial_att': use_spatial_att,
            'use_temporal_att': use_temporal_att,
            'use_pet': use_pet,
            'use_pes': use_pes,
            'attentiondrop': attentiondrop
        }
        self.left_input_map = nn.Sequential(
            nn.Conv2d(num_channel, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.1),
        )
        self.right_input_map = nn.Sequential(
            nn.Conv2d(num_channel, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.1),
        )
        self.body_input_map = nn.Sequential(
            nn.Conv2d(num_channel, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.1),
        )
        self.face_input_map = nn.Sequential(
            nn.Conv2d(num_channel, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.1),
        )
        self.face_graph_layers = nn.ModuleList()
        for index, (in_channels, out_channels, inter_channels, t_kernel, stride) in enumerate(config):
            self.face_graph_layers.append(
                STAttentionBlock(in_channels, out_channels, inter_channels, stride=stride, t_kernel=t_kernel,num_node=26,
                                 num_frame=num_frame,
                                 **param))
            num_frame = int(num_frame / stride + 0.5)
        num_frame = self.num_frame
        self.left_graph_layers = nn.ModuleList()
        for index, (in_channels, out_channels, inter_channels, t_kernel, stride) in enumerate(config):
            self.left_graph_layers.append(
                STAttentionBlock(in_channels, out_channels, inter_channels, stride=stride, num_node=27,
                                 t_kernel=t_kernel, num_frame=num_frame,
                                 **param))
            num_frame = int(num_frame / stride + 0.5)
        num_frame = self.num_frame
        self.right_graph_layers = nn.ModuleList()
        for index, (in_channels, out_channels, inter_channels, t_kernel, stride) in enumerate(config):
            self.right_graph_layers.append(
                STAttentionBlock(in_channels, out_channels, inter_channels, stride=stride, t_kernel=t_kernel,
                                 num_frame=num_frame,
                                 **param))
            num_frame = int(num_frame / stride + 0.5)
        num_frame = self.num_frame
        self.body_graph_layers = nn.ModuleList()
        for index, (in_channels, out_channels, inter_channels, t_kernel, stride) in enumerate(config):
            self.body_graph_layers.append(
                STAttentionBlock(in_channels, out_channels, inter_channels, stride=stride, num_node=79,
                                 t_kernel=t_kernel, num_frame=num_frame,
                                 **param))
            num_frame = int(num_frame / stride + 0.5)
        self.drop_out = nn.Dropout(dropout)


    def forward(self,src_input):
        x = src_input['keypoint']
        N, C, T, V = x.shape
        x = x.permute(0, 1, 2, 3).contiguous().view(N, C, T, V)
        left = self.left_input_map(x[:, :, :, self.cfg['left']])
        right = self.right_input_map(x[:, :, :, self.cfg['right']])
        face = self.face_input_map(x[:, :, :, self.cfg['face']])
        body = self.body_input_map(x[:, :, :, self.cfg['body']])
        for i, m in enumerate(self.face_graph_layers):
            face = m(face)
        for i, m in enumerate(self.left_graph_layers):
            left = m(left)
        for i, m in enumerate(self.right_graph_layers):
            right = m(right)
        for i, m in enumerate(self.body_graph_layers):
            body = m(body)  # [B,256,T/4,N] -> [B,256]
        left = left.permute(0, 2, 1, 3).contiguous()
        right = right.permute(0, 2, 1, 3).contiguous()
        face = face.permute(0, 2, 1, 3).contiguous()
        body = body.permute(0, 2, 1, 3).contiguous()
        body = body.mean(3)
        face = face.mean(3)
        left = left.mean(3)
        right = right.mean(3)
        output = torch.cat([left, face, right, body], dim=-1)
        left_output = torch.cat([left, face], dim=-1)
        right_output = torch.cat([right,  face], dim=-1)
        return output, left_output, right_output, body


class Recognition(nn.Module):
    def __init__(self, cfg):
        super(Recognition, self).__init__()
        self.cfg = cfg
        self.input_type = cfg['input_type']
        self.fuse_method = cfg.get('fuse_method', 'empty')
        self.heatmap_cfg = cfg.get('heatmap_cfg', {})
        self.visual_backbone = None
        self.rgb_visual_head = None
        self.visual_backbone_keypoint = DSTA(cfg=self.cfg['DSTA-Net'],num_channel=3)
        self.fuse_visual_head = VisualHead(**cfg['fuse_visual_head'])
        self.body_visual_head = VisualHead(**cfg['body_visual_head'])
        self.left_visual_head = VisualHead(**cfg['left_visual_head'])
        self.right_visual_head = VisualHead(**cfg['right_visual_head'])

    def forward(self, src_input):
        fuse, left_output, right_output, body = self.visual_backbone_keypoint(src_input)
        fuse_head = self.fuse_visual_head(
            x=fuse,
            mask=src_input['mask'],
        )
        # body_head = self.body_visual_head(
        #     x=body,
        #     mask=src_input['mask'],
        # )
        # left_head = self.left_visual_head(
        #     x=left_output,
        #     mask=src_input['mask'],
        # )
        # right_head = self.right_visual_head(
        #     x=right_output,
        #     mask=src_input['mask'],
        # )

        return fuse_head

class PoseEncoder(nn.Module):
    def __init__(self, cfg):
        super(PoseEncoder, self).__init__()
        self.cfg = cfg
        self.recognition_network = Recognition(cfg)
    
    def forward(self, src_input):
        outputs = self.recognition_network(src_input)
        return outputs




if __name__ == "__main__":
    import yaml

    yaml_path = "/home/ahmedubc/projects/aip-lsigal/ahmedubc/MM_SLT/configs/phoenix-2014t_s2g.yaml"
    with open(yaml_path, 'r') as f:
        cfg = yaml.safe_load(f)["model"]["RecognitionNetwork"]

    model = PoseEncoder(cfg)

    # w_path = "/Users/gufran/Desktop/temp/best.pth"
    # weights = torch.load(w_path, map_location='cpu')["model"]

    # msg = model.load_state_dict(weights, strict=False)
    
    mask = torch.rand(4, 1, 31)
    x = {'keypoint': torch.randn(4, 3, 124, 133), "mask": mask} # -> [B, C, T, K]

    outputs = model(x)

    print(outputs.shape)  # [B, T, D]