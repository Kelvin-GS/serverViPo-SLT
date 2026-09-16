import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.pose_encoder.util import PositionalEncoding, MaskedNorm, PositionwiseFeedForward, MLPHead
# from util import PositionalEncoding, MaskedNorm, PositionwiseFeedForward, MLPHead


class VisualHead(torch.nn.Module):
    def __init__(self, 
        input_size=512, hidden_size=1024, ff_size=2048, pe=True,
        ff_kernelsize=[3,3], pretrained_ckpt=None, is_empty=False, frozen=False,
        ssl_projection_cfg={}):
        super().__init__()
        self.is_empty = is_empty
        self.ssl_projection_cfg = ssl_projection_cfg
        if is_empty==False:
            self.frozen = frozen
            self.hidden_size = hidden_size

            if input_size is None:
                self.fc1 = nn.Identity()
            else:
                self.fc1 = torch.nn.Linear(input_size, self.hidden_size)
            # self.bn1 = nn.BatchNorm1d(num_features=self.hidden_size) 
            self.bn1 = MaskedNorm(num_features=self.hidden_size, norm_type='batch')
            self.relu1 = torch.nn.ReLU()
            self.dropout1 = torch.nn.Dropout(p=0.1)

            if pe:
                self.pe = PositionalEncoding(self.hidden_size)
            else:
                self.pe = torch.nn.Identity()

            self.feedforward = PositionwiseFeedForward(input_size=self.hidden_size,
                ff_size=ff_size,
                dropout=0.1, kernel_size=ff_kernelsize, skip_connection=True)
            
            self.layer_norm = torch.nn.LayerNorm(self.hidden_size, eps=1e-6)

            self.plus_conv = nn.Identity()

            if ssl_projection_cfg!={}:
                self.ssl_projection = MLPHead(embedding_size=self.hidden_size, 
                    projection_hidden_size=ssl_projection_cfg['hidden_size'])


            if self.frozen:
                self.frozen_layers = [self.fc1, self.bn1, self.relu1,  self.pe, self.dropout1, self.feedforward, self.layer_norm]
                for layer in self.frozen_layers:
                    for name, param in layer.named_parameters():
                        param.requires_grad = False
                    layer.eval()

        if pretrained_ckpt:
            self.load_from_pretrained_ckpt(pretrained_ckpt)

    def load_from_pretrained_ckpt(self, pretrained_ckpt):
        checkpoint = torch.load(pretrained_ckpt, map_location='cpu')['model_state']
        load_dict = {}
        for k,v in checkpoint.items():
            if 'recognition_network.visual_head.' in k:
                load_dict[k.replace('recognition_network.visual_head.','')] = v
        self.load_state_dict(load_dict)

    def forward(self, x, mask):
        B, Tin, D = x.shape 
        if self.is_empty==False:
            if not self.frozen:
                #projection 1
                x = self.fc1(x)
                x = self.bn1(x, mask)
                # x = self.bn1(x.transpose(1,2)).transpose(1,2)
                x = self.relu1(x)
                #pe
                x = self.pe(x)
                x = self.dropout1(x)

                #feedforward
                x = self.feedforward(x)
                x = self.layer_norm(x)

                x = x.transpose(1,2)
                x = self.plus_conv(x)
                x = x.transpose(1,2)
            else:
                with torch.no_grad():
                    for ii, layer in enumerate(self.frozen_layers):
                        layer.eval()
                        if ii==1:
                            x = layer(x, mask)
                        else:
                            x = layer(x)
                x = x.transpose(1,2)
                x = self.plus_conv(x)
                x = x.transpose(1,2)


        return x