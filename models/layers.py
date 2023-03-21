
import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """ Squeeze-and-excitation block """
    def __init__(self, channels, r=16):
        super(SEBlock, self).__init__()
        self.r = r 
        self.squeeze = nn.Sequential(nn.Linear(channels, channels//self.r),
                                     nn.ReLU(),
                                     nn.Linear(channels//self.r, channels),
                                     nn.Sigmoid())

    def forward(self, x):
        B, C, H, W = x.size()
        squeeze = self.squeeze(torch.mean(x, dim=(2,3))).view(B,C,1,1)
        return torch.mul(x, squeeze)


class SABlock(nn.Module):
    """ Spatial self-attention block """
    def __init__(self, in_channels, out_channels):
        super(SABlock, self).__init__()
        self.attention = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                                        nn.Sigmoid())
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)

    def forward(self, x):
        attention_mask = self.attention(x)
        features = self.conv(x)
        return torch.mul(features, attention_mask)


class BalancedFusionModule(nn.Module):
    """ CWF：Implementation of Complementary Weighted Fusion Module  """
    def __init__(self, in_channels, out_channels):
        super(BalancedFusionModule, self).__init__()

        self.attention = nn.Sequential(nn.Conv2d(in_channels*2, out_channels, 3, padding=1, bias=False),
                                       nn.Sigmoid())
        self.conv_2 = nn.Conv2d(in_channels*2, out_channels, 3, padding=1, bias=False)

    def forward(self, x, y):
        F = torch.cat((x, y), dim=1)
        attention_mask = self.attention(F)

        features_x = torch.mul(x, attention_mask)
        features_y = torch.mul(y, torch.ones_like(attention_mask) - attention_mask)
        out_x = features_x + x
        out_y = features_y + y

        out = torch.cat((out_x, out_y), dim=1)
        out = self.conv_2(out)
        return out



