'''
简介：包含edge、semseg 、depth 的多任务网络
     WFM模块： （是）
     channels attention（是）
时间：2022.7.26
备注：
作者：wangxy
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.resnet import Bottleneck
from models.layers import SEBlock, BalancedFusionModule



class InitialTaskPredictionModule(nn.Module):
    """
        Make the initial task predictions from the backbone features.
    """

    def __init__(self, AUXILARY_TASKS, TASKS, input_channels, intermediate_channels=256):
        super(InitialTaskPredictionModule, self).__init__()
        self.tasks = TASKS
        layers = {}
        conv_out = {}

        for task in self.tasks:
            if input_channels != intermediate_channels:
                downsample = nn.Sequential(nn.Conv2d(input_channels, intermediate_channels, kernel_size=1,
                                                     stride=1, bias=False), nn.BatchNorm2d(intermediate_channels))
            else:
                downsample = None
            bottleneck1 = Bottleneck(input_channels, intermediate_channels // 4, downsample=downsample)
            # upsampling = nn.Upsample(scale_factor=2)
            bottleneck2 = Bottleneck(intermediate_channels, intermediate_channels // 4, downsample=None)
            conv_out_ = nn.Conv2d(intermediate_channels,  AUXILARY_TASKS.NUM_OUTPUT[task], 1)
            layers[task] = nn.Sequential(bottleneck1, bottleneck2)
            conv_out[task] = nn.Sequential(conv_out_)

        self.layers = nn.ModuleDict(layers)
        self.conv_out = nn.ModuleDict(conv_out)

    def forward(self, x):
        out = {}

        for task in self.tasks:
            out['features_%s' % (task)] = self.layers[task](x)
            out[task] = self.conv_out[task](out['features_%s' % (task)])

        return out


class Multi_feature_Fusion_Module(nn.Module):
    """
        Perform Multi-Task Distillation
        We apply an attention mask to features from other tasks and
        add the result as a residual.
    """

    def __init__(self, tasks, auxilary_tasks, channels):
        super(Multi_feature_Fusion_Module, self).__init__()
        self.tasks = tasks
        self.auxilary_tasks = auxilary_tasks
        self.BalancedFusionModule = BalancedFusionModule(channels, channels)

    def forward(self, x):
        out = {}
        for t in self.tasks:
            other_tasks = [a for a in self.auxilary_tasks if a != t]
            other_task_features = self.BalancedFusionModule(x['features_%s' % (other_tasks[0])], x['features_%s' % (other_tasks[1])])
            out[t] = x['features_%s' % (t)] + other_task_features
        return out


class UDE_Net(nn.Module):
    def __init__(self, TASKS, AUXILARY_TASKS, backbone, backbone_channels):
        super(UDE_Net, self).__init__()
        # General
        self.tasks = TASKS.NAMES
        self.auxilary_tasks = AUXILARY_TASKS.NAMES
        self.channels = backbone_channels

        # Backbone
        self.backbone = backbone

        # Task-specific heads for initial prediction
        self.initial_task_prediction_heads = InitialTaskPredictionModule(AUXILARY_TASKS, self.auxilary_tasks,
                                                                         self.channels)
        # Multi-modal distillation
        self.Multi_feature_Fusion_Module = Multi_feature_Fusion_Module(self.tasks, self.auxilary_tasks, 256)
        # Task-specific heads for  prediction
        heads = {}
        for task in self.auxilary_tasks:
            bottleneck1 = Bottleneck(256, 256 // 4, downsample=None)
            bottleneck2 = Bottleneck(256, 256 // 4, downsample=None)
            # upsampling = nn.Upsample(scale_factor=2)
            conv_out_ = nn.Conv2d(256, AUXILARY_TASKS.NUM_OUTPUT[task], 1)
            heads[task] = nn.Sequential(bottleneck1, bottleneck2, conv_out_)

        self.heads = nn.ModuleDict(heads)
        # self.sigmoid = nn.Sigmoid()
        self.downsample = nn.Sequential(nn.Conv2d(256, 64, kernel_size=1, stride=1, bias=False),
                                   nn.BatchNorm2d(64))

        self.conv1x1 = nn.Conv2d(in_channels=320,
                                   out_channels=256, kernel_size=1, stride=1,
                                   padding=0, bias=True)

        self.channel_attention = SEBlock(channels=320, r=16)

    def forward(self, x):
        img_size = x.size()[-2:]
        out = {}
        features = {}


        # Backbone
        x = self.backbone(x)

        # Initial predictions for every task including auxilary tasks
        x = self.initial_task_prediction_heads(x)
        for task in self.auxilary_tasks:
            out['initial_%s' % (task)] = x[task]
            features['initial_features_%s' % (task)] = self.downsample(x['features_%s' % (task)])

        # Refine features through multi-modal distillation
        x = self.Multi_feature_Fusion_Module(x)

        for task in self.tasks:
            out['middle_%s' % (task)] = F.interpolate(self.heads[task](x[task]), img_size, mode='bilinear')

        multi_features = torch.cat((features['initial_features_edge'],
                                    features['initial_features_depth'],
                                    self.downsample(x['edge']),
                                    self.downsample(x['depth']),
                                    self.downsample(x['semseg'])), dim=1)

        multi_features = self.channel_attention(multi_features) + multi_features
        multi_features = self.conv1x1(multi_features)
        depth = F.interpolate(self.heads['depth'](multi_features), img_size, mode='bilinear')
        out['depth'] = depth
        return out
