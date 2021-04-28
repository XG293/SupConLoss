import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable

import sys, os
import numpy as np
import random
act = torch.nn.ReLU()


import math
from torch.nn.utils.weight_norm import WeightNorm



class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None
    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

class distLinear(nn.Module):
    def __init__(self, indim, outdim):
        super(distLinear, self).__init__()
        self.L = nn.Linear( indim, outdim, bias = False)
        self.class_wise_learnable_norm = True  #See the issue#4&8 in the github 
        if self.class_wise_learnable_norm:      
            WeightNorm.apply(self.L, 'weight', dim=0) #split the weight update component to direction and norm      

        if outdim <=200:
            self.scale_factor = 2; #a fixed scale factor to scale the output of cos value into a reasonably large input for softmax
        else:
            self.scale_factor = 10; #in omniglot, a larger scale factor is required to handle >1000 output classes.

    def forward(self, x):
        x_norm = torch.norm(x, p=2, dim =1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm+ 0.00001)
        if not self.class_wise_learnable_norm:
            L_norm = torch.norm(self.L.weight.data, p=2, dim =1).unsqueeze(1).expand_as(self.L.weight.data)
            self.L.weight.data = self.L.weight.data.div(L_norm + 0.00001)
        cos_dist = self.L(x_normalized) #matrix product by forward function, but when using WeightNorm, this also multiply the cosine distance by a class-wise learnable norm, see the issue#4&8 in the github
        scores = self.scale_factor* (cos_dist) 

        return scores   
    
class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)


    
class WideResNet(nn.Module):
    def __init__(self, depth=28, widen_factor=10, num_classes=200, loss_type = 'dist', stride = 1 ):
        dropRate = 0.5
        flatten = True
        super(WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, stride, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and linear
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.nChannels = nChannels[3]
        
        if loss_type == 'softmax':
            self.linear = nn.Linear(nChannels[3], int(num_classes))
            self.linear.bias.data.fill_(0)
        else:
            self.linear = distLinear(nChannels[3], int(num_classes))
        
        self.num_classes = num_classes
        if flatten:
            self.final_feat_dim = 640
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    
    def forward(self, x,):
        out = x                                     # [bs, 3, 224, 224]
        #print("input dim:", out.size())
        out = self.conv1(out)                       # [bs, 16, 224, 224]
        #print("conv1 dim:", out.size())
        out = self.block1(out)                      # [bs, 160, 224, 224]
        #print("block1 dim:", out.size())
        out = self.block2(out)                      # [bs, 320, 112, 112]
        #print("block2 dim:", out.size())
        out = self.block3(out)                      # [bs, 640, 56, 56]
        #print("block3 dim:", out.size())
        out = self.relu(self.bn1(out))              # [bs, 640, 56, 56]
        #print("relu dim:", out.size())
        out = F.avg_pool2d(out, out.size()[2:])     # [bs, 640, 1, 1]
        #print("avg_pool2d dim:", out.size())
        out = out.view(out.size(0), -1)             # [bs, 640]
        #print("output dim:", out.size())
        out1 = self.linear(out)                     # [bs, num_classes]
        #print("out1 dim:", out1.size())     
        return out#, out1


def wrn28_10(num_classes=200 , loss_type = 'dist'):
    model = WideResNet(depth=28, widen_factor=10, num_classes=num_classes, loss_type = loss_type, stride = 1 )
    return model

model_dict = {
    'wrn28_10': [wrn28_10, 640]
}
class SupConWideResNet(nn.Module):
    """backbone + projection head"""
    def __init__(self, name='resnet50', head='mlp', feat_dim=128):
        super(SupConWideResNet, self).__init__()
        model_fun, dim_in = model_dict[name]
        self.encoder = model_fun()
        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, feat_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

    def forward(self, x):
        feat = self.encoder(x)
        feat = F.normalize(self.head(feat), dim=1) # L2-norm
        return feat