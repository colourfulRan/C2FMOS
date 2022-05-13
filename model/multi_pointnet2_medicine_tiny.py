import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch
import numpy as np
import os
from model.pointnet_util import PointNetConditionFeaturePropagation, PointNetSetConditionAbstraction

from torchsummary import summary
from ptflops import get_model_complexity_info

class get_model(nn.Module):
    def __init__(self, num_classes, remission, num_point):
        super(get_model, self).__init__()
        remission = remission + 1
        self.sa1 = PointNetSetConditionAbstraction(128, 0.1, 32, remission+6, [32, 32, 64], False)  #(remission+3)*2+3
        self.sa2 = PointNetSetConditionAbstraction(32, 0.2, 32, 64+3, [64, 64, 128], False)    #64*2+3
        self.fp2 = PointNetConditionFeaturePropagation(192+64, [256, 128]) #+64
        self.fp1 = PointNetConditionFeaturePropagation(128+64, [128, 128, 128])  #+64
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)


    def forward(self, xyz, task_ids):

        l0_points = xyz
        l0_xyz = xyz[:,:3,:]
        l0_remission = l0_points[:, 3:, :]

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points, task_ids)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points, task_ids)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points, task_ids)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points, task_ids)


        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x, l1_points



class Focal_Dice_Loss(nn.Module):
    def __init__(self, gamma=2):
        super(Focal_Dice_Loss, self).__init__()
        self.gamma = gamma


    def forward(self, pred, target, weight):
        fl_w = 0.5
        logpt = -F.cross_entropy(pred, target)
        pt = torch.exp(logpt)
        focal_loss = -((1-pt)**self.gamma)*logpt
        num = target.size(0)
        smooth = 1
        probs = F.sigmoid(pred)
        m1 = probs.view(num, -1)
        m2 = target.view(num, -1).float()
        intersection = (m1 * m2)
        score = (2. * intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        dice = score.sum() / num

        return fl_w * focal_loss - (1 - fl_w) * dice

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, logits, targets, weights):
        num = targets.size(0)
        smooth = 1
        probs = F.sigmoid(logits)
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1).float()
        intersection = (m1 * m2)
        score =  (2. *intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        score = 1 - score.sum() / num
        return score



class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma


    def forward(self, pred, target, weight):
        logpt = -F.cross_entropy(pred, target)
        pt = torch.exp(logpt)
        focal_loss = -((1-pt)**self.gamma)*logpt
        return focal_loss




if __name__ == '__main__':
    import  torch
    model = get_model(13)
    flops, params = get_model_complexity_info(model, (9, 1024), as_strings=True, print_per_layer_stat=True)
    print('flops:'+flops)
    print('params:'+params)
    #print(flopth(model, in_size=[3, 112, 112]))
    summary(model.cuda(), (9, 1024), batch_size=16)
    # xyz = torch.rand(16, 12, 1024)
    # (model(xyz))