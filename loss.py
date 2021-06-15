import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
from skimage.segmentation import find_boundaries
from scipy import ndimage
from multiprocessing import Pool
from itertools import product
import pathos.pools as pp
from functools import partial
import torch.multiprocessing as mp
from torch.multiprocessing import Pool, set_start_method
from torch.multiprocessing import Process

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage
import MBD
from torchviz import make_dot

import pdb

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=False):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE

def cross_entropy_loss2d(inputs, targets, cuda=True, balance=1.1):
    """
    :param inputs: inputs is a 4 dimensional data nx1xhxw
    :param targets: targets is a 3 dimensional data nx1xhxw
    :return:
    """
    n, c, h, w = inputs.size()
    weights = np.zeros((n, c, h, w))
    for i in range(n):
        t = targets[i, :, :].cpu().data.numpy()
        pos = (t == 1).sum()
        neg = (t == 0).sum()
        valid = neg + pos
        weights[i, t == 1] = neg * 1. / valid
        weights[i, t == 0] = pos * balance / valid
    weights = torch.Tensor(weights)
    if cuda:
        weights = weights.cuda()
    loss = nn.BCEWithLogitsLoss(weights, reduction='mean')(inputs, targets)
    return loss

def DAHU_loss(inputs, seed, boundary, labels):
    # target_image: the boundary total
    EPM = inputs.squeeze()
    labels = labels.squeeze()

    seed = seed.squeeze(axis = 0).squeeze(axis = 0)
    # masking = mask.squeeze(axis=0).squeeze(axis = 0)
    boundary = boundary.squeeze(axis=0).squeeze(axis = 0)

    d, _, _ = seed.shape
    loss_total = 0
    for i in range(d):
        loss = 0
        EPM_s, seed_s, boundary_s = EPM, seed[i,:,:], boundary[i,:,:]

        max_range = torch.max(seed_s).type(torch.uint8)
        seed_s = np.array(seed_s.detach().cpu().numpy()).astype(np.uint8)

        EPM_s = torch.sigmoid(EPM_s)

        EPM_s_clone = EPM_s.clone()
        EPM_s_clone = np.array((EPM_s_clone*255).detach().cpu().numpy()).astype(np.uint8)

        if max_range>1:


            inside = np.array(seed_s == 1).astype(np.uint8)*255
            outside = np.array(seed_s > 1).astype(np.uint8)*255
            #cv2.imwrite('bca' + str(i) + '.png', (inside/3 + outside/3 + EPM_s_clone/3).astype(np.uint8))

            destination = np.array(seed_s).astype(np.uint8)
            contour = MBD.MBD_cut(EPM_s_clone,inside, outside)
            kernel = np.ones((7,7),np.uint8)
            contour = cv2.dilate(contour,kernel,iterations = 1)

            #cv2.imwrite('abc' + str(i) + '.png', contour)

            #contour = (contour).astype(np.uint8)
            #boundary_s = np.array(boundary_s.detach().cpu().numpy()).astype(np.uint8)
            # cv2.imwrite('cab' + str(i) + '.png', (EPM_s_clone/2 + contour/2 ).astype(np.uint8))
            # cv2.imwrite('abc' + str(i) + '.png', (contour/2 + np.array(boundary_s.detach().cpu().numpy()).astype(np.uint8)*125).astype(np.uint8))

            #pdb.set_trace()

            contour = (contour/255).astype(np.uint8)

            contour = torch.from_numpy(np.array([contour])).cuda()
            # cut = contour > 0
            # # MSE loss
            # minimum_value = torch.min(torch.masked_select(EPM_s, cut))
            # #print(minimum_value)
            # min_loss = (torch.ones(1).cuda()-minimum_value)*(torch.ones(1).cuda()-minimum_value)
            # loss = loss + min_loss
            # # BCE loss
                        
            EPM_contour = (EPM_s * contour).unsqueeze(axis=0)
            # boundary_s = boundary_s.unsqueeze(axis=0).unsqueeze(axis=0)
            label_s = (labels * contour).unsqueeze(axis=0)
            # pdb.set_trace()
            # cv2.imwrite('GT' + str(i) + '.png', (label_s.squeeze().detach().cpu().numpy().astype(np.uint8)*255 ))
            # cv2.imwrite('MBD_cut' + str(i) + '.png', ((EPM_contour*255).squeeze().detach().cpu().numpy().astype(np.uint8) ))

            BCE_loss = cross_entropy_loss2d(EPM_contour, label_s, True, 1.1)*10
            loss = loss + BCE_loss 
            # print(BCE_loss)

            # pdb.set_trace()

            # torch.cuda.empty_cache()

        else:
            loss = torch.tensor(0).type(torch.float64)
        loss_total = loss_total + loss # Accumulate sum
    loss_total = loss_total / d
    # make_dot(loss_total).render("attached", format="png")
    # pdb.set_trace()
    return loss_total
