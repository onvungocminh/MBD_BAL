import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from BALoss import *
import scipy.io
from PIL import Image
import pdb


img_file = 'img.png'
img = Image.open(img_file)
img = np.array(img, dtype=np.float32)
seed_file = 'seed.mat'
seed = scipy.io.loadmat(seed_file)["seed"]
boundary_file = 'boundary.mat'
boundary = scipy.io.loadmat(boundary_file)["boundary"]	
contour_file = 'contour.png'
contour = Image.open(contour_file)
contour = contour.convert('L')
contour = np.array(contour)/255.
contour = contour.astype(np.uint8)


img = torch.from_numpy(img.copy()).float()
img = img.unsqueeze(axis=0)
contour = torch.from_numpy(np.array([contour])).float()
seed = torch.from_numpy(np.array([seed])).squeeze(axis=0).float()
boundary = torch.from_numpy(np.array([boundary])).squeeze(axis=0).float()

img, contour, seed, boundary = img.cuda(),  contour.cuda(), seed.cuda(), boundary.cuda()

BAloss = DAHU_loss(img, seed, boundary, contour)
print(BAloss)