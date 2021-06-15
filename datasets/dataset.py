import os
import cv2
import numpy as np
from PIL import Image
import scipy.io
import torch
from torch.utils import data
from skimage import transform
import random
from io import BytesIO
import scipy.io
import matplotlib.pyplot as plt

import pdb

def load_image_with_cache(path, cache=None, lock=None):
	if cache is not None:
		if path not in cache:
			with open(path, 'rb') as f:
				cache[path] = f.read()
		return Image.open(BytesIO(cache[path].content))
	return Image.open(path)


class Data(data.Dataset):
	def __init__(self, root, files,
		mean_bgr = np.array([128.85538516998292]),
		crop_size=None, rgb=True, scale=None, augment=False):
		self.mean_bgr = mean_bgr
		self.root = root
		self.files = files
		self.crop_size = crop_size
		self.rgb = rgb
		self.scale = scale
		self.cache = {}
		self.augment = augment

	def __len__(self):
		return len(self.files)

	def __getitem__(self, index):
		data_file = self.files[index]
		# load Image
		img_file = self.root + data_file[0]

		if not os.path.exists(img_file):
			img_file = img_file.replace('jpg', 'png')
		img = Image.open(img_file)

		contour_file = self.root + data_file[1]
		contour = Image.open(contour_file)
		contour = contour.convert('L')
		contour = np.array(contour)/255.
		contour = contour.astype(np.uint8)

		# load seed image
		seed_file = self.root + data_file[2]
		seed = scipy.io.loadmat(seed_file)["seed"]		

		boundary_file = self.root + data_file[3]
		boundary = scipy.io.loadmat(boundary_file)["boundary"]	

		return self.transform(img, contour, seed, boundary)

	def transform(self, img, contour, seed, boundary, fixed_size=512):

		img = np.array(img, dtype=np.float32)
		img -= self.mean_bgr
		
		if len(img.shape) == 3:
			img = img.transpose((2, 0, 1))

		img = torch.from_numpy(img.copy()).float()
		img = img.unsqueeze(axis=0)

		contour = torch.from_numpy(np.array([contour])).float()
		seed = torch.from_numpy(np.array([seed])).squeeze(axis=0).float()
		boundary = torch.from_numpy(np.array([boundary])).squeeze(axis=0).float()

		return img, contour, seed, boundary



class Data_val(data.Dataset):
	def __init__(self, root, files,
		mean_bgr = np.array([128.85538516998292]),
		crop_size=None, rgb=True, scale=None, augment=False):
		self.mean_bgr = mean_bgr
		self.root = root
		self.files = files
		self.crop_size = crop_size
		self.rgb = rgb
		self.scale = scale
		self.cache = {}
		self.augment = augment

	def __len__(self):
		return len(self.files)

	def __getitem__(self, index):
		data_file = self.files[index]
		# load Image
		img_file = self.root + data_file[0]

		if not os.path.exists(img_file):
			img_file = img_file.replace('jpg', 'png')
		img = Image.open(img_file)

		contour_file = self.root + data_file[1]
		contour = Image.open(contour_file)
		contour = contour.convert('L')
		contour = np.array(contour)/255.
		contour = contour.astype(np.uint8)



		return self.transform(img, contour)

	def transform(self, img, contour, fixed_size=512):

		img = np.array(img, dtype=np.float32)
		img -= self.mean_bgr
		
		if len(img.shape) == 3:
			img = img.transpose((2, 0, 1))

		img = torch.from_numpy(img.copy()).float()
		img = img.unsqueeze(axis=0)

		contour = torch.from_numpy(np.array([contour])).float()


		return img, contour
