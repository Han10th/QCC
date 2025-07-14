import os
import numpy as np
import torch
import torch.nn as nn
from scipy.io import loadmat,savemat
from torch.utils.data import Dataset

def sample_grid(device, size):
	thetaTPS = torch.tensor([[[1,0,0],[0,1,0]]], dtype=torch.float)        
	origin_grid = torch.nn.functional.affine_grid(thetaTPS, (1,2,size[0],size[1]))[0]        
	origin_grid = origin_grid.permute((2,0,1))
	origin_grid = origin_grid.to(device=device, dtype=torch.float32)
	return origin_grid

def vec2map(vector, origin_grid):
	#vector[:,:,:,[0,-1]] = 0
	#vector[:,:,[0,-1],:] = 0
	sample_grid = vector + origin_grid
	sample_grid = sample_grid.permute((0, 2, 3, 1))
	return sample_grid

class OsaShapeLoader(Dataset):
	def __init__(self, foldername='../data_generated/osa_same/',phase='train', ratio=[0.0,0.7], transform = None, log = True):
		assert os.path.exists(foldername), 'Images folder does not exist'
		self.foldername = os.path.abspath(foldername)
		n = 500
		ratio = ratio if phase == 'train' else [ratio[1], 1]
		self.idx_list = [i+1 for i in range(int(n*ratio[0]), int(n*ratio[1]))] + [i+1 for i in range(int(n+n*ratio[0]),int(n+n*ratio[1]))]
		self.transform = transform
		self.log = log
	def __len__(self):
		return len(self.idx_list)
	def __getitem__(self, index):
		name = 'osa_same_%05d'%(self.idx_list[index])
		image_name = os.path.join(self.foldername, name + '.mat')
		mat_data = loadmat(image_name)
		image = np.transpose(mat_data['J'],(2,0,1))
		if self.log: image[-2:] = np.log(abs(image[-2:])+ 1e-15)
		label = np.round((mat_data['nose_size'][0]+2)/10)
		vertices2D = mat_data['map']
		image = torch.from_numpy(image)
		label = torch.from_numpy(label)
		vertices2D = torch.from_numpy(vertices2D)[...,[1,0]]
		if self.transform:
			image = self.transform(image)
		return image,label,vertices2D

class OsaLoader(Dataset):	
	def __init__(self, foldername,phase='train', ratio=[0.0,0.7],transform = None):
		assert os.path.exists(foldername), 'Images folder does not exist'	
		self.foldername = os.path.abspath(foldername)
		n = 500
		ratio = ratio if phase == 'train' else [ratio[1], 1]
		self.idx_list = [i+1 for i in range(int(n*ratio[0]), int(n*ratio[1]))] + [i+1 for i in range(int(n+n*ratio[0]),int(n+n*ratio[1]))]
		self.transform = transform

	def __len__(self):
		return len(self.idx_list)

	def __getitem__(self, index):			
		name = 'osa_%05d'%(self.idx_list[index])
		image_name = os.path.join(self.foldername, name + '.mat')
		mat_data = loadmat(image_name)
		image = np.transpose(mat_data['J'][:,:,[2,3,4]],(2,0,1))
		label = np.round((mat_data['nose_size'][0]+2)/10)
		image = torch.from_numpy(image)
		label = torch.from_numpy(label)
		if self.transform:
			image = self.transform(image)			
		return image,label,name

	
class MnistLoader(Dataset):	
	def __init__(self, foldername,phase='train',transform = None):		
		assert os.path.exists(foldername), 'Images folder does not exist'	
		self.foldername =os.path.abspath(foldername)
		if phase == 'train':
			self.idx_list = [i+1 for i in range(60000)]
		else:
			self.idx_list = [i+1 for i in range(60000,70000)]
		self.transform = transform	

	def __len__(self):
		return len(self.idx_list)

	def __getitem__(self, index):			
		name = 'mnist_%05d'%(self.idx_list[index])
		image_name = os.path.join(self.foldername, name + '.mat')
		mat_data = loadmat(image_name)
		image = np.transpose(mat_data['I'][:,:,[0]],(2,0,1))
		label = mat_data['label'][0]
		image = torch.from_numpy(image)
		label = torch.from_numpy(label)
		if self.transform:
			image = self.transform(image)			
		return image,label[0],name

	
class CifarLoader(Dataset):	
	def __init__(self, foldername,phase='train',transform = None):		
		assert os.path.exists(foldername), 'Images folder does not exist'	
		self.foldername =os.path.abspath(foldername)
		if phase == 'train':
			self.idx_list = [i+1 for i in range(50000)]
		else:
			self.idx_list = [i+1 for i in range(50000,60000)]
		self.transform = transform	

	def __len__(self):
		return len(self.idx_list)

	def __getitem__(self, index):			
		name = 'cifar10_%05d'%(self.idx_list[index])
		image_name = os.path.join(self.foldername, name + '.mat')
		mat_data = loadmat(image_name)
		image = np.transpose(mat_data['I'][:,:,[0,1,2]],(2,0,1))
		label = mat_data['label'][0]
		image = torch.from_numpy(image)
		label = torch.from_numpy(label)
		if self.transform:
			image = self.transform(image)			
		return image,label[0],name