import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.io import loadmat,savemat
from skimage.measure import label
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from torch.utils.data import TensorDataset,Dataset
from torchvision.io import read_image

def save_image(MASK_list, path_list, DICE_list, HD95_list, BETT_list, foldername, modelname):
    N = MASK_list.shape[0]        
    stat_file = open(foldername + '/' + modelname + '_test.csv', 'w+')    
    stat_file.write('DICE, HD95, BETT, path\n')
    stat_file.flush()
    for i in range(N):
        cv2.imwrite(path_list[i], np.uint8(np.round(MASK_list[i,:,:,:]*255)))
        stat_file.write('{}, {}, {}, {}\n'.format(DICE_list[i], HD95_list[i], BETT_list[i], path_list[i]))
        stat_file.flush()            
    stat_file.close()

class KiTS2dataset(Dataset):
    def __init__(self, root, phase, model, transform):
        self.phase=phase
        self.path = root + phase
        self.img_list = os.listdir(self.path + 'im2d')
        self.model = model
        self.transform = transform
    def __len__(self):
        return len(self.img_list)
    def __getitem__(self, index):
        imagename = self.img_list[index]
        path1 = self.path + 'im2d/' + imagename
        path2 = self.path + 'mask/' + imagename
        paths = self.path + self.model + '/' + imagename
        sub_folder = os.path.split(paths)[0]
        if not os.path.exists(sub_folder) and self.phase=='test/':
            os.makedirs(sub_folder)
            print("The new directory is created!   " + sub_folder)
        image = read_image(path1).to(dtype=torch.float32)/255
        mask = read_image(path2).to(dtype=torch.float32)/255
        if self.transform is not None:
            image = self.transform(image)
            mask = self.transform(mask)
        return image, mask, paths


def HAM2NPY(grid_size,folder):    
    foldername1 = folder + '/HAM10000_images_part_1/'
    foldername2 = folder + '/HAM10000_images_part_2/'
    foldername_m = folder + '/HAM10000_segmentations_lesion_tschandl/'
    im_array1 = ims2nps(foldername1,grid_size, 3)
    im_array2 = ims2nps(foldername2,grid_size, 3)
    mask_array = ims2nps(foldername_m,grid_size, 1)
    im_array = np.concatenate((im_array1,im_array2),axis=0)
    np.save(folder+'HAM_image',im_array)
    np.save(folder+'HAM_mask',mask_array)
    
def load_skinMNIST(foldername, perc = 1.00):     
    im = np.load(foldername + 'HAM_image_CCA.npy')
    mask = np.load(foldername + 'HAM_mask_CCA.npy')
    im = (im/255)
    print(im.shape)
    print(mask.shape)

    im_train = torch.from_numpy(im[1:int(perc*im.shape[0]),:,:,:])
    mask_train = torch.from_numpy(mask[0:int(perc*im.shape[0]),:,:,:])
    im_test = torch.from_numpy(im[int(perc*im.shape[0]):,:,:,:])
    mask_test = torch.from_numpy(mask[int(perc*im.shape[0]):,:,:,:])
    trainset = TensorDataset(im_train, im_train[:,0:1,:,:], mask_train)
    testset = TensorDataset(im_test, im_test[:,0:1,:,:], mask_test)           
    
    I_prior = DiskDrawer2D([im.shape[2],im.shape[3]])
    I_prior = torch.from_numpy(I_prior).unsqueeze(axis=0)
    thetaTPS = torch.tensor([[[1,0,0],[0,1,0]]], dtype=torch.float)        
    coor = F.affine_grid(thetaTPS, (1,2,im.shape[2],im.shape[3]))[0]        
    coor = coor.permute((2,0,1))
    print("------------ LOAD COMPLETE ------------")
    return trainset, testset, I_prior, coor

def load_skinMNISTone(foldername, perc = 1.00, i=0):     
    im = np.load(foldername + 'HAM_image_CCA.npy')
    mask = np.load(foldername + 'HAM_mask_CCA.npy')
    im = (im/255)
    print(im.shape)
    print(mask.shape)

    im_train = torch.from_numpy(im[i:i+1,:,:,:])
    mask_train = torch.from_numpy(mask[i:i+1,:,:,:])
    im_test = torch.from_numpy(im[int(perc*im.shape[0]):,:,:,:])
    mask_test = torch.from_numpy(mask[int(perc*im.shape[0]):,:,:,:])
    trainset = TensorDataset(im_train, im_train[:,0:1,:,:], mask_train)
    testset = TensorDataset(im_test, im_test[:,0:1,:,:], mask_test)           
    
    I_prior = DiskDrawer2D([im.shape[2],im.shape[3]])
    I_prior = torch.from_numpy(I_prior).unsqueeze(axis=0)
    thetaTPS = torch.tensor([[[1,0,0],[0,1,0]]], dtype=torch.float)        
    coor = F.affine_grid(thetaTPS, (1,2,im.shape[2],im.shape[3]))[0]        
    coor = coor.permute((2,0,1))
    print("------------ LOAD COMPLETE ------------")
    return trainset, testset, I_prior, coor


def load_ACDC2D(foldername, perc = 0.90):  
    im = np.load(foldername + 'ACDC2_image.npy')
    mask = np.load(foldername + 'ACDC2_mask.npy')
    mask_fill = np.load(foldername + 'ACDC2_mask_fill.npy')
    im = (im/255)
    #mask_fill = mask
    #for i in range(mask.shape[0]):
    #    mask_this = mask[i,0,:,:]
    #    mask_fill[i,0,:,:] = HoleFiller2D(mask_this)        
    print(im.shape)
    print(mask.shape)
    im_train = torch.from_numpy(im[0:int(perc*im.shape[0]),:,:,:])
    mask_train = torch.from_numpy(mask[0:int(perc*im.shape[0]),:,:,:])
    maFi_train = torch.from_numpy(mask_fill[0:int(perc*im.shape[0]),:,:,:])
    im_test = torch.from_numpy(im[int(perc*im.shape[0]):,:,:,:])
    mask_test = torch.from_numpy(mask[int(perc*im.shape[0]):,:,:,:])
    maFi_test = torch.from_numpy(mask_fill[int(perc*im.shape[0]):,:,:,:])
    trainset = TensorDataset(im_train, mask_train, maFi_train)
    testset = TensorDataset(im_test, mask_test, maFi_test)     

    I_disk = DiskDrawer2D([im.shape[2],im.shape[3]])
    I_circular = HoleDigger2D(I_disk)
    I_disk = torch.from_numpy(I_disk).unsqueeze(axis=0)
    I_circular = torch.from_numpy(I_circular).unsqueeze(axis=0)
    thetaTPS = torch.tensor([[[1,0,0],[0,1,0]]], dtype=torch.float)        
    coor = F.affine_grid(thetaTPS, (1,2,im.shape[2],im.shape[3]))[0]        
    coor = coor.permute((2,0,1))
    print("------------ LOAD COMPLETE ------------")
    return trainset, testset, I_disk, I_circular, coor


def augment_deform(im, mask, rotation, padding_width, scale_ranges = [0.8,1.25]):
    '''
    HINT : 
    1.The parameter 'tran_rate' and 'l' is the same for 'h' and 'w', but can be specific if one want to improve this;
    2.The parameter 'scale_ranges' can be determined automatically by 'h' 'w' and 'padding_width';
    '''    
    batch_size,c,h,w = im.shape
    tran_rate = (padding_width) / max([h,w])
    a=(np.random.rand(batch_size,1,1)-.5)*2*rotation
    l = scale_ranges[1] - scale_ranges[0]
    s1 = (np.random.rand(batch_size,1,1)*l)+scale_ranges[0]
    s2 = (np.random.rand(batch_size,1,1)*l)+scale_ranges[0]
    t1=(np.random.rand(batch_size,1,1)-.5)*2* (tran_rate*2 + (1-tran_rate*2)*(1-s1)) # OR [h/2*(1-s1) + (tran_rate*h*s1)]
    t2=(np.random.rand(batch_size,1,1)-.5)*2* (tran_rate*2 + (1-tran_rate*2)*(1-s2))
    s1 = 1./s1
    s2 = 1./s2
    affine_matrix = torch.from_numpy(np.concatenate(
        (np.concatenate((s1*np.cos(a), -s2*np.sin(a), t1), axis = 2),
            np.concatenate((s1*np.sin(a), s2*np.cos(a), t2), axis = 2)),
            axis = 1))
    affine_matrix = affine_matrix.float()
    mapping = F.affine_grid(affine_matrix, (batch_size,c,h,w), align_corners=True)
    mapping_m = F.affine_grid(affine_matrix, (batch_size,1,h,w), align_corners=True)
    im_deformed = F.grid_sample(im, mapping, mode='bilinear', padding_mode='border', align_corners=True)
    mask_deformed = F.grid_sample(mask, mapping_m, mode='bilinear', padding_mode='border', align_corners=True)
    return im_deformed,mask_deformed    


        
def vec2cor2D(vector, original):
    vector[:,:,:,[0,-1]] = 0
    vector[:,:,[0,-1],:] = 0
    sour_coor = vector + original
    sour_coor = sour_coor.permute((0, 2, 3, 1))
    return sour_coor

def visualize_re(IM, MASK, MASK_PRED, MAPPING, file_name):
    IM = IM.data.cpu().numpy()
    IM = (np.transpose( IM, (0,2,3,1))*0.5 + 0.5)*255
    MASK = MASK.data.cpu().numpy()
    MASK = (np.transpose(MASK, (0,2,3,1)))*255
    MASK_PRED = MASK_PRED.data.cpu().numpy()
    MASK_PRED = (np.transpose( MASK_PRED, (0,2,3,1)))*255
    MAPPING = MAPPING.data.cpu().numpy()
    n = IM.shape[0]
    for i in range(n):        
        cv2.imwrite('result_IM/' + file_name + str(i) + '_IM.jpg', IM[i,:,:,:])
        cv2.imwrite('result_IM/' + file_name + str(i) + '_MASK.jpg', MASK[i,:,:,:])
        cv2.imwrite('result_IM/' + file_name + str(i) + '_PRED.jpg', MASK_PRED[i,:,:,:])
        np.save('result_IM/' + file_name + str(i) + '.npy',MAPPING[i,:,:,:])

def save_pair(IM,MASK,path,count):    
    IM = np.transpose(IM, (0,2,3,1))
    MASK = np.transpose(MASK, (0,2,3,1))
    N=IM.shape[0]
    for i in range(N):
        cv2.imwrite(path + '%d_%dIM.jpg'%(count,i), 255*IM[i,:,:,:])
        cv2.imwrite(path + '%d_%dMASK.jpg'%(count,i), 255*np.round(MASK[i,:,:,:]))

def plot_grid(grid, path,count):
    GRID = grid
    N=grid.shape[0]
    for i in range(N):
        fig = plt.figure(figsize=(16, 16), dpi=150)
        segs1f = GRID[i,:,:,:]
        segs2f = segs1f.transpose(1, 0, 2)
        plt.gca().add_collection(LineCollection(segs1f))
        plt.gca().add_collection(LineCollection(segs2f))
        plt.gca().axis('equal')
        fig.savefig(path + '%d_%dGRID.png'%(count,i))
        plt.close()
