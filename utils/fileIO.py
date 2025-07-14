import os
import csv
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.io import loadmat,savemat
def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=2000,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=20,
                        help='Batch size', dest='batch_size')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('-un', '--unet', dest='unload', type=str, default=False,
                        help='Load model from a unet.pth file')
    parser.add_argument('-qcc', '--qcc', dest='qcc', type=str, default=False,
                        help='Load model from for QC conv net')
    parser.add_argument('-w', '--weight', dest='weight', type=float, default=0,
                        help='The weight of the custom loss')
    parser.add_argument('-dd', '--depth_down', dest='depth_down', type=int, default=2,
                        help='The downsample depth')
    parser.add_argument('-dh', '--depth_hidden', dest='depth_hidden', type=int, default=0,
                        help='The hidden depth')
    parser.add_argument('-r', '--ratio', dest='ratio', type=float, default=0.10,
                        help='The ratio of epochs for alternation')
    parser.add_argument('-l1', '--lambda1', dest='lambda1', type=float, default=0.01,
                        help='The weight of the Jacobian loss')
    parser.add_argument('-l2', '--lambda2', dest='lambda2', type=float, default=0.0001,
                        help='The weight of the Laplacian loss')
    return parser.parse_args()

class QccRecorder:
    def __init__(self, args, task_name, field_names=None, writemode = 'w+'):
        self.taskname = task_name
        self.dir_ckp = creat_folder(task_name+'out_ckp/')
        self.dir_log = creat_folder(task_name+'out_log/')
        self.dir_img = creat_folder(task_name+'out_img/')

        if field_names is not None:
            self.Field_Names = field_names
            self.trainF = open(self.dir_log + 'train_lr%.0E.csv' % (args.lr), writemode)
            self.trainW = csv.DictWriter(self.trainF, fieldnames=field_names)
            self.trainW.writeheader()
            self.testF = open(self.dir_log + 'test_lr%.0E.csv' % (args.lr), writemode)
            self.testW = csv.DictWriter(self.testF, fieldnames=field_names)
            self.testW.writeheader()
            self.NofField = len(field_names)
    def get_NofField(self):
        return self.NofField
    def save_ckp(self, epoch, Estimator, Descriptor):
        torch.save(Estimator.state_dict(),
                   self.dir_ckp + 'epoch_Est_{}.pth'.format(epoch))
        torch.save(Descriptor.state_dict(),
                   self.dir_ckp + 'epoch_Des_{}.pth'.format(epoch))
    def save_out(self, epoch, output):
        1+1
    def save_log(self, phase, epoch, losses):
        loss_dict = { "Epoch": epoch }
        for name,value in zip(self.Field_Names, losses):
            loss_dict[name] = value
        if phase == 'TRAIN':
            self.trainW.writerows([losses])
        elif phase == 'TEST':
            self.testW.writerows([losses])

def switch_mode(models, modes=[False, False]):
    if isinstance(modes, bool): modes = [modes]*2
    for model, mode in zip(models, modes):
        model.train() if mode else model.eval()
def creat_folder(foldername):
    if not os.path.exists(foldername):
        os.makedirs(foldername)
        print("The new directory is created :  " + foldername)
    return foldername
def loss_recorder(fileIO, epoch, name, loss_total, loss_eren, loss_belc, loss_lapl, metr_accc=0.0, metr_reca=0.0, metr_prec=0.0, metr_spec=0.0):
    if name == 'TEST':
        print(name + '-[%d]     Loss: %.5f     Cros: %.5f    Belt: %.5f    Lapl: %.5f'%(
            epoch,loss_total,loss_eren,loss_belc,loss_lapl))
        print('                Accc: %02.3f    Reca: %.5f    Prec: %.5f    Spec: %.5f'%(
            metr_accc, metr_reca, metr_prec, metr_spec))
    fileIO.write('{},{},{},{},{},{}\n'.format( 
        epoch,loss_total,loss_eren,loss_belc,loss_lapl,metr_accc))
    fileIO.flush()

def save_result(IM_list,MAP_list,NAME_LIST,dir_img):
    IMarray = torch.concat(IM_list,axis=0)
    MAParray = torch.concat(MAP_list,axis=0)
    IMarray = IMarray.data.cpu().numpy()
    MAParray = MAParray.data.cpu().numpy()
    IMarray = np.transpose(IMarray, (0,2,3,1))
    N=IMarray.shape[0]
    for i in range(N):
        name = NAME_LIST[i]
        grid = MAParray[i,::2,::2,:]
        fig = plt.figure(figsize=(16, 16), dpi=30)
        cv2.imwrite(dir_img + name + '_image.jpg', 255*IMarray[i,:,:,0])        
        plt.gca().add_collection(LineCollection(grid))
        plt.gca().add_collection(LineCollection(grid.transpose(1, 0, 2)))
        plt.gca().axis('equal')
        fig.tight_layout()
        fig.savefig(dir_img + name + '_grid.jpg')
        plt.close()
        mdic = {"grid": grid, "image": 255*IMarray[i,:,:,0]}
        savemat(dir_img + name + '.mat', mdic)