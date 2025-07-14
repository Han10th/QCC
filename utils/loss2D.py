import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.measure import label
import numpy as np


class BCLossFunc(torch.nn.Module):
    def __init__(self, size):
        super(BCLossFunc, self).__init__()
        self.hx1 = torch.tensor(2.0 / size[0])
        self.hx2 = torch.tensor(2.0 / size[1])

    def forward(self, mapping):
        u_SE, v_SE = mapping[:, 1:, 1:, 0:1], mapping[:, 1:, 1:, 1:2]
        u_NE, v_NE = mapping[:, 0:-1, 1:, 0:1], mapping[:, 0:-1, 1:, 1:2]
        u_SW, v_SW = mapping[:, 1:, 0:-1, 0:1], mapping[:, 1:, 0:-1, 1:2]
        u_NW, v_NW = mapping[:, 0:-1, 0:-1, 0:1], mapping[:, 0:-1, 0:-1, 1:2]
        # forward FDM
        u_x1f = (u_SE - u_SW) / self.hx1
        u_x2f = (u_SE - u_NE) / self.hx2
        v_x1f = (v_SE - v_SW) / self.hx1
        v_x2f = (v_SE - v_NE) / self.hx2
        numerator0 = (u_x1f ** 2 + v_x1f ** 2 - v_x2f ** 2 - u_x2f ** 2) ** 2 + (
                    2 * u_x2f * u_x1f + 2 * v_x1f * v_x2f) ** 2
        dedomenator0 = ((u_x1f + v_x2f) ** 2 + (v_x1f - u_x2f) ** 2) ** 2
        mu_square0 = numerator0 / dedomenator0
        # backward FDM
        u_x1b = (u_NE - u_NW) / self.hx1
        u_x2b = (u_SW - u_NW) / self.hx2
        v_x1b = (v_NE - v_NW) / self.hx1
        v_x2b = (v_SW - v_NW) / self.hx2
        numerator1 = (u_x1b ** 2 + v_x1b ** 2 - v_x2b ** 2 - u_x2b ** 2) ** 2 + (
                    2 * u_x2b * u_x1b + 2 * v_x1b * v_x2b) ** 2
        dedomenator1 = ((u_x1b + v_x2b) ** 2 + (v_x1b - u_x2b) ** 2) ** 2
        mu_square1 = numerator1 / dedomenator1

        return (torch.exp(mu_square0 + mu_square1) - 1).mean()


class LAPLossFunc(torch.nn.Module):
    def __init__(self, size):
        super(LAPLossFunc, self).__init__()
        kernel = torch.tensor([[0., 1., 0.],
                               [1., -4., 1.],
                               [0., 1., 0.]]).unsqueeze(0).unsqueeze(0)
        self.weight = torch.nn.Parameter(data=kernel, requires_grad=False)
        self.hx1 = torch.tensor(2.0 / size[0])
        self.hx2 = torch.tensor(2.0 / size[1])

    def forward(self, x):
        x1 = x[:, :, :, 0]
        x2 = x[:, :, :, 1]
        x1 = F.conv2d(x1.unsqueeze(1), self.weight, padding=0) / (self.hx1)
        x2 = F.conv2d(x2.unsqueeze(1), self.weight, padding=0) / (self.hx2)
        return (torch.cat([x1, x2], dim=1) ** 2).mean()


class detDFunc(nn.Module):
    def __init__(self, device, size):
        super(detDFunc, self).__init__()
        self.detD_loss = detD2d(device, int(size[0]), int(size[1]))
    def forward(self, mapping):
        # pred = convert_map(mapping)
        return self.detD_loss(mapping)
def detD2d(device, H, W):
    h, w = H-1, W-1
    hstep, wstep = 2/h, 2/w
    step = torch.tensor([wstep, hstep]).to(device=device)
    relu =torch.nn.ReLU()
    def detD_loss(mapping):
        """
        Inputs:
            mapping: (N, 2, h, w), torch tensor
        Outputs:
            loss: (N, (h-1)*(w-1)*2), torch tensor
        """
        # N, H, W, C = mapping.shape
        mappingOO = mapping[:,0:h,0:w,:]
        mappingPO = mapping[:,0:h,1:w+1,:]
        mappingOP = mapping[:,1:h+1,0:w,:]     
        F_PO = (mappingPO - mappingOO)/step[0]
        F_OP = (mappingOP - mappingOO)/step[1]   
        det = F_PO[:,:,:,0]*F_OP[:,:,:,1] - F_PO[:,:,:,1]*F_OP[:,:,:,0]
        loss = torch.mean(relu(-det))
        return loss #mu
    return detD_loss


class DOTlossFunc(torch.nn.Module):
    def __init__(self):
        super(DOTlossFunc, self).__init__()

    def forward(self, x, y=None):
        x = x / ((x**2).sum(dim=1, keepdim=True) + 1e-15).sqrt()
        if not y: y = x.roll(1,0)
        M = x.shape[-1]
        out2 = M - y.permute(0,2,1) @ x.sum(dim=2, keepdim=True)
        out1 = M - x.permute(0,2,1) @ y.sum(dim=2, keepdim=True)
        rl = 2*(1-(x*y).sum(dim=1)) / (out1 + out2).squeeze(-1)
        return rl.sum(-1).mean()