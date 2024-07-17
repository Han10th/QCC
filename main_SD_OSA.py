import logging
import os
import sys

import numpy as np
from tqdm import tqdm
import oflibpytorch as of
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms,datasets

from utils.cnnclass import ClassifierMINI
from utils.unet2d import UNet2D
from utils.loss2D import DOTlossFunc,BCLossFunc,LAPLossFunc
from utils.utils import sample_grid, vec2map, OsaShapeLoader
from utils.fileIO import QccRecorder, get_args


def evaluate(Estimator, Descriptor, data, origin_grid, device):
    # criterion_Main = torch.nn.CrossEntropyLoss()
    criterion_DoTl = DOTlossFunc()
    criterion_BelC = BCLossFunc(size).to(device)
    criterion_Lapl = LAPLossFunc(size).to(device)

    IM, label, vertices2D = data
    IM = IM.to(device=device, dtype=torch.float32)
    vertices2D = 2 * (vertices2D.to(device=device, dtype=torch.float32) - 0.5)

    vector = 2 * (Estimator(IM) - .5)
    mapping = vec2map(vector, origin_grid)
    IMt = F.grid_sample(IM, mapping, mode='bilinear', padding_mode='border', align_corners=True)
    vertices2D = of.Flow(-vector, ref='t').track(vertices2D, int_out=False)
    # !!!!!!!!!! WHAT DOES THE ABOVE MEAN?

    shape = Descriptor(IMt)
    shape_vertex = F.grid_sample(shape, vertices2D.unsqueeze(2), mode='bilinear', padding_mode='border',
                                 align_corners=True).squeeze()

    loss_main = criterion_DoTl(shape_vertex)
    loss_belc = criterion_BelC(mapping)
    loss_lapl = criterion_Lapl(mapping)

    losses = [loss_main, loss_belc, loss_lapl]

    return mapping,losses
def train_net(Estimator,
              Descriptor,
              Recorder,
              args):
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.info('Using device {}'.format(args.device))

    pathname = '../scratch/osa_same/'
    batch_size = args.batch_size
    device = args.device
    epochs = args.epochs
    ratio = args.ratio
    Nr = 1000
    lr = args.lr
    a1,a2 = args.lambda1,args.lambda2
    # pathname = args.pathname
    transform = transforms.Compose([
        transforms.Resize(size)])

    origin_grid = sample_grid(device, size)
    trainset = OsaShapeLoader(pathname, phase='train', transform=transform)
    testset = OsaShapeLoader(pathname, phase='test', transform=transform)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=1)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=1)
    n_train = len(trainset)
    n_test = len(testset)
    print('''Starting training:
            Epochs:             {}
            Batch size:         {}
            Learn rate:         {}
            Run Device:         {}
            Train Size:         {}
            Test Size:          {}
            Coef Jaco:          {}
            Coef Lapl:          {}
        '''.format(epochs, batch_size, lr,
                   device.type, n_train, n_test, a1, a2))
    optimizerE = torch.optim.RMSprop(Estimator.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    optimizerC = torch.optim.Adam(Descriptor.parameters(), lr=1e-3)

    BESTacc = 999.9

    for epoch in range(epochs):
        with tqdm(total=n_train, desc='Epoch {}/{}'.format(epoch, epochs), unit='img') as pbar:

            PHASE_NAME = "TRAIN"
            TRAIN_MODE = (epoch < ratio * Nr)
            Estimator.eval() if TRAIN_MODE else Estimator.train()
            Descriptor.train() if TRAIN_MODE else Descriptor.eval()

            # NOTICE : No need to change the below :
            this_n = n_train if PHASE_NAME == 'TRAIN' else n_test
            this_loader = train_loader if PHASE_NAME == 'TRAIN' else n_test
            running_losses = np.array([0.0]*(Recorder.get_NofField()))
            for count, data in enumerate(this_loader):  ###############NEED TO BE REVISED ACCORDINGLY###########
                mapping, losses = evaluate(Estimator, Descriptor, data, origin_grid, device)
                loss = losses[0] + a1 * losses[1] + a2 * losses[2]
                running_losses += np.array([loss.item(),
                                            losses[0].item(),
                                            losses[1].item(),
                                            losses[2].item()
                                            ]) * len(data[0])/this_n

                optimizerC.zero_grad() if TRAIN_MODE else optimizerE.step()
                loss.backward()
                optimizerC.step() if TRAIN_MODE else optimizerE.zero_grad()

                pbar.set_postfix(**{'loss (B)': loss.item(),
                                    'Main (B)': losses[0].item(),
                                    'BClo (B)': losses[1].item(),
                                    'Lapl (B)': losses[2].item()})
                pbar.update(batch_size)

            Recorder.save_log(PHASE_NAME,epoch,running_losses)
            if BESTacc > running_losses[0]:
                Recorder.save_ckp(epoch, Estimator, Descriptor)
                BESTacc = running_losses[0]

            PHASE_NAME = "TEST"
            Estimator.eval()
            Descriptor.eval()
            this_n = n_train if PHASE_NAME == 'TRAIN' else n_test
            this_loader = train_loader if PHASE_NAME == 'TRAIN' else n_test
            running_losses = np.array([0.0]*(Recorder.get_NofField()))
            for count, data in enumerate(this_loader):  ###############NEED TO BE REVISED ACCORDINGLY###########
                mapping, losses = evaluate(Estimator, Descriptor, data, origin_grid, device)
                loss = losses[0] + a1 * losses[1] + a2 * losses[2]
                running_losses += np.array([loss.item(),
                                            losses[0].item(),
                                            losses[1].item(),
                                            losses[2].item()
                                            ]) * len(data[0])/this_n

            Recorder.save_ckp(epoch,Estimator,Descriptor)


if __name__ == '__main__':
    size = [128, 128]
    args = get_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True
    Recorder = QccRecorder(args, '../Result/OSAshape/', ["Total","MAIN","QC","LAP"])

    Estimator = UNet2D(n_input=5, n_output=2, depth_down=args.depth_down, depth_hidden=args.depth_hidden, IsSeg=True)
    Descriptor = UNet2D(n_input=5, n_output=16, depth_down=args.depth_down, depth_hidden=args.depth_hidden, IsSeg=False)
    Estimator.to(device=args.device)
    Descriptor.to(device=args.device)
    # if args.qcc: Estimator.load_state_dict(torch.load(args.qcc, map_location=args.device))

    train_net(Estimator=Estimator,
              Descriptor=Descriptor,
              Recorder=Recorder,
              args=args)
