import logging
import os
import sys
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms,datasets

from utils.cnnclass import VGG
from utils.cnn2d import EstimateCNN
from utils.loss2D import BCLossFunc,LAPLossFunc
from utils.utils import sample_grid, vec2map
from utils.fileIO import creat_folder, get_args, loss_recorder, save_result

task_name = 'Cifar/'
size = [32, 32]
dir_ckp = task_name+'out_ckp/'
dir_log = task_name+'out_log/'
dir_img = task_name+'out_img/'
creat_folder(dir_ckp)
creat_folder(dir_log)
creat_folder(dir_img)

def train_net(Estimator,
              Classifier,
              args):  
    pathname = '../scratch/Cifar_data/'###############NEED TO BE REVISED ACCORDINGLY###########
    depth_hidden = args.depth_hidden
    depth_down = args.depth_down
    batch_size = args.batch_size
    device = args.device
    epochs = args.epochs
    ratio = args.ratio
    lr = args.lr
    a1 = args.lambda1
    a2 = args.lambda2
    Nr = 1000
    #pathname = args.pathname
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size)])
    trainset = datasets.CIFAR10(root=pathname, train=True,download=True, transform=transform)###############NEED TO BE REVISED ACCORDINGLY###########
    testset = datasets.CIFAR10(root=pathname, train=False, download=True, transform=transform)###############NEED TO BE REVISED ACCORDINGLY###########
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
               device.type, n_train, n_test, a1,a2))
    origin_grid = sample_grid(device, size)
    origin_key = sample_grid(device, [int(size[0]/2**depth_down),int(size[1]/2**depth_down)])
    optimizerE = torch.optim.RMSprop(Estimator.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    optimizerC = torch.optim.Adam(Classifier.parameters(), lr=1e-3)
    criterion_Main = torch.nn.CrossEntropyLoss()
    criterion_BelC = BCLossFunc(size).to(device)
    criterion_Lapl = LAPLossFunc(size).to(device)
    
    BESTacc = 0.0
    trainF = open(dir_log+ 'train_ds%d_lr%.0E.csv'%(depth_hidden,lr), 'w+')
    trainF.write('Epoch,LOSS E,Main E,BelC E,Lapl E\n')
    trainF.flush()
    testF = open(dir_log+ 'test_ds%d_lr%.0E.csv'%(depth_hidden,lr), 'w+')
    testF.write('Epoch,LOSS E,Main E,BelC E,Lapl E\n')
    testF.flush()
    for epoch in range(epochs):
        with tqdm(total=n_train, desc='Epoch {}/{}'.format(epoch, epochs), unit='img') as pbar:
            if epoch < ratio * Nr:
                Estimator.eval()
                Classifier.train()
            else:
                Estimator.train()
                Classifier.eval()

            running_loss = 0.0
            running_main = 0.0
            running_belc = 0.0
            running_lapl = 0.0
            for count,(IM, label) in enumerate(train_loader):###############NEED TO BE REVISED ACCORDINGLY###########
                n_sample = IM.shape[0]
                IM = IM.to(device=device, dtype=torch.float32) 
                label = label.to(device=device) 
            
                vector = 2*(Estimator(IM)-.5)
                mapping = vec2map(vector,origin_grid)   
                IMt = F.grid_sample(IM, mapping, mode='bilinear', padding_mode='border', align_corners=True)
                predict = Classifier(IMt)

                loss_main = criterion_Main(predict, label)
                loss_belc = criterion_BelC(mapping)
                loss_lapl = criterion_Lapl(mapping)
                loss = loss_main + a1*loss_belc + a2*loss_lapl
                running_loss += n_sample*loss.item()
                running_main += n_sample*loss_main.item()
                running_belc += n_sample*loss_belc.item()
                running_lapl += n_sample*loss_lapl.item()
                
                pbar.set_postfix(**{'loss (B)': loss.item(),
                                    'Main (B)': loss_main.item(),
                                    'BClo (B)': loss_belc.item(),
                                    'Lapl (B)': loss_lapl.item()})   
                if epoch < ratio*Nr:
                    optimizerC.zero_grad()
                    loss.backward()
                    optimizerC.step()
                else:
                    optimizerE.zero_grad()
                    loss.backward()
                    optimizerE.step()
                pbar.update(batch_size)    

            loss_recorder(trainF,epoch,'TRAIN',
                            running_loss / n_train,running_main / n_train,
                            running_belc / n_train,running_lapl / n_train)

        with torch.no_grad():
            Estimator.eval()
            Classifier.eval()
            running_loss = 0.0
            running_main = 0.0
            running_belc = 0.0
            running_lapl = 0.0
            running_accc = 0.0
            IM_list = []
            MAP_list = []
            NAME_LIST = ()
            for count,(IM, label) in enumerate(test_loader): 
                n_sample = IM.shape[0]  
                IM = IM.to(device=device, dtype=torch.float32) 
                label = label.to(device=device) 
            
                vector = 2*(Estimator(IM)-.5)
                mapping = vec2map(vector,origin_grid)   
                IMt = F.grid_sample(IM, mapping, mode='bilinear', padding_mode='border', align_corners=True)
                predict = Classifier(IMt)

                loss_main = criterion_Main(predict, label)
                loss_belc = criterion_BelC(mapping)
                loss_lapl = criterion_Lapl(mapping)
                loss = loss_main + a1*loss_belc + a2*loss_lapl

                running_loss += n_sample*loss.item()
                running_main += n_sample*loss_main.item()
                running_belc += n_sample*loss_belc.item()
                running_lapl += n_sample*loss_lapl.item()

                running_accc += (torch.max(predict,axis=1)[1]==(label)).sum()

            loss_recorder(testF,epoch,'TEST',
                            running_loss / n_test,running_main / n_test,
                            running_belc / n_test,running_lapl / n_test,
                            running_accc / n_test,0,0,0)
        
            if BESTacc < running_accc/n_test:
                BESTacc = running_accc/n_test
                if epoch < ratio * Nr:
                    torch.save(Classifier.state_dict(),dir_ckp + 'epoch{}_C.pth'.format(epoch))
                else:
                    torch.save(Estimator.state_dict(),dir_ckp + 'epoch{}_E.pth'.format(epoch))
                logging.info('Checkpoint {} saved !'.format(epoch))   

    trainF.close()     
    testF.close()

    



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True
    logging.info('Using device {}'.format(args.device))
    
    Estimator = EstimateCNN(n_input = 3, n_output = 2, depth_down = args.depth_down, depth_hidden = args.depth_hidden)
    Classifier = VGG()###############NEED TO BE REVISED ACCORDINGLY###########
    if args.unload:
        Estimator.load_state_dict(
            torch.load(args.unload,map_location=args.device))
        logging.info('Mapping Estor loaded from {}'.format(args.unload))
    Estimator.to(device=args.device)
    Classifier.to(device=args.device)

    train_net(Estimator=Estimator,Classifier=Classifier,args=args)

