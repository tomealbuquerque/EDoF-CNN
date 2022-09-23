# =============================================================================
# Create folders with images per stack 
# =============================================================================

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', choices=['cervix93'], default='cervix93')
parser.add_argument('--image_size', choices=[640], default=640)
parser.add_argument('--method', choices=[
    'EDOF_CNN_max','EDOF_CNN_3D','EDOF_CNN_concat','EDOF_CNN_backbone','EDOF_CNN_modified'], default='EDOF_CNN_modified')
parser.add_argument('--Z', choices=[3,5,7,9], type=int, default=7)
parser.add_argument('--fold', type=int, choices=range(3),default=1)
parser.add_argument('--epochs', type=int, default=75)
parser.add_argument('--batchsize', type=int, default=2)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--cudan', type=int, default=1)
args = parser.parse_args()


import numpy as np
from time import time
from torch import optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import KFold
import torch
import dataset, models
from tqdm import tqdm
from PIL import Image
from matplotlib import cm
import os 

device = torch.device('cuda:'+str(args.cudan) if torch.cuda.is_available() else 'cpu')


tr_ds = dataset.Dataset('train', dataset.aug_transforms, args.dataset, args.Z, args.fold)
tr = DataLoader(tr_ds, args.batchsize, True,  pin_memory=True)
ts_ds = dataset.Dataset('test', dataset.val_transforms, args.dataset, args.Z, args.fold)
ts = DataLoader(ts_ds, args.batchsize,False,  pin_memory=True)

#to view images
tst = DataLoader(ts_ds, 1,False,  pin_memory=True)


prefix = '-'.join(f'{k}-{v}' for k, v in vars(args).items())


if args.method=='EDOF_CNN_max':
    model = models.EDOF_CNN_max()
elif args.method=='EDOF_CNN_3D':
    model = models.EDOF_CNN_3D(args.Z)
elif args.method=='EDOF_CNN_backbone':
    model = models.EDOF_CNN_backbone()
elif args.method=='EDOF_CNN_modified':
    model = models.EDOF_CNN_modified()
else: 
    model = models.EDOF_CNN_concat()


model.load_state_dict(torch.load('results\\Baseline_results_without_TVL\\'+str(prefix)+'.pth'))
model = model.to(device)


def save_image_stacks():
    Yhats=[]
    Ytrues=[]
    stacks=[]
    model.eval()
    with torch.no_grad():
        for XX, Y in tst:
              XX = [X.to(device) for X in XX]
              Y = Y.to(device, torch.float)
              Yhat = model(XX)
              Yhats.append(Yhat[0].cpu().numpy())
              Ytrues.append(Y[0].cpu().numpy())
              stacks.append([z.cpu().numpy() for z in XX])
              
    for i in range(len(tst)):
        path = 'test_images_for_EDOF\\image_'+str(i)
        os.makedirs(path, exist_ok=True)
        stack = stacks[i]
        for s in range(args.Z):
            stack0 = Image.fromarray(stack[s][0,0,:,:]* 255)
            if stack0.mode != 'RGB':
                stack0 = stack0.convert('RGB')
            stack0.save('test_images_for_EDOF\\image_'+str(i)+'\\image_'+str(i)+'_stack_'+str(s)+'.png')
    
        x = np.moveaxis(Yhats[i], 0,2 )
        xt = np.moveaxis(Ytrues[i], 0,2 )
        x = x[:, :, 0]
        xt = xt[:, :, 0]
        # img = Image.fromarray(np.uint8(x*255), 'RGB')
        img = Image.fromarray(x* 255)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        # img.save('image/teste'+str(i)+'_epoch_'+str(epochv)+'.png')
        img.save('test_images_for_EDOF\\image_pred_edof_'+str(i)+'.png')
        # imgt = Image.fromarray(np.uint8(xt*255), 'RGB')
        imgt = Image.fromarray(xt* 255)
        if imgt.mode != 'RGB':
            imgt = imgt.convert('RGB')
        imgt.save('test_images_for_EDOF\\image_full_edof_'+str(i)+'.png')


save_image_stacks()