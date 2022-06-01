# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 10:24:27 2022

@author: albu
"""


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

device = torch.device('cuda:'+str(args.cudan) if torch.cuda.is_available() else 'cpu')



tr_ds = dataset.Dataset('train', dataset.aug_transforms, args.dataset, args.Z, args.fold)
tr = DataLoader(tr_ds, args.batchsize, True,  pin_memory=True)
ts_ds = dataset.Dataset('test', dataset.val_transforms, args.dataset, args.Z, args.fold)
ts = DataLoader(ts_ds, args.batchsize,False,  pin_memory=True)

#to view images
tst = DataLoader(ts_ds, 1,False,  pin_memory=True)


def view_images(epochv):
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
              
    from PIL import Image
    from matplotlib import cm
    
    for i in range(3):
        if args.epochs==75:
            stack = stacks[i]
            for s in range(args.Z):
                stack0 = Image.fromarray(stack[s][0,0,:,:]* 255)
                if stack0.mode != 'RGB':
                    stack0 = stack0.convert('RGB')
                stack0.save('teste_'+str(i)+'_stack_'+str(s)+'.png')
        
        x = np.moveaxis(Yhats[i], 0,2 )
        xt = np.moveaxis(Ytrues[i], 0,2 )
        x = x[:, :, 0]
        xt = xt[:, :, 0]
        # img = Image.fromarray(np.uint8(x*255), 'RGB')
        img = Image.fromarray(x* 255)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        # img.save('image/teste'+str(i)+'_epoch_'+str(epochv)+'.png')
        img.save('teste'+str(i)+'.png')
        # imgt = Image.fromarray(np.uint8(xt*255), 'RGB')
        imgt = Image.fromarray(xt* 255)
        if imgt.mode != 'RGB':
            imgt = imgt.convert('RGB')
        imgt.save('teste_true'+str(i)+'.png')



def test(val):
    model.eval()
    avg_loss_val = 0
    with torch.no_grad():
        for XX, Y in val:
            XX = [X.to(device, torch.float) for X in XX]
            Y = Y.to(device, torch.float)
            Yhat = model(XX)
            loss = model.loss(Yhat, Y.to(torch.float))
            avg_loss_val += loss / len(val)
    return avg_loss_val



def train(tr, val, epochs=args.epochs, verbose=True):
    for epoch in range(epochs):
        if verbose:
            print(f'* Epoch {epoch+1}/{args.epochs}')
        tic = time()
        model.train()
        avg_acc = 0
        avg_loss = 0
        for XX, Y in tr:
            XX = [X.to(device, torch.float) for X in XX]
            Y = Y.to(device, torch.float)
            opt.zero_grad()
            Yhat = model(XX)
            loss = model.loss(Yhat, Y)
            loss.backward()
            opt.step()
            avg_loss += loss / len(tr)

        dt = time() - tic
        out = ' - %ds - Loss: %f' % (dt, avg_loss)
        if val:
            model.eval()
            out += ', Test loss: %f' % test(val)
        if verbose:
            print(out)
        scheduler.step(avg_loss)
        
        # view_images(epoch)

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


opt = optim.Adam(model.parameters(), args.lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, verbose=True,patience=5)
train(tr, ts)


# torch.save(model.state_dict(), 'results\\'+str(prefix)+'.pth')

   
#print some metrics 
def predict_metrics(data):
    model.eval()
    Phat = []
    Y_true=[]
    with torch.no_grad():
        for XX, Y in data:
            XX = [X.to(device, torch.float) for X in XX]
            Y = Y.to(device, torch.float)
            Yhat = model(XX)
            Phat += list(Yhat.cpu().numpy())
            Y_true += list(Y.cpu().numpy())
    return Y_true, Phat



from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, normalized_root_mse 


data_test = DataLoader(ts_ds, 1,False,  pin_memory=True)
Y_true, Phat = predict_metrics(data_test)

mse = np.mean([mean_squared_error(Y_true[i], Phat[i]) for i in range(len(Y_true))])
rmse = np.mean([normalized_root_mse(Y_true[i], Phat[i]) for i in range(len(Y_true))])
ssim =np.mean([ssim(Y_true[i], Phat[i],channel_axis=0) for i in range(len(Y_true))]) 
psnr =np.mean([peak_signal_noise_ratio(Y_true[i], Phat[i]) for i in range(len(Y_true))]) 



f = open('results\\'+ str(prefix)+'.txt', 'a+')
f.write('\n\nModel:'+str(prefix)+
    ' \nMSE:'+ str(mse)+
    ' \nRMSE:'+ str(rmse)+
    ' \nSSIM:'+str(ssim)+
    ' \nPSNR:'+ str(psnr))
f.close()


# def test_cyto(path_f='test_data_aligned',img_size=640):
#     cyto_ds = dataset.Dataset_folder(dataset.val_transforms, path_f , args.Z,img_size)
#     cyto_ts = DataLoader(cyto_ds, 1 ,False,  pin_memory=True)
#     model.eval()
#     avg_loss_val = 0
#     with torch.no_grad():
#         for XX in tqdm(cyto_ts):
#             print(XX)
#             XX = [X.to(device, torch.float) for X in XX]
#             Yhat = model(XX)
#     final_edf=Yhat.cpu().numpy()
#     img=Image.fromarray(final_edf[0,0,:,:]* 255)
#     if img.mode != 'RGB':
#         img = img.convert('RGB')
#     img.save('teste_cyto.png')