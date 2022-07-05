from torch.utils.data import Dataset
from torchvision import models, transforms
import pickle
import torch
import numpy as np
import cv2
import glob
from PIL import Image, ImageOps
import PIL
import os 
from utils_files.automatic_brightness_and_contrast import automatic_brightness_and_contrast

class Dataset(Dataset):
    def __init__(self, type, transform, dataset, Z, fold):
        self.X, self.Y = pickle.load(open(f'dataset\\data_{dataset}_zstacks_{Z}.pickle', 'rb'))[fold][type]
        self.transform = transform
        self.Z = Z
    def __getitem__(self, i):
        X0 = self.transform(self.X[i][0])
        X1 = self.transform(self.X[i][1])
        X2 = self.transform(self.X[i][2])

     
        if self.Z==5:
            X3 = self.transform(self.X[i][3])
            X4 = self.transform(self.X[i][4])
        elif self.Z==7:
            X3 = self.transform(self.X[i][3])
            X4 = self.transform(self.X[i][4])
            X5 = self.transform(self.X[i][5])
            X6 = self.transform(self.X[i][6])
        elif self.Z==9:
            X3 = self.transform(self.X[i][3])
            X4 = self.transform(self.X[i][4])
            X5 = self.transform(self.X[i][5])
            X6 = self.transform(self.X[i][6])
            X5 = self.transform(self.X[i][5])
            X6 = self.transform(self.X[i][6])
            X7 = self.transform(self.X[i][7])
            X8 = self.transform(self.X[i][8])

        Y = self.transform(self.Y[i])
        
        if self.Z==3:
            X_all=[X0, X1, X2]
        elif self.Z==5:
            X_all=[X0, X1, X2, X3, X4]
        elif self.Z==7:
            X_all=[X0, X1, X2, X3, X4, X5, X6]
        elif self.Z==9:
            X_all=[X0, X1, X2, X3, X4, X5, X6, X7, X8]
            
        r=torch.randperm(self.Z)
        
        #return X_all, Y
        return [X_all[u] for u in r],Y

    def __len__(self):
        return len(self.X)




class Dataset_folder(Dataset):
    def __init__(self, transform, path, Z, img_size):
        self.path = path
        X_STACKS_per_folder=[]
        for idxx,image_name in enumerate(glob.glob(os.path.join(self.path, "*.jpg"))):
            im = cv2.imread(image_name)
            imnew=cv2.resize(im,(img_size,img_size))
            X_STACKS_per_folder.append(imnew)
        
        self.X = np.array(X_STACKS_per_folder)
        self.X = np.expand_dims(self.X, axis=0)
        self.transform = transform
        self.Z = Z
        
    def __getitem__(self, i):
        X0 = self.transform(self.X[i][0])
        X1 = self.transform(self.X[i][1])
        X2 = self.transform(self.X[i][2])

     
        if self.Z==5:
            X3 = self.transform(self.X[i][3])
            X4 = self.transform(self.X[i][4])
        elif self.Z==7:
            X3 = self.transform(self.X[i][3])
            X4 = self.transform(self.X[i][4])
            X5 = self.transform(self.X[i][5])
            X6 = self.transform(self.X[i][6])
        elif self.Z==9:
            X5 = self.transform(self.X[i][5])
            X6 = self.transform(self.X[i][6])
            X7 = self.transform(self.X[i][7])
            X8 = self.transform(self.X[i][8])

        # Y = self.transform(self.Y[i])
        
        if self.Z==3:
            X_all=[X0, X1, X2]
        elif self.Z==5:
            X_all=[X0, X1, X2, X3, X4]
        elif self.Z==7:
            X_all=[X0, X1, X2, X3, X4, X5, X6]
        elif self.Z==9:
            X_all=[X0, X1, X2, X3, X4, X5, X6, X7, X8]
            
        r=torch.randperm(self.Z)
        
        #return X_all, Y
        return [X_all[u] for u in r]

    def __len__(self):
        return len(self.X)





aug_transforms = transforms.Compose([
    transforms.ToPILImage(),
    # transforms.Resize((512, 512)),
    # automatic_brightness_and_contrast(clip_hist_perc=0.5),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    # vgg normalization
    # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

val_transforms = transforms.Compose([
    transforms.ToPILImage(),
    #  transforms.Resize((512, 512)),
    # automatic_brightness_and_contrast(clip_hist_perc=0.5),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),  # vgg normalization
    # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])


aug_transforms_rgb = transforms.Compose([
    transforms.ToPILImage(),
    # transforms.Resize((512, 512)),
    # automatic_brightness_and_contrast(clip_hist_perc=0.5),
    transforms.ToTensor(),
    # vgg normalization
    # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

val_transforms_rgb = transforms.Compose([
    transforms.ToPILImage(),
    #  transforms.Resize((512, 512)),
    # automatic_brightness_and_contrast(clip_hist_perc=0.5),
    transforms.ToTensor(),  # vgg normalization
    # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])



# if __name__ == '__main__':

# ds = Dataset('test', aug_transforms, 'cervix93', 7, 0)

# X0, X1, X2, X3, X4 = ds[0]

# dsss=ds[1][0]
