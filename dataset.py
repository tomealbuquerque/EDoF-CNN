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



class automatic_brightness_and_contrast:
    """Rotate by one of the given angles."""

    def __init__(self, clip_hist_perc):
        self.clip_hist_perc = clip_hist_perc

    def __call__(self, x):
        # x=np.array(x) 
        self.clip_hist_percent=self.clip_hist_perc 
        gray = cv2.cvtColor(np.float32(x), cv2.COLOR_BGR2GRAY)
        
        # Calculate grayscale histogram
        hist = cv2.calcHist([gray],[0],None,[256],[0,256])
        hist_size = len(hist)
        
        # Calculate cumulative distribution from the histogram
        accumulator = []
        accumulator.append(float(hist[0]))
        for index in range(1, hist_size):
            accumulator.append(accumulator[index -1] + float(hist[index]))
        
        # Locate points to clip
        maximum = accumulator[-1]
        self.clip_hist_percent *= (maximum/100.0)
        
        self.clip_hist_percent /= 2.0

        # Locate left cut
        minimum_gray = 0
        while accumulator[minimum_gray] <  self.clip_hist_percent:
            minimum_gray += 1
        
        # Locate right cut
        maximum_gray = hist_size -1
        while accumulator[maximum_gray] >= (maximum -  self.clip_hist_percent):
            maximum_gray -= 1
        
        # Calculate alpha and beta values
        alpha = 255 / (maximum_gray - minimum_gray)
        beta = -minimum_gray * alpha
        auto_result = cv2.convertScaleAbs(np.float32(x), alpha=alpha, beta=beta)
        
        image=Image.fromarray(np.uint8(auto_result)*255)
        R, G, B = image.split()
        new_image = PIL.Image.merge("RGB", (R, G, B))
        new_image = ImageOps.invert(new_image)
        return new_image

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


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
    # transforms.RandomAffine(180, (0, 0.1), (0.9, 1.1)),
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomVerticalFlip(),
    # transforms.ColorJitter(saturation=(1, 2.0)),
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






# if __name__ == '__main__':

# ds = Dataset('test', aug_transforms, 'cervix93', 7, 0)

# X0, X1, X2, X3, X4 = ds[0]

# dsss=ds[1][0]
