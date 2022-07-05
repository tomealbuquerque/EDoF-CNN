# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 13:15:24 2021

@author: albu
"""

from os import listdir
import os
from os.path import isfile, join, isdir
import pandas as pd
import numpy as np 
import pickle
from sklearn.model_selection import StratifiedKFold
mypath=r"data\\"

geral_dir = [f for f in listdir(mypath) if isdir(join(mypath, f))]

axis_path=[]
img_path=[]
images_list=[]
labels_list=[]
dataframe=[]

index=[]
multiclass_labels=[]
multiclass_images=[]
focus_images=[]

i=0
for i in range(len(geral_dir)):
    axis_path=join(mypath,geral_dir[i]) 
    axis_dir = [f for f in listdir(axis_path) if isdir(join(axis_path, f))]
    for z in range(len(axis_dir)):
        img_path=join(axis_path,axis_dir[z])
        onlyfiles = [f for f in listdir(img_path) if isfile(join(img_path, f))]
        images=[join(img_path,l) for l in onlyfiles if l!='focused_image.txt']
        images_list=np.concatenate((images_list, images), axis=0)
        print(images_list)

#         if images!=[]:
#             onlyfiles[-1]=='focused_image.txt'
#             focus_img=open(join(img_path,onlyfiles[-1])).read()
#             label_path=focus_img+'.jpg'
#             path_focus_img=join(img_path,label_path)
            
#             #get focus image index on images list 
#             for i in range(len(images)): 
#                 if images[i]==path_focus_img:
#                     index=i
            
#             #Put labels in images 0-focus 1,2,3,4,5...-out of focus
#             for j in range(index+1):
#                 multiclass_images.append(images[j])
#                 if index-j>11:
#                     alt=11
#                     multiclass_labels.append(alt)
#                 else:    
#                     multiclass_labels.append(index-j)

                
#             print(os.path.basename(images[int(index)]))
            
            
#         focus_images.append(path_focus_img)



# #Put labels in images 0-focus 1-out of focus
# X,Y=np.array(multiclass_images),np.array(multiclass_labels)


# skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=1234);

# data_dict=[{'train':(X[tr],Y[tr]),'test':(X[ts],Y[ts])} for tr,ts in skf.split(X,Y)]

# # final_array=np.column_stack((images_list,labels))

# file = open('data.pickle', 'wb')

# # dump information to that file
# pickle.dump(data_dict, file)

# # close the file
# file.close()
            