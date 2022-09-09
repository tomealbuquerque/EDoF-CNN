import os
import glob
from os import listdir
from os.path import isfile, join, isdir
import shutil

mypath_new=r"E:\DATA_SETS\low_cost_microscopy_dataset\data_fraunhofer"
# mypath_comp ="E:\DATA_SETS\low_cost_microscopy_dataset\data_fraunhofer_elastic_aligned_elastic_only_edof"
mypath=r"E:\DATA_SETS\new_data_stacks\data_channel_position_aligned - Copy"


geral_dir = [f for f in listdir(mypath) if isdir(join(mypath, f))]

axis_path=[]

for gd in range(len(geral_dir)):
    axis_path=join(mypath,geral_dir[gd]) 
    axis_dir = [f for f in listdir(axis_path) if isdir(join(axis_path, f))]
    for ad in axis_dir:
        path=join(mypath,geral_dir[gd], ad)
        folder = os.path.split(path)[1]
        head = os.path.split(path)[0] 
        head = os.path.split(head)[-1]
        
        if os.path.exists(join(mypath_new,head,folder)):
            pass 
        else:
            shutil.rmtree(join(mypath,head,folder))
        