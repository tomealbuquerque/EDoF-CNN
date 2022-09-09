import time
import pyautogui
import subprocess
import os
from os import listdir
from os.path import join, isdir



mypath=r"E:\DATA_SETS\low_cost_microscopy_dataset\data_fraunhofer_elastic_aligned_elastic_only"

geral_dir = [f for f in listdir(mypath) if isdir(join(mypath, f))]

axis_path=[]

for gd in range(len(geral_dir)):
    axis_path=join(mypath,geral_dir[gd]) 
    axis_dir = [f for f in listdir(axis_path) if isdir(join(axis_path, f))]
    for ad in axis_dir:
        path=join(mypath,geral_dir[gd], ad)
        time.sleep(3)
        p=subprocess.Popen(["ImageJ\\ImageJ.exe"])
        time.sleep(6)
        
        
        # print(pyautogui.size())
        # pyautogui.moveTo(768, 209, duration = 1)
        # open File
        pyautogui.click(751, 58)
        # print(pyautogui.position())
        #Import
        pyautogui.click(768, 209)
        # import image stacks
        pyautogui.click(1100, 209)
        #click on browse
        pyautogui.click(1082, 135)
        pyautogui.press('delete')
        # ##write folder name
        pyautogui.write(path+'\\',interval=0.0025) 
        pyautogui.press('enter')
        # ##select OK
        pyautogui.click(984, 478)
        
        # #click on stacks stacks
        
        pyautogui.click(893, 141)
        
        time.sleep(1) # Sleep for 3 seconds
        # #select plugins
        pyautogui.click(1033, 55)
        # select extended depth of field
        pyautogui.click(1015, 320)
        pyautogui.click(1280, 313)
        # print(pyautogui.position())
        pyautogui.click(818, 162,clicks=5)
        pyautogui.click(819, 352,clicks=5)
        
        time.sleep(7)
        
        #open File
        pyautogui.click(751, 58)
        #open save as
        pyautogui.click(769, 350)
        #open save as jpg
        pyautogui.click(769, 350)
        #open save as jpg
        pyautogui.click(969, 350)
        pyautogui.press('enter')
        time.sleep(3)
        p.kill()
        print(path)
    