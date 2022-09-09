# =============================================================================
# code to generate tables from results\ .txt files
# 
# =============================================================================
import numpy as np

for m in (["EDOF_CNN_max","EDOF_CNN_3D","EDOF_CNN_fast","EDOF_CNN_pairwise"]):
    for s in ([3, 5]):
        mse=[]
        rmse=[]
        ssim=[]
        pnsr=[]
        for i in range(5):
            text_file = open("results_fraunhofer_elastic\\dataset-fraunhofer_elastic-image_size-512-method-"+str(m)+"-Z-"+str(s)+"-fold-"+str(i)+"-epochs-200-batchsize-6-lr-0.001-cudan-3-image_channels-grayscale.txt", "r")
            
           
            lines = text_file.readlines()
            mse.append(float(lines[3].split(":")[1]))
            rmse.append(float(lines[4].split(":")[1]))
            ssim.append(float(lines[5].split(":")[1]))
            pnsr.append(float(lines[6].split(":")[1]))
           
        msem=np.mean(mse)
        rmsem=np.mean(rmse)
        ssimm=np.mean(ssim)
        pnsrm=np.mean(pnsr)
        msed=np.std(mse)
        rmsed=np.std(rmse)
        ssimd=np.std(ssim)
        pnsrd=np.std(pnsr)
        
        
        print("&& "+str(s)+" & $"+str(round(msem,5))+" \pm "+ str(round(msed, 5))+"$ & $"+
              str(round(rmsem,5))+" \pm "+ str(round(rmsed, 5))+"$ & $"+
              str(round(ssimm*100,3))+" \pm "+ str(round(ssimd*100, 3))+"$ & $"+
              str(round(pnsrm,3))+" \pm "+ str(round(pnsrd, 3))+"$")