import math
import csv
import matplotlib.pyplot as plt
import numpy as np
import cv2
import scipy as sp
from skimage.measure import label, regionprops
from skimage.transform import rotate
from scipy import ndimage
import glob
from keras.models import Model
import os
from skimage.morphology import convex_hull_image
from sympy import Point, Line, pi

address_data='E:/Mutual papers/Brain/Fold5 is Test/DataTrain/IMG//'
address_test='E:/Mutual papers/Brain/Fold5 is Test/fold5/IMG//'
address_res_data='E:/Mutual papers/Brain/Fold5 is Test/DataTrain/Map//'
address_res_test='E:/Mutual papers/Brain/Fold5 is Test/fold5/Map//'
my_model = Multi_Task_Brain (weights = 'C:/Users/zsn/Desktop/Figshare/Data/Fold4_is_Test/Logs//val_dice_coef_192-0.96.h5')
#Evaluvating with test dataset
Outputsss={ 'segmentation': FinaltestMask, 'classification':Test_Lable} 
score=my_model.evaluate([FinaltestData , FinalMapTest],Outputsss,batch_size=10)
print("Test:accuaracy", str(score[1]*100))
print("Test: Dice ",str(score[0]*100))

for filename in sorted(glob.glob(address_test+"*.bmp")):
    print(filename)
    imgName = filename.split("\\")[-1]
    img = plt.imread(filename)
    img_test= np.empty(shape=(1,512,512,1),dtype=np.uint8) 
    img_test[0,:,:,0] = img
    net_output = my_model.predict(img_test)
    out_seg = net_output
    out_seg = out_seg[0,:,:,0]
#    out_seg[out_seg>=0.5]=1
#    out_seg[out_seg<0.5]=0
    out_seg *=255
    out_seg = out_seg.astype(np.uint8)
    cv2.imwrite(address_res_test+imgName[:-4]+"_output.bmp" , out_seg)