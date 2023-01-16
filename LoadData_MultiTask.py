from PIL import Image # used for loading images
import numpy as np
import os # used for navigating to image path
import imageio # used for writing images
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.preprocessing import image
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from tqdm import tqdm
import glob
import matplotlib.pyplot as plt
import os
from scipy import misc
from keras import backend as K

#Produce Dictionary
naming_dict = {} 

f = open("C:/Users/zsn/Desktop/Figshare/Data/Lables.csv", "r")
fileContents = f.read()
fileContents = fileContents.split('\n')
for i in range(len(fileContents)-1):
  fileContents[i] = fileContents[i].split(',')
  row=fileContents[i][1:]
  for j in range(0, len(row)): 
      row[j] = int(row[j])         
  naming_dict[fileContents[i][0]] = row
  
# Load Train Data
train_image = []
mask_image = []
lables = []
FinalMap = []
FinaltrainData = []
address='C:/Users/zsn/Desktop/Figshare/Data/Fold4_is_Test/IMG//'
address_mask='C:/Users/zsn/Desktop/Figshare/Data/Fold4_is_Test/GT//'
address_map='C:/Users/zsn/Desktop/Figshare/Data/Fold4_is_Test/Map//'
i=0
for filename in sorted(glob.glob(address+"*.bmp")):
    print(filename)
    imgName = filename.split("\\")[-1] 
    img=plt.imread(filename)
    mapp=plt.imread(address_map+imgName[:-4]+"_Map.bmp")
    FinalMap[i,:,:,0]= mapp.astype(np.uint8)
    FinaltrainData[i,:,:,0]= img.astype(np.uint8)
    FinaltrainData[i,:,:,1]= mapp.astype(np.uint8)
    #Gth
    mask=plt.imread(address_mask+imgName)
    mask=mask[:,:,0]
    mask=mask/255
    mask_image.append(mask)    
    #lables
    lables.append( naming_dict[str(imgName[:-4])])
    i = i +1 
train_image = np.array(train_image)
mask_image=np.array(mask_image)
y=np.array(lables)
FinaltrainData = FinaltrainData.reshape((FinaltrainData.shape[0], 256, 256, 2))
FinalMap = FinalMap.reshape((FinalMap.shape[0], 256, 256, 1))
mask_image = mask_image.reshape((mask_image.shape[0], 256, 256, 1))
FinaltrainMask=mask_image.astype(np.float32)

testData = []
testMask=[]
lablesTest = []
FinalMapTest = []
FinaltestData = []
address='C:/Users/zsn/Desktop/Figshare/Data/fold4/IMG//'
address_mask_test='C:/Users/zsn/Desktop/Figshare/Data/Fold4/GT//'
address_map='C:/Users/zsn/Desktop/Figshare/Data/Fold4/Map//'
i = 0 
for filename in sorted(glob.glob(address+"*.bmp")):
    print(filename)
    imgName = filename.split("\\")[-1]  
    img=plt.imread(filename)
    mapp=plt.imread(address_map+imgName[:-4]+"_Map.bmp")
    testData.append(img)
    FinalMapTest[i,:,:,0]= mapp.astype(np.uint8)
    FinaltestData[i,:,:,0]= img.astype(np.uint8)
    FinaltestData[i,:,:,1]= mapp.astype(np.uint8)
    #load Mask
    mask=plt.imread(address_mask_test+imgName)
    mask=mask[:,:,0]
    mask=mask/255
    testMask.append(mask)       
    lablesTest.append( naming_dict[str(imgName[:-4])])
    i = i+ 1
    
testData = np.array(testData)
Test_Lable=np.array(lablesTest)
testMask=np.array(testMask)
FinaltestData = FinaltestData.reshape((FinaltestData.shape[0], 256, 256, 2))
FinalMapTest= FinalMapTest.reshape((FinalMapTest.shape[0], 256, 256, 1))
FinaltestMask= testMask.reshape((testMask.shape[0], 256, 256, 1))
FinaltestMask = FinaltestMask.astype(np.float32)
FinaltestData=FinaltestData.astype(np.uint8)
