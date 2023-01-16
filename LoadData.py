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
  
# Load Train Data
train_image = []
mask_image = []


address='C:/Users/zsn/Desktop/Figshare/Data/Fold4_is_Test/IMG/Enhanced/'
address_mask='C:/Users/zsn/Desktop/Figshare/Data/Fold4_is_Test/GT//'
for filename in sorted(glob.glob(address+"*.bmp")):
    print(filename)
    imgName = filename.split("\\")[-1] 
    img=plt.imread(filename)
    train_image.append(img)
    #Gth
    mask=plt.imread(address_mask+imgName)
    mask=mask[:,:,0]
    mask=mask/255
    mask_image.append(mask)    
    
train_image = np.array(train_image)
mask_image=np.array(mask_image)
FinaltrainData=train_image.astype(np.uint8)
FinaltrainData = FinaltrainData.reshape((FinaltrainData.shape[0], 512, 512, 1))
mask_image = mask_image.reshape((mask_image.shape[0], 512, 512, 1))
FinaltrainMask=mask_image.astype(np.float32)

testData = []
testMask=[]
lablesTest = []
address='C:/Users/zsn/Desktop/Figshare/Data/fold4/IMG//'
address_mask_test='C:/Users/zsn/Desktop/Figshare/Data/Fold4/GT//'
for filename in sorted(glob.glob(address+"*.bmp")):
    print(filename)
    imgName = filename.split("\\")[-1]  
    img=plt.imread(filename)
    testData.append(img)
    #load Mask
    mask=plt.imread(address_mask_test+imgName)
    mask=mask[:,:,0]
    mask=mask/255
    testMask.append(mask)       
    
testData = np.array(testData)
testMask=np.array(testMask)
FinaltestData = testData.reshape((testData.shape[0], 512, 512, 1))
FinaltestMask= testMask.reshape((testMask.shape[0], 512, 512, 1))
FinaltestMask = FinaltestMask.astype(np.float32)
FinaltestData=FinaltestData.astype(np.uint8)
