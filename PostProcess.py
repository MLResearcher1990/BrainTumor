
import cv2
import matplotlib.pyplot as plt
import numpy as np 
import math
import scipy as sp   
import scipy.ndimage
import os
import glob
from scipy import ndimage
import cv2 
import numpy as np 
import matplotlib.pyplot as plt
from PIL import Image
from skimage.measure import label, regionprops
from PIL import Image, ImageOps

def padding(img, expected_size):
    desired_size = expected_size
    delta_width = desired_size - img.size[0]
    delta_height = desired_size - img.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
    return ImageOps.expand(img, padding)


def resize_with_padding(img, expected_size):
    img.thumbnail((expected_size[0], expected_size[1]))
    # print(img.size)
    delta_width = expected_size[0] - img.size[0]
    delta_height = expected_size[1] - img.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
    return ImageOps.expand(img, padding)

def Center(img):
    label_img = label(img)
    regions = regionprops(label_img)
    array=[]

    for props in regions:
        array.append( props.area )
        maxarea = max (array)
        for props in regions:    
            y0, x0 = props.centroid
            area=props.area
            if  area == maxarea:
                xx0=x0
                yy0=y0
#    fig, ax = plt.subplots()
#    ax.imshow(mapp, cmap=plt.cm.gray)
#    ax.plot(xx0, yy0, '.g', markersize=10)

    return maxarea,xx0,yy0

address_map = "C:/Users/zsn/Desktop/fold1/Map/"
address_img = "C:/Users/zsn/Desktop/fold1/IMG/"
address_gt="C:/Users/zsn/Desktop/fold1/Mask/"

SavedFolder_imgCrop = "C:/Users/zsn/Desktop/fold1/IMG_Crop/"
SavedFolder_GTCrop = "C:/Users/zsn/Desktop/fold1/Mask_Crop/"
SavedFolder_GTMap="C:/Users/zsn/Desktop/fold1/Mask_Map/"

for filename in sorted(glob.glob( address_img + "*.bmp")):
    WholeName = filename.split("\\")[-1] 
    img= plt.imread(filename) 
    print(WholeName)
    mapp=plt.imread(address_map+WholeName[:-4]+"_Map.bmp") 
    Gt=cv2.imread(address_gt+WholeName,0) 
    if np.max(mapp)>0:
        maxarea,xx0,yy0=Center(mapp)
        a=abs(int(yy0-128))
        b=abs(int(yy0+128))
        c=abs(int(xx0-128))
        d=abs(int(xx0+128))
        e , f = img.shape
                
        nn = abs(e-128)
        ll = abs(f-128)
        if (round(xx0-128) < 0)  :
            img = Image.open(filename)
            img=resize_with_padding(img, (f+ll , f+ll ))
            Gt=Image.open(address_gt+WholeName,0) 
            Gt=resize_with_padding(Gt, (f+ll , f+ll ))
            mapp=Image.open(address_map+WholeName[:-4]+"_Map.bmp") 
            mapp = resize_with_padding(mapp, (f+ll , f+ll ))
            maxarea,xx0,yy0=Center(mapp)
            ee = round(abs(xx0-128))
            rr = round(abs(yy0-128))
            yy0 = round(yy0)
            xx0 = round (xx0)
            imgg= img [rr:(yy0+128),ee:(xx0+128)] 
            GT= Gt[rr:(yy0+128),ee:(xx+128)]
            mapp= mapp [rr:(yy0+128),ee:(xx0+128)]
            
        if (round(yy0-128)< 0)  :
            img = Image.open(filename)
            img= resize_with_padding (img, (e+nn , e+nn ))
            Gt=Image.open(address_gt+WholeName,0) 
            GT=resize_with_padding(GT, (e+nn , e+nn ))
            mapp=Image.open(address_map+WholeName[:-4]+"_Map.bmp") 
            mapp = resize_with_padding(mapp, (e+nn, e+nn ))
            maxarea,xx0,yy0=Center(mapp)
            ee = round(abs(xx0-128))
            rr = round(abs(yy0-128))
            imgg= img [rr:(yy0+128),ee:(xx0+128)]
            mapp= mapp [rr:(yy0+128),ee:(xx0+128)]   
            GT= Gt [rr:(yy0+128),ee:(xx0+128)]
        if (abs(b-a) ==256 and abs(d-c) == 256 ):
            print ("daaali")
            imgg=img[a:b,c:d]    
            GT=Gt[a:b,c:d]  
            mapp= mapp[a:b,c:d] 
        if (abs(b-a) !=256 or abs(d-c) != 256 ):
            if abs(b-a) !=256:
                print ("dali")
                z=256-(b-a)
                GT=Gt[a:b+z,c:d] 
                imgg=img[a:b+z,c:d] 
                mapp= mapp[a:b+z,c:d] 
            if abs(d-c) !=256:
                print ("daali")
                z=abs(256-(d-c)) 
                imgg=img[a:b,c:d+z]     
                GT=Gt[a:b,c:d+z]  
                mapp= mapp[a:b,c:d+z]  
        #imgg=img[abs(int(yy0-128)):abs(int(yy0+128)),abs(int(xx0-128)):abs(int(xx0+128))]   
       # GT=Gt[abs(int(yy0-128)):abs(int(yy0+128)),abs(int(xx0-128)):abs(int(xx0+128))]    
        if (sum(sum(GT/255)) - sum(sum(Gt/255))) >= 0 :
            cv2.imwrite(SavedFolder_imgCrop+WholeName,imgg)  
            cv2.imwrite(SavedFolder_GTCrop+WholeName,GT) 
            cv2.imwrite(SavedFolder_GTMap+WholeName,mapp) 
    



