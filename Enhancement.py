import glob
import cv2
import os
import numpy as np

def reversal(input_path,output_path):
    input_files = os.listdir(input_path)
    for file in input_files:
        file_path=os.path.join(input_path,file)
        save_path=os.path.join(output_path,file)
        img=cv2.imread(file_path)
        reversal=255-img
        cv2.imwrite(save_path,reversal)

        del img
        del reversal


def CLAHE(input_path,output_path):
    input_files = os.listdir(input_path)
    for file in input_files:
        imgName = filename.split("\\")[-1] 
        img = plt.imread(input_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Contrast Limited Adaptive Histogram Equalization
        # cliplimit control the contrast of the result, the bigger limit, the higher contrast
        # cliplimit=2.0 is suitable for our CXR task
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahed_img=clahe.apply(img)
        cv2.imwrite(output_path+imgName, clahed_img)
        del img
        del gray
        del clahe

def histogram_equalization(input_path,output_path):
    input_files = os.listdir(input_path)
    for file in input_files:
        file_path = os.path.join(input_path, file)
        save_path = os.path.join(output_path, file)
        img = cv2.imread(file_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        equ=cv2.equalizeHist(gray)
        cv2.imwrite(save_path, equ)
        del img
        del gray
        del que

input_path='C:/Users/zsn/Desktop/Figshare/Data/Fold4_is_Test/IMG/'
output_path='C:/Users/zsn/Desktop/Figshare/Data/Fold4_is_Test/IMG/Enhanced/'

for filename in sorted(glob.glob(input_path+"*.bmp")):
    print(filename)
    imgName = filename.split("\\")[-1] 
    img=plt.imread(filename,0)
    med_img = cv2.medianBlur(img, 5) 
# create a CLAHE object (Arguments are optional)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    out = clahe.apply(med_img)
    cv2.imwrite(output_path+imgName, out)
    
# Display the images side by side using cv2.hconcat
out1 = cv2.hconcat([img,out])
plt.imshow (out1)
