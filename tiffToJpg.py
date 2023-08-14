#!/usr/bin/env python
# coding: utf-8




import os
import gc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tifffile as tifi
import PIL
import cv2
from tqdm import tqdm, trange

import tensorflow as tf 
from tensorflow import keras 
from keras.utils import to_categorical
from keras import layers

def getridofEmptySpaces(img):
    
    print('Original image shape : ', img.shape)
    print('Removing rows ..................')

#     Code to identify empty rows.
    idx = []
    for i in trange(img.shape[0]):
        if len(np.unique(img[i,:])) <= 100:
            idx.append(i)
            
    print('Rows removed : ', len(idx))
    
    img = np.delete(img, idx, axis=0)
    
    print('New image shape : ', img.shape)
    
    print('Removing columns ..................')
    idxy = []
    for i in trange(img.shape[1]):
        
        if len(np.unique(img[:,i])) <= 100:
            idxy.append(i)
            
    print('Columns removed : ', len(idxy))
            
    img = np.delete(img, idxy, axis=1)
    print('New image shape : ', img.shape)
    
            
    del idx, idxy
    gc.collect()
    
    return img

#Get Train Data
df_train = pd.read_csv('/home/bear/prakash/aai541/finalproject/data/train.csv')
print("Train size:", len(df_train), "Train Unique Patient Samples:", len(df_train.patient_id.unique()))


# For each Tiff file, save it as jpg file with 256 x 256 dimenision
for x in range(int(df_train.size)):
    img_path = "./data/train/"+df_train.image_id[x]+".tif"
    img_path2 = "./data/train_jpg2/"+df_train.image_id[x]+".jpg"
    if(os.path.isfile(img_path2)):
        print(img_path2, " exists")
    else:
        img_tmp = tifi.imread(img_path)
        img_tmp = getridofEmptySpaces(img_tmp)
        img_tmp = cv2.resize(img_tmp, (256, 256))
        img_path2 = "./data/train_jpg2/"+df_train.image_id[x]+".jpg"
        cv2.imwrite(img_path2, img_tmp)
        print("Created ",img_path2)
        del img_tmp  # to free memory
        gc.collect() # to free memory
        # print(gc.collect())
print("Completed Tiff to PDF Conversion")

