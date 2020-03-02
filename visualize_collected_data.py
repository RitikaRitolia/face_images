# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 15:24:24 2020

@author: KIIT
"""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle

data_dir=os.path.join(os.getcwd(),'store_arr')
img_dir=os.path.join(os.getcwd(),'images')

with open(os.path.join(data_dir,'images.p'),'rb')as f:
    images = pickle.load(f)
    
with open(os.path.join(data_dir,'labels.p'),'rb')as f:
    labels = pickle.load(f)

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
labels=le.fit_transform(labels)

fig, axes = plt.subplots(nrows=5,ncols=5,figsize=(10,10))
fig.tight_layout()

for i in range(5):
  for j in range(5):
    imgs = images[labels==j]
    image = imgs[np.random.randint(0,len(imgs)-1),:,:]
    axes[j][i].imshow(image)