#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 10:39:21 2019

@author: nirmalenduprakash
"""
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
    
def imgplt(img,title=None):
  plt.figure()
  plt.title(title)
  plt.imshow(img)
  plt.axis('off')
  plt.show()
  plt.close()
  
def read_images(directory,normalize=True):
    imgs=[]
    for item in os.walk(directory):
      files=item[2]  
      for i in files:
          if i.split('.')[-1]=='jpeg' or i.split('.')[-1]=='jpg' or i.split('.')[-1]=='png' or i.split('.')[-1]=='JPG': 
            try:
                im=cv2.imread(directory+i)
                im=cv2.resize(im,(512, 384))
                im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
                imgs.append(im)
            except:
                continue
    imgs = np.array(imgs)
    if(normalize):
        imgs=imgs/255.
    return imgs    