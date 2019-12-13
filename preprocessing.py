import numpy as np
import os
import cv2
import random
import h5py

def read_images(self, folder):
    imgs=[]
    for item in os.walk(folder):
        files=item[2]
    for file in files:
        img=cv2.imread(folder+'/'+file)
        if(img.shape==(384,512,3)):
            imgs.append(img)
    return imgs

def create_dataset(self):
    # read images into array
    brochure=read_images('pics/paper/brochure')
    cardboard=read_images('pics/paper/cardboard')
    receipt=read_images('pics/paper receipt')
    writing=read_images('pics/paper/writing a5')
    plastic=read_images('pics/plastic/bottle')
    plastic.append(read_images('pics/plastic/food container'))
    plastic.append(read_images('pics/plastic/plastic packaging'))
    can=read_images('pics/cans')
    
    paper=[]
    #taking random 400 paper images, to have a balanced dataset
    for typ in [brochure,writing,receipt,cardboard]:
        paper.extend(random.sample(typ,100))
    
    dataset=np.zeros(shape=(len(paper)+len(plastic)+len(can),384,512,3))
    labelset=np.zeros(shape=(dataset.shape[0]))
    
    h5f = h5py.File('data.h5', 'w')
    h5f.create_dataset('data', data=np.array(dataset))
    h5f.create_dataset('label', data=np.array(labelset))
    h5f.close()


