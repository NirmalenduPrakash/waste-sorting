import numpy as np
import os
import cv2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import MaxPooling2D,AveragePooling2D,UpSampling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import add
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,LearningRateScheduler,History
import h5py
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
import AE
from util import *

def load_data():
    f                   = h5py.File('data.h5','r')
    dataset                   = f['data']
    labelset                   = f['label']
    dataset=np.array(dataset)
    labelset=np.array(labelset)
    f.close()
    return dataset,labelset

def preprocess(dataset,labelset):
    trDat,tsDat,trLbl,tsLbl=train_test_split(dataset,labelset,test_size=0.2,random_state=42)
    trDat       = trDat.astype('float32')/255
    tsDat       = tsDat.astype('float32')/255
    trLbl=to_categorical(trLbl)
    tsLbl=to_categorical(tsLbl)
    return trDat,tsDat,trLbl,tsLbl

def reslyr(inputs,numFilters=16,kernelSize=3,strides=1,activation='relu',batchNorm=True,convFirst=True,lyrName=None):
    convLyr=Conv2D(numFilters,kernelSize,strides=strides,padding='same',kernel_initializer='he_normal',kernel_regularizer='l2',name=lyrName+'_conv' if lyrName else None)
    if convFirst:
        x=convLyr(inputs)
        if batchNorm:
            x=BatchNormalization(name=lyrName+"_bn" if lyrName else None)(x)
        if activation is not None:
            x=Activation(activation,name=lyrName+"_activation" if lyrName else None)(x)
    else:
        if batchNorm:
            x=BatchNormalization(name=lyrName+"_bn" if lyrName else None)(x)
        if activation is not None:
            x=Activation(activation,name=lyrName+"_activation" if lyrName else None)(x)
        x=convLyr(x)
    return x

def resblkv1(inputs,num_filters=16,downsample_on_first=True,num_blocks=3):
      x=inputs
      for run in range(0,num_blocks):
        strides=1
        if(downsample_on_first and run==0):
          strides=2
        y=reslyr(inputs=x,strides=strides,numFilters=num_filters)
        y=reslyr(inputs=y,activation=None,numFilters=num_filters)
        if(downsample_on_first and run==0):
          x=reslyr(inputs=x,strides=strides,batchNorm=False,activation=None,numFilters=num_filters)
        x=add([x,y])
        x=Activation('relu')(x)
      return x
  
def createResnetV1(input_shape,num_classes=3):
      inputs=Input(shape=input_shape)
      v=reslyr(inputs)
      v=resblkv1(v,downsample_on_first=False,num_filters=16,num_blocks=3)
      v=resblkv1(v,downsample_on_first=True,num_filters=32,num_blocks=3)
      v=resblkv1(v,downsample_on_first=True,num_filters=64,num_blocks=3)
      v=AveragePooling2D(pool_size=8)(v)
      v=Flatten()(v)   
      v=Dense(3,kernel_initializer='he_normal',name='class_probabs')(v)
      v=Activation('softmax')(v)
      model=Model(inputs=inputs,outputs=v)      
      return model 

def lr_schedule(epoch):
    lr = 1e-3
    if epoch > 10:
        lr = 1e-4
    if epoch > 20:
        lr = 1e-5
    if epoch > 30:
        lr = 1e-6
    if epoch > 40:
        lr = 1e-7
    print('Learning rate: ', lr)
    return lr

def train(trDat,tsDat,trLbl,tsLbl):
    optmz = optimizers.RMSprop(lr=0.001)
    model=createResnetV1((trDat.shape[1],trDat.shape[2],trDat.shape[3]))
    model.compile(loss='categorical_crossentropy',optimizer=optmz,metrics=['accuracy'])
    model.summary()
    checkpoint = ModelCheckpoint(filepath='classifier_weights.h5',
                             monitor='val_acc',
                             verbose=1,save_best_only=True)
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)
    lr_scheduler = LearningRateScheduler(lr_schedule)
    model.fit(trDat,trLbl,validation_data=(tsDat,tsLbl),epochs=50,batch_size=32,shuffle=True,callbacks=[checkpoint,lr_reducer,lr_scheduler])

def load_weights(model):
    model.load_weights('classifier_weights.h5')
    return model

if(__name__=='__main__'):
    classifier_model=createResnetV1((384,512,3))
    classifier_model.load_weights('classifier_weights.h5')
    ae=AE.create_model()
    ae.load_weights('model_weights.h5')
    for direc in ['test_images/true/','test_images/false/']:
        images=read_images(direc)
        for img in images:
            imgplt(img)
            if(AE.classify(ae,img.reshape(1,384,512,3))):
                pred=classifier_model.predict(img.reshape(1,384,512,3))
                pred=np.argmax(pred[0])
                if(pred==0):
                    print('paper') 
                elif(pred==1):    
                    print('plastic')
                else:
                    print('can')
            else:
                print('non-recyclable')