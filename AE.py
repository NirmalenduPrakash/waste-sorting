import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D,BatchNormalization,Activation,Conv2DTranspose,Input,Dense,ZeroPadding2D
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model
from sklearn.metrics import mean_squared_error
from tensorflow.keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,LearningRateScheduler,History
import util
#from google.colab import drive
#drive.mount('/content/drive')

#function for displaying images:


def load_data():
    data_file=h5py.File('data.h5')
    
    img = np.array(data_file["data"])
    labels = np.array(data_file["label"])
    
    img = img/255.
    num_classes=len(np.unique(labels))
    noise_factor = 0.4
    img_noisy = img + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=img.shape)
    return img_noisy,img

def simpleConv2D(inp,filters,size=3,strides=1,padding='valid',layer_name=None):
  x = Conv2D(filters,kernel_size=size,padding=padding,strides=(strides,strides),activation='relu',name=layer_name)(inp)
  return x

def simpleDeConv(inp,filters,size=3,strides=1,padding='valid'):
  x = Conv2DTranspose(filters,kernel_size=size,strides=(strides,strides),padding=padding,activation='relu')(inp)
  return x

def create_autoencoder(input_shape):
  Inp=Input(shape=input_shape)
  x=simpleConv2D(Inp,6,strides=2)
  x=simpleConv2D(x,12,strides=2)
  x=simpleConv2D(x,24,strides=2) 
  x=simpleConv2D(x,48,strides=2)
  x=simpleConv2D(x,96,strides=2)
  x=simpleConv2D(x,192,strides=2)
  #x=simpleConv2D(x,384,strides=2,layer_name='encoder')
  encoder=Model(Inp,x)
  #x=simpleDeConv(x,192,strides=2)
  x=simpleDeConv(x,96,strides=2)
  x=simpleDeConv(x,48,strides=2)
  x=simpleDeConv(x,24,strides=2)
  x=simpleDeConv(x,12,strides=2)
  x=simpleDeConv(x,6,strides=2)
  x=simpleDeConv(x,3,strides=2)
  x=ZeroPadding2D(padding=((1,0),(1,0)))(x)
  decoder = Model(Inp,x) 
  #x=simpleConv2D(x,192,strides=2)
  #x=simpleConv2D(x,384,strides=2)  
  return encoder,decoder

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

def create_model():
    encoder,decoder= create_autoencoder((384,512,3))
    decoder.summary()
    decoder.compile(optimizer='rmsprop',loss='mse')
    return decoder

def train():
    img_noisy,img=load_data() 
    print(f"size of the train dataset {img.shape}")
    decoder=create_model()
    checkpoint = ModelCheckpoint(filepath='model_weights.h5',
                                 monitor='loss',
                                 verbose=1,save_best_only=True)
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                   cooldown=0,
                                   patience=5,
                                   min_lr=0.5e-6)
    lr_scheduler = LearningRateScheduler(lr_schedule)
    
    decoder.fit(img_noisy,img,batch_size=32,epochs=50,shuffle=True,callbacks=[checkpoint,lr_reducer,lr_scheduler])

def load_weights(model):
    model.load_weights('model_weights.h5')
    return model

def test(imgs,model):
    pred=model.predict(imgs)
    for i in range(len(imgs)):
        util.imgplt(imgs[i],'Original')
        util.imgplt(pred[i],'Reconstructed')
        print('mse error: ',np.sum(np.square(imgs[i]-pred[i])))
        
def classify(model,img):
    pred=model.predict(img)
    return np.sum(np.square(img-pred))<35000        