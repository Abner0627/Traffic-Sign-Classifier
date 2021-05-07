import config
import os
os.environ['PYTHONHASHSEED']=str(config.seed)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import random

def reset_random_seeds():
   os.environ['PYTHONHASHSEED']=str(config.seed)
   tf.random.set_seed(config.seed)
   np.random.seed(config.seed)
   random.seed(config.seed)

reset_random_seeds()

class resnet_block(keras.Model):
    def __init__(self, outdim, cv_1x1=False):
        super(resnet_block, self).__init__()
        self.F = keras.Sequential([
            keras.layers.Conv2D(outdim, kernel_size=(3,3), padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            keras.layers.Conv2D(outdim, kernel_size=(3,3), padding='same'),
            keras.layers.BatchNormalization()
        ])
        if cv_1x1:
            self.cv1 = keras.layers.Conv2D(outdim, kernel_size=(1,1))
        else:
            self.cv1 = None
    def call(self, x):
        y1 = self.F(x)
        if self.cv1:
            y2 = self.cv1(x)
        else:
            y2 = x
        out = keras.activations.relu(y1+y2)  
        return out

class ResNet18(keras.Model):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.cvrgb = keras.Sequential([
            keras.layers.Conv2D(32, kernel_size=(7,7), strides=2, padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPool2D(2)
        ])
        self.cvgray = keras.Sequential([
            keras.layers.Conv2D(32, kernel_size=(7,7), strides=2, padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPool2D(2)
        ]) 
        self.bk1 = resnet_block(64)
        self.bk1_2 = resnet_block(64)
        self.bk2 = resnet_block(128, cv_1x1=True)
        self.bk3 = resnet_block(128)
        self.bk3_2 = resnet_block(128)    
        self.pool = keras.layers.GlobalMaxPool2D()  

        self.FC = keras.Sequential([
            keras.layers.Flatten(),
            keras.layers.Dense(64),
            keras.layers.ReLU(),
            keras.layers.Dense(43)
        ])     

    def call(self, img):
        rgb = img[:,:,:,:3]
        gray = (img[:,:,:,-1])[:,:,:,np.newaxis]
        y1_rgb = self.cvrgb(rgb)
        y1_gray = self.cvgray(gray)
        y1 = tf.concat((y1_rgb, y1_gray), -1)
        y2 = self.bk1_2(self.bk1(y1))
        y3 = self.bk2(y2)
        y4 = self.bk3_2(self.bk3(y3))
        y5 = self.pool(y4)
        pred = self.FC(y5)
        return pred          

#%% Test
if __name__ == "__main__":
    x = np.random.rand(32,32,32,4)
    F = ResNet18()
    y = F(x)
    print(y.shape)
