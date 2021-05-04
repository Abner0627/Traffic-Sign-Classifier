import tensorflow as tf
import tensorflow.keras as keras
import numpy as np

class resnet_block(keras.Model):
    def __init__(self, outdim, cv_1x1=False):
        super(resnet_block, self).__init__()
        self.F = keras.Sequential([
            keras.layers.Conv2D(outdim, kernel_size=3, padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            keras.layers.Conv2D(outdim, kernel_size=3, padding='same'),
            keras.layers.BatchNormalization()
        ])
        if cv_1x1:
            self.cv1 = keras.layers.Conv2D(outdim, kernel_size=1)
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
            keras.layers.Conv2D(32, kernel_size=7, strides=2, padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPool2D(2)
        ])
        self.cvgray = keras.Sequential([
            keras.layers.Conv2D(32, kernel_size=7, strides=2, padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPool2D(2)
        ]) 
        self.bk1 = resnet_block(64)
        self.bk2 = resnet_block(128, cv_1x1=True)
        self.bk3 = resnet_block(128)    
        self.pool = keras.layers.GlobalMaxPool2D()  

        self.FC = keras.Sequential([
            keras.layers.Flatten(),
            keras.layers.Dense(64),
            keras.layers.ReLU(),
            keras.layers.Dense(43),
            keras.layers.Softmax()
        ])     

    def call(self, rgb, gray):
        x = self.cvrgb(rgb)
        print(x.shape)
        return rgb          

#%% Test
if __name__ == "__main__":
    x = np.random.rand(32,32,32,3)
    x2 = np.random.rand(32,32,32,1)
    F = ResNet18()
    y = F(x, x2)
    # print(y.size())
