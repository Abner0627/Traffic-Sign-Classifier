import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cv2
import func
import config

import torch
import torch.optim as optim
import torch.nn as nn
import tensorflow.keras as keras
import tensorflow as tf

def _config():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-T", default=False, help="tensorflow")
    parser.add_argument("-P", default=False, help="pytorch")
    return parser.parse_args()

#%%
P = './test_img'
args = _config()
sP = './test_img/result'
data_list = os.listdir(P)
tot = []
for i in range(5):
    imgs = (cv2.imread(os.path.join(P, data_list[0])))[np.newaxis,...]
    tot.append(imgs)

img = np.vstack(np.array(tot))
img_RGB = img.transpose(0,3,1,2)
img_Gray = func._gray(img_RGB)

tesD = np.concatenate((img_RGB, img_Gray), axis=1)

#%% TF
if args.T:
    tesD = np.reshape(tesD, (-1,32,32,4))
    tesD = tf.image.convert_image_dtype(tesD, dtype=tf.float16, saturate=False)
    model = keras.models.load_model('model_tf')
    pred = model.predict(tesD)
    print(pred.shape)
