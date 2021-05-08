import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cv2
import func
import config

def _config():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-T", default=False, help="tensorflow")
    parser.add_argument("-P", default=False, help="pytorch")
    return parser.parse_args()

#%%
P = './test_img'
M = './model'
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
    import tensorflow.keras as keras
    import tensorflow as tf
    tesD = np.reshape(tesD, (-1,32,32,4))
    tesD = tf.image.convert_image_dtype(tesD, dtype=tf.float16, saturate=False)
    model = keras.models.load_model(os.path.join(M, 'model_tf'))
    pro_model = keras.Sequential([model, keras.layers.Softmax()])
    pred = pro_model.predict(tesD)
    np.save(os.path.join(sP, 'Pred_tf.npy'), pred)
    print(pred.shape)

elif args.P:
    import torch
    model = torch.load(os.path.join(M, 'model_pt.pth'))
    model.eval()
    model.cpu()
    tesD = torch.from_numpy(tesD).type(torch.FloatTensor)
    RGB, GRAY = tesD[:,:3,:,:], tesD[:,-1,:,:].unsqueeze(1)
    pred = torch.nn.functional.softmax(model(RGB, GRAY), dim=-1)
    pred = pred.data.numpy()
    np.save(os.path.join(sP, 'Pred_pt.npy'), pred)
    print(pred.shape)

else:
    import matplotlib.pyplot as plt
    pt = np.load(os.path.join(sP, 'Pred_pt.npy'))
    pt = np.load(os.path.join(sP, 'Pred_pt.npy'))