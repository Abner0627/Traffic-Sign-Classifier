import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
P2 = './traffic-signs-data'
M = './model'
args = _config()
sP = './test_img/result'
data_list = os.listdir(P)
tot = []
data_list = data_list[:-1]
print(data_list)
for i in range(len(data_list)):
    imgs = (cv2.imread(os.path.join(P, data_list[i])))[np.newaxis,...]
    tot.append(imgs)

img = np.vstack(np.array(tot))
img_RGB = img.transpose(0,3,1,2)
img_Gray = func._gray(img_RGB)

tesD = np.concatenate((img_RGB, img_Gray), axis=1)

#%% TF
if args.T:
    import tensorflow.keras as keras
    import tensorflow as tf
    tesD = tesD.transpose(0,2,3,1)
    print(tesD.shape)
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
    pred = model(RGB, GRAY)
    pred = torch.nn.functional.softmax((pred), dim=-1)
    pred = pred.data.numpy()
    np.save(os.path.join(sP, 'Pred_pt.npy'), pred)
    print(pred.shape)

else:
    pt = np.load(os.path.join(sP, 'Pred_pt.npy'))
    tf = np.load(os.path.join(sP, 'Pred_tf.npy'))
    print(img.shape)
    print(np.argmax(pt, axis=1))
    print(np.argmax(tf, axis=1))
    sign = np.array(pd.read_csv(os.path.join(P2, 'signnames.csv'), header=None))[1:,:]
    barx = np.arange(43)
    fig, ax = plt.subplots(5,2, figsize=(25,10))
    pred_idx = np.argmax(pt[0,...])
    pred_name = sign[pred_idx, 1]
    real_idx = real_idx_L[0]
    real_name = sign[real_idx, 1]

    ax[0,0].imshow(cv2.cvtColor(img[0,...], cv2.COLOR_BGR2RGB))

    ax[0,1].bar(barx, pt[0,...])
    # plt.show()
