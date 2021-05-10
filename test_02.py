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
F = './traffic-signs-data/signnames.csv'
M = './model'
args = _config()
sP = './test_img/result'
sP2 = './img'
data_list = os.listdir(P)
tot = []
data_list = data_list[:-1]
# print(data_list)
for i in range(len(data_list)):
    imgs = (cv2.imread(os.path.join(P, data_list[i])))[np.newaxis,...]
    print(data_list[i])
    tot.append(imgs)


img = np.vstack(np.array(tot))
img_RGB = img.transpose(0,3,1,2)
img_Gray = func._gray(img_RGB)

tesD = func._norm(np.concatenate((img_RGB, img_Gray), axis=1))
# print(tesD.shape)
#%% TF
if args.T:
    import tensorflow.keras as keras
    import tensorflow as tf
    tesD = tesD.transpose(0,2,3,1)
    tesD = tf.image.convert_image_dtype(tesD, dtype=tf.float16, saturate=False)
    model = keras.models.load_model(os.path.join(M, 'model_tf'))
    pro_model = keras.Sequential([model, keras.layers.Softmax()])
    pred = pro_model.predict(tesD)
    np.save(os.path.join(sP, 'Pred_tf.npy'), pred)
    # print(pred.shape)

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
    # print(pred.shape)

else:
    pt = np.load(os.path.join(sP, 'Pred_pt.npy'))
    tf = np.load(os.path.join(sP, 'Pred_tf.npy'))
    realN = [13, 15, 25, 28, 32]
    predN_pt = np.argmax(pt, axis=1)
    predN_tf = np.argmax(tf, axis=1)
    # print(predN_pt)
    # print(predN_tf)
    barx = np.arange(43)
    name = np.array(pd.read_csv(os.path.join(F), header=None))[1:, :]

    fig, ax = plt.subplots(5,2, figsize=(20,20))
    for i in range(5):
        real = name[realN[i], 1]
        pred = name[predN_pt[i], 1]
        c = 'royalblue' if real==pred else 'firebrick'
        ax[i,0].imshow(cv2.cvtColor(img[i,...], cv2.COLOR_BGR2RGB))
        ax[i,0].set_title(real, size=30)
        ax[i,1].bar(barx, pt[i,...], color=c)
        ax[i,1].set_xticks(barx)
        ax[i,1].set_xlim(-1, 43)
        ax[i,1].set_xticklabels(barx)
        ax[i,1].set_title(pred, size=30)
    plt.tight_layout()
    plt.savefig(os.path.join(sP2, 'pt_result.png'))

    fig, ax = plt.subplots(5,2, figsize=(20,20))
    for i in range(5):
        real = name[realN[i], 1]
        pred = name[predN_tf[i], 1]
        c = 'royalblue' if real==pred else 'firebrick'
        ax[i,0].imshow(cv2.cvtColor(img[i,...], cv2.COLOR_BGR2RGB))
        ax[i,0].set_title(real, size=30)
        ax[i,1].bar(barx, tf[i,...], color=c)
        ax[i,1].set_xticks(barx)
        ax[i,1].set_xlim(-1, 43)
        ax[i,1].set_xticklabels(barx)
        ax[i,1].set_title(pred, size=30)
    plt.tight_layout()
    plt.savefig(os.path.join(sP2, 'tf_result.png'))    
