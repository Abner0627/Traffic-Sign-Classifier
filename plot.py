import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import random

#%% Args
def config():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", "-P", default=0, help="input the consumption data path")
    return parser.parse_args()
args = config()

#%% Load
dpath = './traffic-signs-data'
traF = 'train.p'
valF = 'valid.p'
tesF = 'test.p'
sP = './img'

with open(os.path.join(dpath, traF), 'rb') as f:
    traD = pickle.load(f)
with open(os.path.join(dpath, valF), 'rb') as f:
    valD = pickle.load(f)
with open(os.path.join(dpath, tesF), 'rb') as f:
    tesD = pickle.load(f)

# Img
traRGB = traD['features']
valRGB = valD['features']
tesRGB = tesD['features']

tra_label = traD['labels']
val_label = valD['labels']
tes_label = tesD['labels']

if args.plot==0:
    sF = 'img_' + str(args.plot) + '.png'
    ns = 40
    num_label = []
    total = []
    for i in range(43):
        idx = list(np.array(np.where(tra_label==i)).squeeze())
        for j in range(ns):
            A = random.sample(idx, ns)
            img = traRGB[A[j],:,:,:]
            if j==0:
                out = img
            else:
                out = np.hstack((out, img))
        total.append(out)

    fig, ax = plt.subplots(1, 1, figsize=(15,15))
    ax.imshow(np.vstack(total))
    plt.tight_layout()
    plt.axis('off')
    plt.savefig(os.path.join(sP, sF))

elif args.plot==1:
    sF = 'img_' + str(args.plot) + '.png'
    num_label = []
    for i in range(43):
        idx = list(np.array(np.where(tra_label==i)).squeeze())
        num_label.append(len(idx))    
    num_label = np.array(num_label)

    fig, ax = plt.subplots(1, 1, figsize=(15,15))
    ax.bar(num_label)
    plt.tight_layout()
    plt.savefig(os.path.join(sP, sF))    
    plt.show()


