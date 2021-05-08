import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import random

#%% Args
def _config():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-P", "--plot", default='0', help="plot")
    return parser.parse_args()


def _norm(x):
    mu = np.mean(x)
    std = np.std(x)
    nor = (x-mu)/std
    return nor

#%% Load
dpath = './traffic-signs-data'
traF = 'train.p'
valF = 'valid.p'
tesF = 'test.p'
sP = './img'
M = './model'
args = _config()

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

if args.plot=='0':
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
    plt.show()
    # plt.savefig(os.path.join(sP, sF))

elif args.plot=='1':
    sF = 'img_' + str(args.plot) + '.png'
    num_label = []
    for i in range(43):
        idx = list(np.array(np.where(tra_label==i)).squeeze())
        num_label.append(len(idx))    
    num_label = np.array(num_label)
    x = [0, 7, 14, 21, 28, 35, 42]
    fig, ax = plt.subplots(1, 1, figsize=(8,5))
    ax.bar(np.arange(43), num_label)
    ax.set_xticks(x)
    ax.set_xticklabels(x)
    plt.title('Training Data')
    plt.tight_layout()
    plt.savefig(os.path.join(sP, sF))    
    # plt.show()

elif args.plot=='2':
    sF = 'img_' + str(args.plot) + '.png'
    img0 = traRGB[1019,:,:,:][np.newaxis,:,:,:].transpose(0,3,1,2)
    img1 = traRGB[520,:,:,:][np.newaxis,:,:,:].transpose(0,3,1,2)
    img2 = traRGB[627,:,:,:][np.newaxis,:,:,:].transpose(0,3,1,2)

    img0 = np.reshape(img0, (32*32*3))
    img1 = np.reshape(img1, (32*32*3))
    img2 = np.reshape(img2, (32*32*3))

    bins = np.linspace(0, 255, 256)

    sl0, _ = np.histogram(img0, bins=bins)
    sl1, _ = np.histogram(img1, bins=bins)
    sl2, _ = np.histogram(img2, bins=bins)

    fig, ax = plt.subplots(1, 1, figsize=(8,5))
    x = np.arange(255)
    ax.bar(x, sl0, alpha=0.7, label='Sample 0')
    ax.bar(x, sl1, alpha=0.7, label='Sample 1')
    ax.bar(x, sl2, alpha=0.7, label='Sample 2')
    ax.set_xlim(0, 180)
    plt.legend(loc=1)
    plt.tight_layout()
    plt.savefig(os.path.join(sP, sF))    
    # plt.show() 

elif args.plot=='3':
    sF = 'img_' + str(args.plot) + '.png'
    img0 = traRGB[1019,:,:,:][np.newaxis,:,:,:].transpose(0,3,1,2)
    img1 = traRGB[520,:,:,:][np.newaxis,:,:,:].transpose(0,3,1,2)
    img2 = traRGB[627,:,:,:][np.newaxis,:,:,:].transpose(0,3,1,2)

    img0 = _norm(np.reshape(img0, (32*32*3)))
    img1 = _norm(np.reshape(img1, (32*32*3)))
    img2 = _norm(np.reshape(img2, (32*32*3)))

    bins = np.linspace(0, 1, 256)

    sl0, _ = np.histogram(img0, bins=bins)
    sl1, _ = np.histogram(img1, bins=bins)
    sl2, _ = np.histogram(img2, bins=bins)

    fig, ax = plt.subplots(1, 1, figsize=(8,5))
    x = np.linspace(0, 1, 255)
    ax.bar(x, sl0, alpha=0.7, label='Sample 0', width=0.05)
    ax.bar(x, sl1, alpha=0.7, label='Sample 1', width=0.05)
    ax.bar(x, sl2, alpha=0.7, label='Sample 2', width=0.05)
    ax.set_xlim(-1, 2)
    plt.legend(loc=1)
    plt.tight_layout()
    plt.savefig(os.path.join(sP, sF))    
    # plt.show()   

elif args.plot=='4':
    sF = 'img_' + str(args.plot) + '.png'
    num_label = []
    for i in range(43):
        idx = list(np.array(np.where(val_label==i)).squeeze())
        num_label.append(len(idx))    
    num_label = np.array(num_label)
    x = [0, 7, 14, 21, 28, 35, 42]
    fig, ax = plt.subplots(1, 1, figsize=(8,5))
    ax.bar(np.arange(43), num_label)
    ax.set_xticks(x)
    ax.set_xticklabels(x)
    plt.title('Validation Data')
    plt.tight_layout()
    plt.savefig(os.path.join(sP, sF))    
    # plt.show() 

elif args.plot=='5':
    sF = 'img_' + str(args.plot) + '.png'
    with open(os.path.join(M, 'loss_acc_pt.npy'), 'rb') as f:          
        L = np.load(f)
        A = np.load(f)
    fig, ax = plt.subplots(1, 2, figsize=(10,5))
    fig.suptitle("model_pt", fontsize="x-large")
    ax[0].plot(L, color='dodgerblue')
    ax[0].set_title('Loss')
    ax[0].set_xlabel('Epoch')
    ax[1].plot(A, color='darkorange')
    ax[1].set_title('Accuracy')
    ax[1].set_xlabel('Epoch')
    plt.tight_layout()
    plt.savefig(os.path.join(sP, sF))    
    # plt.show() 

elif args.plot=='6':
    sF = 'img_' + str(args.plot) + '.png'
    with open(os.path.join(M, 'loss_acc_tf.npy'), 'rb') as f:          
        L = np.load(f)
        A = np.load(f)
    fig, ax = plt.subplots(1, 2, figsize=(10,5))
    fig.suptitle("model_tf", fontsize="x-large")
    ax[0].plot(L, color='dodgerblue')
    ax[0].set_title('Loss')
    ax[0].set_xlabel('Epoch')
    ax[1].plot(A, color='darkorange')
    ax[1].set_title('Accuracy')
    ax[1].set_xlabel('Epoch')
    plt.tight_layout()
    plt.savefig(os.path.join(sP, sF))    
    # plt.show()     


