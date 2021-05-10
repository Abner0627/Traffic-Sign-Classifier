import numpy as np
import os
import pickle
import func
import config
import model_pt
import random

import torch
import torch.optim as optim
import torch.nn as nn

#%% Load
dpath = './traffic-signs-data'
traF = 'train.p'
valF = 'valid.p'
M = './model'

with open(os.path.join(dpath, traF), 'rb') as f:
    traD = pickle.load(f)
with open(os.path.join(dpath, valF), 'rb') as f:
    valD = pickle.load(f)

# Img
traRGB = traD['features'].transpose(0,3,1,2)
valRGB = valD['features'].transpose(0,3,1,2)

traGray = func._gray(traRGB)
valGray = func._gray(valRGB)

traImg = np.concatenate((traRGB, traGray), axis=1)
valImg = np.concatenate((valRGB, valGray), axis=1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(config.seed)

#%% Parameters
bz = config.batch
model = model_pt.ResNet18()
optim_m = optim.Adam(model.parameters(), lr=config.lr, amsgrad=config.amsgrad)
loss_func = nn.CrossEntropyLoss()

model = model.to(device)
loss_func = loss_func.to(device)

#%% Pack
tra_data = torch.from_numpy(func._norm(traImg)).type(torch.FloatTensor)
val_data = torch.from_numpy(func._norm(valImg)).type(torch.FloatTensor)

tra_label = torch.from_numpy(traD['labels']).type(torch.FloatTensor)
val_label = torch.from_numpy(valD['labels']).type(torch.FloatTensor)

tra_dataset = torch.utils.data.TensorDataset(tra_data, tra_label)
val_dataset = torch.utils.data.TensorDataset(val_data, val_label)

tra_dataloader = torch.utils.data.DataLoader(dataset = tra_dataset, batch_size=bz, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(dataset = val_dataset, batch_size=bz, shuffle=False)

#%% Training
L, A = [], []
for epoch in range(config.Epoch):
    model.train()
    for ntra, (Data, Label) in enumerate(tra_dataloader):
        optim_m.zero_grad()
        data_rgb = Data[:,:3,:,:].to(device)
        data_gray = Data[:,-1,:,:].unsqueeze(1)
        data_gray = data_gray.to(device)
        val = Label.type(torch.long).to(device)
        pred = model(data_rgb, data_gray)
        
        loss = loss_func(pred, val)
        loss.backward()
        optim_m.step()
    
    model.eval()

    with torch.no_grad():
        for nval, (Data_V, Label_V) in enumerate(val_dataloader):
            data_rgb = Data_V[:,:3,:,:].to(device)
            data_gray = Data_V[:,-1,:,:].unsqueeze(1)
            data_gray = data_gray.to(device)
            pred = model(data_rgb, data_gray)

            out = pred.cpu().data.numpy()
            pr  = np.argmax(out, axis=1)
            if nval==0:
                prd = pr
            else:
                prd = np.concatenate((prd, pr))

        va = valD['labels']
        hd = np.sum(prd==va)
        acc = (hd/va.shape[0])
        loss_np = loss.item()
        print('epoch[{}] >> loss:{:.4f}, val_acc:{:.4f}'
                .format(epoch+1, loss_np, acc)) 

        L.append(loss_np)
        A.append(acc)

with open(os.path.join(M, 'loss_acc_pt.npy'), 'wb') as f:          
    np.save(f, np.array(L))
    np.save(f, np.array(A))

torch.save(model, os.path.join(M, 'model_pt.pth'))         
