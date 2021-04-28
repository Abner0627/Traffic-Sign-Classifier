import numpy as np
import os
import pickle
import func
import config
import model
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.optim as optim
import torch.nn as nn

#%% Load
dpath = './traffic-signs-data'
traF = 'train.p'
valF = 'valid.p'
tesF = 'test.p'

with open(os.path.join(dpath, traF), 'rb') as f:
    traD = pickle.load(f)
with open(os.path.join(dpath, valF), 'rb') as f:
    valD = pickle.load(f)
with open(os.path.join(dpath, tesF), 'rb') as f:
    tesD = pickle.load(f)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

#%% Parameters
bz = config.batch
model = model.ResNet18()
optim_m = optim.Adam(model.parameters(), lr=config.lr)
loss_func = nn.CrossEntropyLoss()

model.to(device)
loss_func.to(device)

#%% Pack
tra_data = torch.from_numpy(func._norm(traD['features'].transpose(0,3,1,2))).type(torch.FloatTensor)
tes_data = torch.from_numpy(func._norm(tesD['features'].transpose(0,3,1,2))).type(torch.FloatTensor)
val_data = torch.from_numpy(func._norm(valD['features'].transpose(0,3,1,2))).type(torch.FloatTensor)

tra_label = torch.from_numpy(traD['labels']).type(torch.FloatTensor)
tes_label = torch.from_numpy(tesD['labels']).type(torch.FloatTensor)
val_label = torch.from_numpy(valD['labels']).type(torch.FloatTensor)

tra_dataset = torch.utils.data.TensorDataset(tra_data, tra_label)
tes_dataset = torch.utils.data.TensorDataset(tes_data, tes_label)
val_dataset = torch.utils.data.TensorDataset(val_data, val_label)

tra_dataloader = torch.utils.data.DataLoader(dataset = tra_dataset, batch_size=bz, shuffle=True)
tes_dataloader = torch.utils.data.DataLoader(dataset = tes_dataset, batch_size=bz, shuffle=False)
val_dataloader = torch.utils.data.DataLoader(dataset = val_dataset, batch_size=bz, shuffle=False)

#%% Training
for epoch in range(config.Epoch):
    model.train()
    for ntra, (Data, Label) in enumerate(tra_dataloader):
        optim_m.zero_grad()
        # print(Data.size())
        # plt.imshow(Data.data.numpy()[0,0,:,:])
        # plt.show()
        gray = torchvision.transforms.functional.rgb_to_grayscale(Data)
        data_rgb = Data.to(device)
        data_gray = gray.to(device)
        val = Label.type(torch.long).to(device)
        pred = model(data_rgb, data_gray)
        
        loss = loss_func(pred, val)
        loss.backward()
        optim_m.step()
    
    model.eval()
    with torch.no_grad():
        for nval, (Data_V, Label_V) in enumerate(val_dataloader):
            gray_V = torchvision.transforms.functional.rgb_to_grayscale(Data_V)
            data_rgb = Data_V.to(device)
            data_gray = gray_V.to(device)
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

        print('epoch[{}], loss:{:.4f}, val_acc:{:.4f}'
                .format(epoch+1, loss.item(), acc))    