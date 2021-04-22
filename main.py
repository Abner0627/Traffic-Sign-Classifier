import numpy as np
import os
import pickle
import func
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
import torch.nn as nn

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

# print(traD.keys())
# A = traD['labels']
# print(traD['features'].shape)
# print(A.shape)

bz=32

tra_data = torch.from_numpy(traD['features'].transpose(0,3,1,2)).type(torch.FloatTensor)
tes_data = torch.from_numpy(tesD['features'].transpose(0,3,1,2)).type(torch.FloatTensor)
val_data = torch.from_numpy(valD['features'].transpose(0,3,1,2)).type(torch.FloatTensor)

tra_label = torch.from_numpy(traD['labels']).type(torch.FloatTensor)
tes_label = torch.from_numpy(tesD['labels']).type(torch.FloatTensor)
val_label = torch.from_numpy(valD['labels']).type(torch.FloatTensor)

tra_dataset = torch.utils.data.TensorDataset(tra_data, tra_label)
tes_dataset = torch.utils.data.TensorDataset(tes_data, tes_label)
val_dataset = torch.utils.data.TensorDataset(val_data, val_label)

tra_dataloader = torch.utils.data.DataLoader(dataset = tra_dataset, batch_size=bz, shuffle=True)
tes_dataloader = torch.utils.data.DataLoader(dataset = tes_dataset, batch_size=bz, shuffle=False)
val_dataloader = torch.utils.data.DataLoader(dataset = val_dataset, batch_size=bz, shuffle=False)

print(traD['labels'])
