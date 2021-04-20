import numpy as np
import os
import pickle
import matplotlib as plt

dpath = './traffic-signs-data'
traF = 'train.p'
with open(os.path.join(dpath, traF), 'rb') as f:
    traD = pickle.load(f)
