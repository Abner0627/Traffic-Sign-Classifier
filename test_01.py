#%% Packages
import numpy as np
import config
import pickle
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import func

def _config():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-T", default=False, help="tensorflow")
    parser.add_argument("-P", default=False, help="pytorch")
    return parser.parse_args()

#%% Load
dpath = './traffic-signs-data'
tesF = 'test.p'
M = './model'
args = _config()

with open(os.path.join(dpath, tesF), 'rb') as f:
    tesD = pickle.load(f)

# Img
tesRGB = tesD['features'].transpose(0,3,1,2)
tesGray = func._gray(tesRGB)

tes_data = func._norm(np.concatenate((tesRGB, tesGray), axis=1))
tes_label = tesD['labels']

#%% TF
if args.T:
    import tensorflow.keras as keras
    import tensorflow as tf
    tesDa = tes_data.transpose(0,2,3,1)
    tesDa = tf.image.convert_image_dtype(tesDa, dtype=tf.float16, saturate=False)
    model = keras.models.load_model(os.path.join(M, 'model_tf'))
    pro_model = keras.Sequential([model, keras.layers.Softmax()])
    pred = pro_model.predict(tesDa)
    pro_pred = np.argmax(pred, axis=1)

    hd = np.sum(pro_pred==tes_label)
    acc = (hd/tes_label.shape[0])

    print('\n=========================')
    print('test_acc >> {:.4f}'.format(acc)) 
    print('=========================')

elif args.P:
    import torch
    tesDa = torch.from_numpy(tes_data).type(torch.FloatTensor)
    tesLa = torch.from_numpy(tes_label).type(torch.FloatTensor)
    tes_dataset = torch.utils.data.TensorDataset(tesDa, tesLa)
    tes_dataloader = torch.utils.data.DataLoader(dataset = tes_dataset, batch_size=32, shuffle=False)
    model = torch.load(os.path.join(M, 'model_pt.pth'))
    model.cpu()
    model.eval()
    with torch.no_grad():
        for ntes, (Data_E, Label_E) in enumerate(tes_dataloader):
            data_rgb = Data_E[:,:3,:,:]
            data_gray = Data_E[:,-1,:,:].unsqueeze(1)
            pred = model(data_rgb, data_gray)

            out = pred.cpu().data.numpy()
            pr  = np.argmax(out, axis=1)
            if ntes==0:
                prd = pr
            else:
                prd = np.concatenate((prd, pr))

        te = tesD['labels']
        hd = np.sum(prd==te)
        acc = (hd/te.shape[0])

        print('\n=========================')
        print('test_acc >> {:.4f}'.format(acc)) 
        print('=========================')