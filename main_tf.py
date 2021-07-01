#%% Packages
import numpy as np
import config
import pickle
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow.keras as keras
import tensorflow as tf
import func
import model_tf

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

tra_data = func._norm(np.concatenate((traRGB, traGray), axis=1)).transpose(0,2,3,1)
val_data = func._norm(np.concatenate((valRGB, valGray), axis=1)).transpose(0,2,3,1)
tra_label = traD['labels']
val_label = valD['labels']

tra_data = tf.image.convert_image_dtype(tra_data, dtype=tf.float16, saturate=False)
val_data = tf.image.convert_image_dtype(val_data, dtype=tf.float16, saturate=False)

#%% Parameters
bz = config.batch
model = model_tf.ResNet18()
optim_m = keras.optimizers.Adam(learning_rate=config.lr, amsgrad=config.amsgrad)

#%% Train
model.compile(optimizer=optim_m, 
              loss=keras.losses.SparseCategoricalCrossentropy(
                  from_logits=True), 
              metrics=['accuracy'])
history = model.fit(tra_data, tra_label, batch_size=bz,
                    epochs=config.Epoch, verbose=2, shuffle=True,
                    validation_data=(val_data, val_label))

loss = np.array(history.history['loss'])
val_acc = np.array(history.history['val_accuracy'])

model.save(os.path.join(M, 'model_tf'))

with open(os.path.join(M, 'loss_acc_tf.npy'), 'wb') as f:          
    np.save(f, loss)
    np.save(f, val_acc)
