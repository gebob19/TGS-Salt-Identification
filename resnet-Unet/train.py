#!/usr/bin/env python
# coding: utf-8


import numpy as np

from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
import h5py

from helpers import TrainValTensorBoard, get_data
from models import get_unet
from losses import dice_loss

from keras import backend as K
import tensorflow as tf

sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, device_count = {'CPU' : 16, 'GPU' : 1}))
K.set_session(sess)

# In[3]:


LOGDIR='logs'


# In[4]:

print('getting data...')
xtrain, xval, ytrain, yval, dtrain, dval, idtrain, idval = get_data()

# trim down data for testing purposes

# xtrain = xtrain[:200]
# ytrain = ytrain[:200]

# xval = xval[:50]
# yval = yval[:50]

H, W = 224, 224
lr = 3e-3
loss = dice_loss

BATCH_SIZE = 5
EPOCHS = 5

print('getting model...')
model = get_unet(H, W, lr, loss)


# In[5]:


datagen = ImageDataGenerator(zoom_range=0.1,
                             width_shift_range=0.1,
                             height_shift_range=0.1,
                             vertical_flip=True)
datagen.fit(xtrain)


# In[6]:


# define callbacks
lr_plat = ReduceLROnPlateau(monitor='val_dice_coef',
                               factor=0.2,
                               patience=5,
                               verbose=1,
                               min_delta=1e-4,
                               mode='max')
early_stop = EarlyStopping(monitor='val_dice_coef',
                           patience=10,
                           verbose=1,
                           min_delta=1e-4,
                           mode='max')
# m_checkpoint = ModelCheckpoint(monitor='val_dice_coef',
#                              filepath='model_weights.hdf5',
#                              save_best_only=True,
#                              mode='max')
tb = TrainValTensorBoard(write_graph=False)
callbacks = [lr_plat, early_stop, tb]


# In[7]:

print('starting training...')
model.fit_generator(generator=datagen.flow(xtrain, ytrain, batch_size=BATCH_SIZE),
                    steps_per_epoch=np.ceil(float(len(xtrain)) / float(BATCH_SIZE)),
                    epochs=EPOCHS,
                    verbose=1,
                    callbacks=callbacks,
                    validation_data=(xval, yval), 
                    validation_steps=np.ceil(float(len(xval)) / float(BATCH_SIZE)))


# In[ ]:




