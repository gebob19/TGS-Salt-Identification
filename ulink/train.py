#!/usr/bin/env python
# coding: utf-8

# In[38]:


import sys
sys.path.insert(0, '..')
from helpers import TrainValTensorBoard, get_data, createGenerator
from losses import dice_loss, binary_crossentropy
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from model import ulink
# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')


# In[44]:


data_path = '../data/original-data'

xtrain, xval, ytrain, yval, dtrain, dval, idtrain, idval = get_data(data_path)


# In[22]:


H, W, C = 101, 101, 1

BATCH_SIZE = 40
EPOCHS = 100
data_aug = False

start_neuron = 16

# start @ 1e-3
learning_rate = 1e-3
loss = binary_crossentropy

model = ulink((H, W, C), start_neuron, learning_rate, loss, gpus=2)
# model.load_weights('ulink_weights.hdf5')


# In[42]:


lr_plat = ReduceLROnPlateau(monitor='val_iou',
                               factor=0.2,
                               patience=4,
                               verbose=1,
                               min_delta=1e-4,
                               mode='max')
early_stop = EarlyStopping(monitor='val_iou',
                           patience=16,
                           verbose=1,
                           min_delta=1e-4,
                           mode='max')
m_checkpoint = ModelCheckpoint(monitor='val_iou',
                             filepath='ulink_weights.hdf5',
                             save_best_only=True,
                             verbose=1,
                             mode='max')
tb = TrainValTensorBoard(write_graph=False)
callbacks = [lr_plat, early_stop, m_checkpoint, tb]


# In[46]:


if data_aug:
    datagen = ImageDataGenerator(horizontal_flip=True)
    model.fit_generator(datagen.flow(xtrain, ytrain, batch_size=BATCH_SIZE),                                             steps_per_epoch=np.ceil(float(len(xtrain)) / float(BATCH_SIZE)),
                        epochs=EPOCHS,
                        verbose=1,
                        callbacks=callbacks,
                        validation_data=(xval, yval),
                        validation_steps=np.ceil(float(len(xval)) / float(BATCH_SIZE)))
else:
    model.fit([xtrain, dtrain], ytrain, batch_size=BATCH_SIZE,
              epochs=EPOCHS,
              shuffle = True,
              verbose=1,
              callbacks=callbacks,
              validation_data=([xval, dval], yval))


# In[ ]:




