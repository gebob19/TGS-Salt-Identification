
# coding: utf-8

# In[24]:


import pandas as pd
import numpy as np
import tensorflow as tf

from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator

from losses import dice_loss
from models import linknet
from helpers import get_data, TrainValTensorBoard

# gpu util
from keras import backend as K
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, device_count = {'CPU' : 16, 'GPU' : 1}))
K.set_session(sess)


# In[17]:


data_path = '../data/proc-data'

xtrain, xval, ytrain, yval, dtrain, dval, idtrain, idval = get_data(data_path)


# In[25]:


lr = 1e-2
BATCH_SIZE = 5
EPOCHS = 1

# dim based off the linknet paper
H, W, C = 256, 128, 1

model = linknet((H, W, C), lr, dice_loss)


# In[292]:


datagen = ImageDataGenerator(zoom_range=0.1, width_shift_range=0.01, 
                             height_shift_range=0.3,
                             horizontal_flip=True)
datagen.fit(xtrain)


# In[293]:


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
m_checkpoint = ModelCheckpoint(monitor='val_dice_coef',
                             filepath='model_weights.hdf5',
                             save_best_only=True,
                             mode='max')
tb = TrainValTensorBoard(write_graph=False)
callbacks = [lr_plat, early_stop, m_checkpoint, tb]


# In[26]:


model.fit_generator(generator=datagen.flow(xtrain, ytrain, batch_size=BATCH_SIZE),
                    steps_per_epoch=np.ceil(float(len(xtrain)) / float(BATCH_SIZE)),
                    epochs=EPOCHS,
                    verbose=1,
                    callbacks=callbacks,
                    validation_data=(xval, yval), 
                    validation_steps=np.ceil(float(len(xval)) / float(BATCH_SIZE)))



