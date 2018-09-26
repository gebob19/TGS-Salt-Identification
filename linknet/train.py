
# coding: utf-8

# In[44]:


import pandas as pd
import numpy as np
import tensorflow as tf

from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping

from models import linknet
import sys
sys.path.insert(0, '..')
from losses import dice_loss, bce_logdice_loss, bce_dice_loss, binary_crossentropy, lovasz_dice_loss, lovasz_loss
from helpers import get_data, TrainValTensorBoard, createGenerator

from keras import backend as K
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, device_count = {'CPU' : 16, 'GPU' : 1}))
K.set_session(sess)

data_path = 'data/proccessed-data'

xtrain, xval, ytrain, yval, dtrain, dval, idtrain, idval = get_data(data_path)


lr = 4e-5
BATCH_SIZE = 64
EPOCHS = 100

# dim based off the linknet paper
H, W, C = 256, 256, 1
filter_sizes = [32, 64, 128, 256, 512]

model = linknet((H, W, C), lr, filter_sizes, binary_crossentropy)
# model.load_weights('model_weights.hdf5')

# define callbacks
lr_plat = ReduceLROnPlateau(monitor='val_binary_accuracy',
                               factor=0.5,
                               patience=5,
                               verbose=1,
                               min_delta=1e-4,
                               mode='max')
early_stop = EarlyStopping(monitor='val_binary_accuracy',
                           patience=20,
                           verbose=1,
                           min_delta=1e-4,
                           mode='max')
m_checkpoint = ModelCheckpoint(monitor='val_dice_coef',
                             filepath='model_weights1.hdf5',
                             save_best_only=True,
                             verbose=1,
                             mode='max')
tb = TrainValTensorBoard(write_graph=False)
callbacks = [lr_plat, early_stop, m_checkpoint, tb]

model.fit_generator(generator=createGenerator(xtrain, dtrain, ytrain, BATCH_SIZE),
                    steps_per_epoch=np.ceil(float(len(xtrain)) / float(BATCH_SIZE)),
                    epochs=EPOCHS,
                    verbose=1,
                    callbacks=callbacks,
                    validation_data=([xval, dval], yval), 
                    validation_steps=np.ceil(float(len(xval)) / float(BATCH_SIZE)))

# model.fit([xtrain, dtrain], ytrain, batch_size=BATCH_SIZE,
#         epochs=EPOCHS,
#         verbose=1,
#         callbacks=callbacks,
#         validation_data=([xval, dval], yval))

# !ipython nbconvert --to=python train.ipynb

