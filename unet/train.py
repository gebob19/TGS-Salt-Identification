
# coding: utf-8

# In[86]:


import numpy as np
import tensorflow as tf

from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator

from model import unet
import sys
sys.path.insert(0, '..')
from helpers import TrainValTensorBoard, get_data, createGenerator
from losses import dice_loss, bce_dice_loss, bce_logdice_loss

from keras import backend as K
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, device_count = {'CPU' : 16, 'GPU' : 1}))
K.set_session(sess)

# In[6]:


data_path = '../linknet/data/proccessed-data'

xtrain, xval, ytrain, yval, dtrain, dval, idtrain, idval = get_data(data_path)


# In[87]:


H, W, C = 256, 256, 1

learning_rate = 3e-5
BATCH_SIZE = 30
EPOCHS = 100

filter_sizes = [32, 64, 128, 256]
bn_fsize = 64
loss = dice_loss

model = unet((H, W, C), filter_sizes, bn_fsize, learning_rate, loss)


# In[82]:


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
                             save_best_only=False,
                             verbose=1,
                             mode='max')
tb = TrainValTensorBoard(write_graph=False)
callbacks = [lr_plat, early_stop, m_checkpoint, tb]


# In[83]:


# attempt to over train model first => evaulate its capacity
model.fit([xtrain, dtrain], ytrain, batch_size=BATCH_SIZE,
#         steps_per_epoch=np.ceil(float(len(xtrain)) / float(BATCH_SIZE)),
        epochs=EPOCHS,
        verbose=1,
        callbacks=callbacks,
        validation_data=([xval, dval], yval))
#         validation_steps=np.ceil(float(len(xval)) / float(BATCH_SIZE)))


# In[ ]:


# fit with depth dimension
# model.fit_generator(generator=createGenerator(xtrain, dtrain, ytrain),
#                     steps_per_epoch=np.ceil(float(len(xtrain)) / float(BATCH_SIZE)),
#                     epochs=EPOCHS,
#                     verbose=1,
#                     callbacks=callbacks,
#                     validation_data=([xval, dval], yval), 
#                     validation_steps=np.ceil(float(len(xval)) / float(BATCH_SIZE)))


# In[ ]:


# get_ipython().system('ipython nbconvert --to=python train.ipynb')

