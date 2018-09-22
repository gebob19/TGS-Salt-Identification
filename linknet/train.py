
# coding: utf-8

# In[44]:


import pandas as pd
import numpy as np
import tensorflow as tf

from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator

from losses import dice_loss, bce_logdice_loss, bce_dice_loss
from models import linknet
from helpers import get_data, TrainValTensorBoard

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[26]:


data_path = '../data/processed-data'

xtrain, xval, ytrain, yval, dtrain, dval, idtrain, idval = get_data(data_path)


# In[52]:


lr = 1e-2
BATCH_SIZE = 5
EPOCHS = 1

# dim based off the linknet paper
H, W, C = 256, 256, 1

model = linknet((H, W, C), lr, bce_dice_loss)


# In[53]:


# https://github.com/keras-team/keras/issues/3386
def createGenerator( X, I, Y):
    while True:
        # suffled indices    
        idx = np.random.permutation(X.shape[0])
        # create image generator
        datagen = ImageDataGenerator(zoom_range=0.1,
                                     width_shift_range=0.01, 
                                     height_shift_range=0.3,
                                     horizontal_flip=True)

        batches = datagen.flow( X[idx], Y[idx], batch_size=BATCH_SIZE, shuffle=False)
        idx0 = 0
        for batch in batches:
            idx1 = idx0 + batch[0].shape[0]
            yield [batch[0], I[ idx[ idx0:idx1 ] ]], batch[1]

            idx0 = idx1
            if idx1 >= X.shape[0]:
                break


# In[54]:


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


# In[55]:


model.fit_generator(generator=createGenerator(xtrain, dtrain, ytrain),
                    steps_per_epoch=np.ceil(float(len(xtrain)) / float(BATCH_SIZE)),
                    epochs=EPOCHS,
                    verbose=1,
                    callbacks=callbacks,
                    validation_data=([xval, dval], yval), 
                    validation_steps=np.ceil(float(len(xval)) / float(BATCH_SIZE)))


# In[28]:


# !ipython nbconvert --to=python train.ipynb

