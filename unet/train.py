
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
from losses import dice_loss, bce_dice_loss, bce_logdice_loss, binary_crossentropy

from keras import backend as K
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, device_count = {'CPU' : 16, 'GPU' : 1}))
K.set_session(sess)

# In[6]:


data_path = '../data/processed-data'
xtrain, xval, ytrain, yval, dtrain, dval, idtrain, idval = get_data(data_path)


# In[87]:


H, W, C = 256, 256, 1

learning_rate = 4e-5
BATCH_SIZE = 5
EPOCHS = 100

filter_sizes = [16, 32, 64, 128]#, 256]#512]#, 1024]
loss = binary_crossentropy

model = unet((H, W, C), filter_sizes, learning_rate, loss)
model.load_weights('model_weights.hdf5')

# In[82]:


# define callbacks
lr_plat = ReduceLROnPlateau(factor=0.2,
                           patience=5,
                           verbose=1,
                           min_lr=1e-7,
                           mode='max')
early_stop = EarlyStopping(monitor='val_loss',
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


# # attempt to over train model first => evaulate its capacity
# model.fit([xtrain, dtrain], ytrain, batch_size=BATCH_SIZE,
# #         steps_per_epoch=np.ceil(float(len(xtrain)) / float(BATCH_SIZE)),
#         epochs=EPOCHS,
#         verbose=1,
#         callbacks=callbacks,
#         validation_data=([xval, dval], yval))
# #         validation_steps=np.ceil(float(len(xval)) / float(BATCH_SIZE)))


# In[ ]:


# fit with depth dimension
model.fit_generator(generator=createGenerator(xtrain, dtrain, ytrain, BATCH_SIZE),
                    steps_per_epoch=np.ceil(float(len(xtrain)) / float(BATCH_SIZE)),
                    epochs=EPOCHS,
                    verbose=1,
                    callbacks=callbacks,
                    validation_data=([xval, dval], yval), 
                    validation_steps=np.ceil(float(len(xval)) / float(BATCH_SIZE)))

# In[ ]:


# get_ipython().system('ipython nbconvert --to=python train.ipynb')

