#!/usr/bin/env python
# coding: utf-8

# In[38]:


import sys
sys.path.insert(0, '..')
from helpers import TrainValTensorBoard, get_data, createGenerator
from losses import dice_loss, binary_crossentropy
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping

# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')


# In[44]:


data_path = '../data/original-data'

xtrain, xval, ytrain, yval, dtrain, dval, idtrain, idval = get_data(data_path)


# In[22]:


H, W, C = 101, 101, 1

BATCH_SIZE = 32
EPOCHS = 100
data_aug = False

start_neuron = 16
learning_rate = 1e-2
loss = binary_crossentropy


# In[23]:


from model import ulink


# In[41]:


model = ulink((H, W, C), start_neuron, learning_rate, loss, gpus=1)


# In[42]:


lr_plat = ReduceLROnPlateau(monitor='val_iou',
                               factor=0.2,
                               patience=5,
                               verbose=1,
                               min_delta=1e-4,
                               mode='max')
early_stop = EarlyStopping(monitor='val_iou',
                           patience=10,
                           verbose=1,
                           min_delta=1e-4,
                           mode='max')
m_checkpoint = ModelCheckpoint(monitor='val_iou',
                             filepath='ulink_weights.hdf5',
                             save_best_only=True,
                             mode='max')
tb = TrainValTensorBoard(write_graph=False)
callbacks = [lr_plat, early_stop, m_checkpoint, tb]


# In[46]:


if data_aug:
    datagen = ImageDataGenerator(zoom_range=0.1,
                                     width_shift_range=0.1,
                                     height_shift_range=0.1,
                                     horizontal_flip=True)
    model.fit_generator(datagen.flow(xtrain, ytrain, batch_size=BATCH_SIZE),                                             steps_per_epoch=np.ceil(float(len(xtrain)) / float(BATCH_SIZE)),
                        epochs=EPOCHS,
                        verbose=1,
                        callbacks=callbacks,
                        validation_data=(xval, yval),
                        validation_steps=np.ceil(float(len(xval)) / float(BATCH_SIZE)))
else:
    model.fit(xtrain, ytrain, batch_size=BATCH_SIZE,
              epochs=EPOCHS,
              verbose=1,
              callbacks=callbacks,
              validation_data=(xval, yval))


# In[ ]:




