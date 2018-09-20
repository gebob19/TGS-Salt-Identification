#!/usr/bin/env python
# coding: utf-8

# In[254]:
lr = 5e-5
BATCH_SIZE = 128
EPOCHS = 100


import pandas as pd
import numpy as np
from helpers import get_data, TrainValTensorBoard

# In[184]:


from keras.layers import Input, Conv2D, Activation, BatchNormalization, add,                          MaxPool2D, GlobalAveragePooling2D, Dense
from keras import Model
from keras.regularizers import l2

from keras import backend as K
import tensorflow as tf

sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, device_count = {'CPU' : 4, 'GPU' : 1}))
K.set_session(sess)


# In[282]:


salt_df = pd.read_csv('salt-identifier.csv')
salt_df.head()


# In[283]:


xtrain, xval, ytrain, yval, dtrain, dval, idtrain, idval = get_data('../tfmodel-data')


# In[284]:


idtrain = idtrain.astype(str)
idval = idval.astype(str)

ytrain = np.array([salt_df[salt_df.id == i]['salt'].values[0] for i in idtrain])
yval = np.array([salt_df[salt_df.id == i]['salt'].values[0] for i in idval])

ytrain.shape, xtrain.shape, yval.shape, xval.shape


# In[286]:


def init_block(input_img):
    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', kernel_regularizer=l2(0.))(input_img)
    x = MaxPool2D((3, 3), strides=(2, 2), padding='same')(x)
    return x


# In[287]:


def encoder(input_img):
    e1 = encode_block(input_img, [64, 64])
    e2 = encode_block(e1, [64, 128])
    e3 = encode_block(e2, [128, 256])
    e4 = encode_block(e3, [256, 512])
    return e4


# In[288]:


def encode_block(input_tensor, filters, ksize=(3, 3)):
    f_in, f_out = filters
    
    x = Conv2D(f_out, (1, 1), strides=(2, 2), padding='same', kernel_regularizer=l2(0.))(input_tensor)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    
    x = Conv2D(f_out, ksize, strides=(1, 1), padding='same', kernel_regularizer=l2(0.))(x)
    x = BatchNormalization(axis=3)(x)
    
    shortcut = Conv2D(f_out, (1, 1), strides=(2, 2), padding='same', kernel_regularizer=l2(0.))(input_tensor)
    shortcut = BatchNormalization(axis=3)(shortcut)
    
    x = add([x, shortcut])
    ec1 = Activation('relu')(x)
    
    x = Conv2D(f_out, ksize, strides=(1, 1), padding='same', kernel_regularizer=l2(0.))(ec1)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    
    x = Conv2D(f_out, ksize, strides=(1, 1), padding='same', kernel_regularizer=l2(0.))(x)
    x = BatchNormalization(axis=3)(x)
    
    x = add([x, ec1])
    x = Activation('relu')(x)
    
    return x


# In[289]:


H, W, C = 224, 224, 1

# salt / no salt
classes = 1

input_img = Input((H, W, C))

x = init_block(input_img)
x = encoder(x)

x = GlobalAveragePooling2D()(x)
x = Dense(classes, activation='softmax')(x)


# In[290]:


from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint,                             EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adagrad
from keras.losses import binary_crossentropy


# In[291]:

tf_model = Model(input_img, x)
tf_model.compile(loss=binary_crossentropy, optimizer=Adagrad(lr), metrics=['accuracy'])
tf_model.load_weights('tfmodel_weights.hdf5')
# In[292]:


datagen = ImageDataGenerator(zoom_range=0.01, 
                             vertical_flip=True,
                             horizontal_flip=True)
datagen.fit(xtrain)


# In[293]:


# define callbacks
lr_plat = ReduceLROnPlateau(monitor='val_acc',
                               factor=0.5,
                               patience=3,
                               verbose=1,
                               min_delta=1e-9,
                               mode='max')
early_stop = EarlyStopping(monitor='val_acc',
                           patience=10,
                           verbose=1,
                           min_delta=1e-4,
                           mode='max')
m_checkpoint = ModelCheckpoint(monitor='val_acc',
                             filepath='tfmodel_weights.hdf5',
                             save_best_only=True,
                             mode='max')
tb = TrainValTensorBoard(write_graph=False)
callbacks = [lr_plat, early_stop, m_checkpoint, tb]


# In[294]:


tf_model.fit_generator(generator=datagen.flow(xtrain, ytrain, batch_size=BATCH_SIZE),
                    steps_per_epoch=np.ceil(float(len(xtrain)) / float(BATCH_SIZE)),
                    epochs=EPOCHS,
                    verbose=1,
                    callbacks=callbacks,
                    validation_data=(xval, yval), 
                    validation_steps=np.ceil(float(len(xval)) / float(BATCH_SIZE)))


# In[ ]:




