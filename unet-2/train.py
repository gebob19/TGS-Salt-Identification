from model import unet

import sys
sys.path.insert(0, '..')
from helpers import TrainValTensorBoard, get_data, createGenerator
from losses import dice_loss, binary_crossentropy, bce_dice_loss, iou
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import tensorflow as tf 

data_path = '../data/processed-data'

# 224 x 224 x 3
xtrain, xval, ytrain, yval, dtrain, dval, idtrain, idval = get_data(data_path)

tf.reset_default_graph()

BATCH_SIZE = 15
EPOCHS = 100
data_aug = False

learning_rate = 4e-3
loss = binary_crossentropy


model, unfreeze_layers = unet(loss, learning_rate, gpus=2, freeze=False)
model.load_weights('model_weights.hdf5')

# # unfreeze
# for layer in unfreeze_layers:
#     layer.trainable = True
    
# # recompile 

# from keras.optimizers import Adam, RMSprop
# from keras.metrics import binary_accuracy

# model.compile(loss=loss, optimizer=Adam(learning_rate), 
#               metrics=[binary_accuracy, iou])


lr_plat = ReduceLROnPlateau(monitor='iou',
                               factor=0.2,
                               patience=4,
                               verbose=1,
                               min_delta=1e-4,
                               mode='max')
early_stop = EarlyStopping(monitor='iou',
                           patience=16,
                           verbose=1,
                           min_delta=1e-4,
                           mode='max')
m_checkpoint2 = ModelCheckpoint(
                             filepath='model_weights.hdf5',
                             save_best_only=False,
                             verbose=1)

m_checkpoint = ModelCheckpoint(monitor='iou',
                             filepath='best_val_weights.hdf5',
                             save_best_only=True,
                             verbose=1,
                             mode='max')
tb = TrainValTensorBoard(write_graph=False)
callbacks = [lr_plat, early_stop, m_checkpoint, m_checkpoint2, tb]


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
    model.fit(xtrain, ytrain, batch_size=BATCH_SIZE,
              epochs = EPOCHS,
              shuffle = True,
              verbose=1,
              callbacks=callbacks,
              validation_data=(xval, yval))