from model import unet

import sys
sys.path.insert(0, '..')
from helpers import TrainValTensorBoard, get_data, createGenerator
from losses import dice_loss, binary_crossentropy, bce_dice_loss
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

data_path = '../data/processed-data'

# 224 x 224 x 3
xtrain, xval, ytrain, yval, dtrain, dval, idtrain, idval = get_data(data_path)

tf.reset_default_graph()

BATCH_SIZE = 5
EPOCHS = 100
data_aug = False

learning_rate = 1e-3
loss = binary_crossentropy


model = unet(gpus=2, freeze=True)

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
                             filepath='model_weights.hdf5',
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
    model.fit(xtrain, ytrain, batch_size=BATCH_SIZE,
              epochs = EPOCHS,
              shuffle = True,
              verbose=1,
              callbacks=callbacks,
              validation_data=(xval, yval))