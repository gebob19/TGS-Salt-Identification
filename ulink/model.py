from keras.layers import Input, Conv2D, Activation, BatchNormalization, add, \
                         MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout, \
                         Conv2DTranspose, Cropping2D, Lambda, Multiply, concatenate
from keras import Model
from keras.optimizers import Adam, RMSprop
from keras.metrics import binary_accuracy

from keras.regularizers import l2

from keras.utils import multi_gpu_model
import sys
sys.path.insert(0, '..')
from losses import iou

def batchnorm_relu(x):
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def conv_block(x, fs, batch_norm=True):
    x = Conv2D(fs, (3, 3), strides=(1, 1), padding='same')(x)
    if batch_norm:
        x = batchnorm_relu(x)
    return x

def residual_block(input_tensor, fs, batch_norm=False):
    x = batchnorm_relu(input_tensor)
    
    x = conv_block(x, fs)
    x = conv_block(x, fs, batch_norm=False)
    
    x = add([x, input_tensor])
    
    if batch_norm: 
        x = batchnorm_relu(x)
    return x

def encoder_block(x, fs, pdrop):
    x = Conv2D(fs, (3, 3), activation=None, padding='same')(x)
    x = residual_block(x, fs)
    x = residual_block(x, fs, batch_norm=True)
    
    pool = MaxPooling2D((2, 2))(x)
    pool = Dropout(pdrop)(pool)
    
    return x, pool

def bottleneck(x, fs):
    x = Conv2D(fs, (3, 3), activation=None, padding='same')(x)
    x = residual_block(x, fs)
    x = residual_block(x, fs, batch_norm=True)
    
    return x

def decoder_block(x, encode, fs, pdrop, padding='same'):
    x = Conv2DTranspose(fs, (3, 3), strides=(2, 2), padding=padding)(x)
    x = concatenate([x, encode])
    x = Dropout(pdrop)(x)
    
    x = Conv2D(fs, (3, 3), activation=None, padding='same')(x)
    x = residual_block(x, fs)
    x = residual_block(x, fs, batch_norm=True)
    return x

def ulink(input_shape, fs, learning_rate, loss, gpus):
    input_img = Input(input_shape)
    
    e1, x = encoder_block(input_img, fs*1, pdrop=0.25)
    e2, x = encoder_block(x, fs*2, pdrop=0.5)
    e3, x = encoder_block(x, fs*4, pdrop=0.5)
    e4, x = encoder_block(x, fs*8, pdrop=0.5)
    
    x = bottleneck(x, fs*16)
    
    x = decoder_block(x, e4, fs*8, pdrop=0.5)
    x = decoder_block(x, e3, fs*4, pdrop=0.5, padding='valid')
    x = decoder_block(x, e2, fs*2, pdrop=0.5)
    x = decoder_block(x, e1, fs*1, pdrop=0.25, padding='valid')
    
    x = Conv2D(1, (1,1), padding="same", activation='sigmoid')(x)
    
    model = Model(input_img, x)
    if gpus > 1:
        model = multi_gpu_model(model, gpus=gpus)
        
    model.compile(loss=loss, optimizer=Adam(learning_rate), 
                  metrics=[binary_accuracy, iou])
    return model