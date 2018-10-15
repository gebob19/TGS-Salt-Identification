import keras
import tensorflow as tf
from keras import layers
from keras import Model
from keras.applications.resnet50 import ResNet50
from keras.layers import Input, Conv2D, Activation, BatchNormalization, add, \
                         MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout, \
                         Conv2DTranspose, Cropping2D, Lambda, Multiply, concatenate

from keras.regularizers import l2
from keras.optimizers import Adam, RMSprop
from keras.metrics import binary_accuracy

from keras.utils import multi_gpu_model
import sys
sys.path.insert(0, '..')
from losses import iou

reg = 1e-6

def freeze_layers(layers):
    for layer in layers:
        layer.trainable = False
        
def ResNetEncode(input_shape, freeze=False):
    encoder = ResNet50(weights=None, include_top=False,
                   input_shape=input_shape)
            
    # extract concat layers (based off keras source code)
    concat_layers = []
    concat_layers_names = ['add_3', 'add_7', 'add_13']

    for ln in concat_layers_names:
        layer_out = encoder.get_layer(ln).output
        concat_layers.append(layer_out)
        
    # freeze model layers
    if freeze: freeze_layers(encoder.layers)
        
    return encoder.inputs, concat_layers, encoder.outputs[-1], encoder.layers

def batchnorm_relu(x):
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def conv_block(x, fs, batch_norm=True):
    x = Conv2D(fs, (3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(reg))(x)
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

def decoder_block(x, encode, fs, pdrop, pad=False, padding='same'):
    x = Conv2DTranspose(fs, (3, 3), strides=(2, 2), padding=padding, kernel_regularizer=l2(reg))(x)
    x.set_shape(x._keras_shape)
    
    if encode is not None:
        if pad: 
            encode = Lambda(keras.backend.spatial_2d_padding, arguments={'padding':((1, 0), (1, 0)), 'data_format':"channels_last"}, output_shape=x._keras_shape[1:])(encode)
        x = concatenate([encode, x])
    x = Dropout(pdrop)(x)
    
    x = Conv2D(fs, (3, 3), activation=None, padding='same', kernel_regularizer=l2(reg))(x)
    x = residual_block(x, fs)
    x = residual_block(x, fs, batch_norm=True)
    return x

# def crop_and_concat(x1,x2):
#     x1_shape = tf.shape(x1)
#     x2_shape = tf.shape(x2)

#     # offsets for the top left corner of the crop
#     offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, 0]
#     size = [-1, x2_shape[1], x2_shape[2], -1]
    
#     x1_crop = Lambda(tf.slice, arguments={'begin':offsets, 'size':size}, output_shape=x2._keras_shape[1:])(x1)
#     return concatenate([x1_crop, x2])

def unet(loss, learning_rate, gpus, freeze):
    
    inputs, concat_layers, encode_out, unfreeze_layers = ResNetEncode((224, 224, 3), freeze=freeze)
    x = decoder_block(encode_out, concat_layers[-1], 1024, 0.5)
    x = decoder_block(x, concat_layers[-2], 512, 0.5)
    x = decoder_block(x, concat_layers[-3], 256, 0.5, pad=True)
    x = decoder_block(x, None, 128, 0.)
    x = decoder_block(x, None, 64, 0.)
    y_pred = Conv2D(1, (1,1), padding="same", activation='sigmoid', kernel_regularizer=l2(reg))(x)
    model = Model(inputs, y_pred)
    
    if gpus > 1:
        model = multi_gpu_model(model, gpus=gpus)
        
    model.compile(loss=loss, optimizer=Adam(learning_rate), 
                  metrics=[binary_accuracy, iou])
    return model, unfreeze_layers
    
    
    