from keras.layers import Input, Conv2D, Activation, BatchNormalization, add, \
                         MaxPool2D, GlobalAveragePooling2D, Dense, \
                         Conv2DTranspose, Cropping2D, Lambda, Multiply, concatenate, Dropout
from keras import Model
from keras.metrics import binary_accuracy
from keras.optimizers import Adam, RMSprop

from keras.regularizers import l2

import sys
sys.path.insert(0, '..')
from losses import dice_coef, lovasz_loss


def init_block(input_img, fsize):
    x = Conv2D(fsize, (7, 7), strides=(2, 2), padding='same', kernel_regularizer=l2(0.))(input_img)
    x = MaxPool2D((3, 3), strides=(2, 2), padding='same', kernel_regularizer=l2(0.))(x)
    return x

def encode_block(input_tensor, filters, ksize=(3, 3)):
    f_in, f_out = filters
    
    x = Conv2D(f_out, (1, 1), strides=(2, 2), padding='same', kernel_regularizer=l2(0.))(input_tensor)
#     x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    
    x = Conv2D(f_out, ksize, strides=(1, 1), padding='same', kernel_regularizer=l2(0.))(x)
#     x = BatchNormalization(axis=3)(x)
    
    shortcut = Conv2D(f_out, (1, 1), strides=(2, 2), padding='same', kernel_regularizer=l2(0.))(input_tensor)
#     shortcut = BatchNormalization(axis=3)(shortcut)
    
    x = add([x, shortcut])
    ec1 = Activation('relu')(x)
    
    x = Conv2D(f_out, ksize, strides=(1, 1), padding='same', kernel_regularizer=l2(0.))(ec1)
#     x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    
    x = Conv2D(f_out, ksize, strides=(1, 1), padding='same', kernel_regularizer=l2(0.))(x)
#     x = BatchNormalization(axis=3)(x)
    
    x = add([x, ec1])
    x = Activation('relu')(x)
    
    return x

def encoder(input_img, filter_sizes):
    f1, f2, f3, f4 = filter_sizes
    
    e1 = encode_block(input_img, [f1, f1])
    e2 = encode_block(e1, [f1, f2])
    e3 = encode_block(e2, [f2, f3])
    e4 = encode_block(e3, [f3, f4])
    return [e1, e2, e3, e4]

def decode_block(input_tensor, fsizes, ksize=(3, 3)):
    f_in, f_out = fsizes
    x = Conv2D(int(f_in/4), (1, 1), strides=(1, 1), padding='same', kernel_regularizer=l2(0.))(input_tensor)
    x = Conv2DTranspose(int(f_in/4), ksize, strides=(2, 2), padding='same', kernel_regularizer=l2(0.))(x)
    x.set_shape(x._keras_shape)
    x = Conv2D(f_out, (1, 1), strides=(1, 1), padding='same', kernel_regularizer=l2(0.))(x)                   
    return x

def decoder(e1, e2, e3, e4, filter_sizes):
    f1, f2, f3, f4 = filter_sizes
    
    d4 = decode_block(e4, [f4, f3])
    d4 = add([e3, d4])
    d3 = decode_block(d4, [f3, f2])
    d3 = add([e2, d3])
    d2 = decode_block(d3, [f2, f1])
    d2 = add([e1, d2])
    
    d1 = decode_block(d2, [f1, f1])
    return d1

def output_block(input_tensor, ksize=(3, 3)):
    x = Conv2DTranspose(32, ksize, strides=(2, 2), padding='same', kernel_regularizer=l2(0.))(input_tensor)
    x.set_shape(x._keras_shape)

    x = Conv2D(32, ksize, strides=(1, 1), padding='same', kernel_regularizer=l2(0.))(x)
    x = Conv2DTranspose(1, (2, 2), strides=(2, 2), padding='same', kernel_regularizer=l2(0.))(x)
    x.set_shape(x._keras_shape)
    
    return x

def bottle_neck(input_tensor, fsize, ksize=(1, 1)):
    b1 = Conv2D(fsize, ksize, strides=(1, 1), padding='same', kernel_regularizer=l2(0.))(input_tensor)
    b2 = Conv2D(fsize, ksize, strides=(1, 1), padding='same', kernel_regularizer=l2(0.))(b1)
    b3 = Conv2D(fsize, ksize, strides=(1, 1), padding='same', kernel_regularizer=l2(0.))(b2)
    b4 = Conv2D(fsize, ksize, strides=(1, 1), padding='same', kernel_regularizer=l2(0.))(b3)
    return add([b1, b2, b3, b4])
    

def linknet(input_shape, learing_rate, filter_sizes, loss):
    input_img = Input(input_shape)
    depth_input = Input((1,))
    
    x = init_block(input_img, filter_sizes[0])

    e1, e2, e3, e4 = encoder(x, filter_sizes)
    x = add([e4, depth_input])
    x = bottle_neck(x, filter_sizes[-1])
    x = decoder(e1, e2, e3, e4, filter_sizes)

    y_pred = output_block(x)
    
    model = Model([input_img, depth_input], y_pred)
    model.compile(loss=loss, optimizer=Adam(learing_rate), metrics=[dice_coef, binary_accuracy])
    
    return model