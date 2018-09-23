from keras.layers import Input, Conv2D, Activation, BatchNormalization, add, \
                         MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout, \
                         Conv2DTranspose, Cropping2D, Lambda, Multiply, concatenate
from keras import Model
from keras.optimizers import Adam, RMSprop
from keras.metrics import binary_accuracy

from keras.regularizers import l2
import sys
sys.path.insert(0, '..')
from losses import dice_coef

def decoder(x, out_layers, fsizes, kernel_size=3, strides=1):
    for i in range(len(fsizes)-1, -1, -1):
        x = Conv2DTranspose(fsizes[i], (3, 3), strides=(2, 2), padding="same")(x)
        x = concatenate([out_layers[i], x])
        x = Dropout(0.5)(x)
        x = Conv2D(fsizes[i], kernel_size=kernel_size, strides=strides, 
                   activation='relu', padding='same')(x)
        x = Conv2D(fsizes[i], kernel_size=kernel_size, strides=strides, 
                   activation='relu', padding='same')(x)
    return x

def encoder_block(x, fs):
    x = Conv2D(fs, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(fs, (3, 3), activation='relu', padding='same')(x)
    pool = MaxPooling2D((2, 2))(x)
    pool = Dropout(0.5)(pool)
    
    return x, pool

def encoder(x, filter_sizes):
    f1, f2, f3, f4 = filter_sizes
    
    e1, x = encoder_block(x, f1)
    e2, x = encoder_block(x, f2)
    e3, x = encoder_block(x, f3)
    e4, x = encoder_block(x, f4)
    return x, [e1, e2, e3, e4]

def bottle_neck(x, fs, depth):
    x = Conv2D(fs, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(fs, (3, 3), activation='relu', padding='same')(x)
    x = add([x, depth])
    x = Dropout(0.5)(x)
    x = Conv2D(fs, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(fs, (3, 3), activation='relu', padding='same')(x)
    x = add([x, depth])
    x = Dropout(0.5)(x)
    return x


def unet(input_shape, filter_sizes, bn_fsize, learning_rate, loss):
    input_img = Input(input_shape)
    input_depth = Input((1,))

    x, concat_layers = encoder(input_img, filter_sizes)
    x = bottle_neck(x, bn_fsize, input_depth)
    x = decoder(x, concat_layers, filter_sizes)
    x = Conv2D(1, (1, 1), activation='sigmoid')(x)

    model = Model([input_img, input_depth], x)
    model.compile(loss=loss, optimizer=Adam(learning_rate), metrics=[dice_coef, binary_accuracy])
    return model
    