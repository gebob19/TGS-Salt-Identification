from keras.layers import Input, Conv2D, Activation, BatchNormalization, add, \
                         MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout, \
                         Conv2DTranspose, Cropping2D, Lambda, Multiply, concatenate
from keras import Model
from keras.optimizers import Adam, RMSprop
from keras.metrics import binary_accuracy

from keras.utils import multi_gpu_model
from keras.regularizers import l2
import sys
sys.path.insert(0, '..')
from losses import dice_coef

def decoder(x, concat_layers, filter_sizes, kernel_size=3, strides=1):
    f1, f2, f3, f4 = filter_sizes
    e1, e2, e3, e4 = concat_layers
    
    x = decoder_block(x, f1, e4, p_drop=0.5)
    x = decoder_block(x, f2, e3, p_drop=0.5)
    x = decoder_block(x, f3, e2, p_drop=0.25)
    x = decoder_block(x, f4, e1, p_drop=0.25)
    
    return x

def decoder_block(x, fs, concat_layer, p_drop, kernel_size=(3, 3)):
    x = Conv2DTranspose(fs, (3, 3), strides=(2, 2), padding="same")(x)
    x = concatenate([concat_layer, x])
    x = Dropout(p_drop)(x)
    x = Conv2D(fs, kernel_size=kernel_size, strides=(1, 1), 
               activation='relu', padding='same')(x)
    x = Conv2D(fs, kernel_size=kernel_size, strides=(1, 1), 
               activation='relu', padding='same')(x)
    return x

def encoder_block(x, fs, pdrop):
    x = Conv2D(fs, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(fs, (3, 3), activation='relu', padding='same')(x)
    pool = MaxPooling2D((2, 2))(x)
    pool = Dropout(pdrop)(pool)
    
    return x, pool

def encoder(x, filter_sizes):
    f1, f2, f3, f4 = filter_sizes
    
    e1, x = encoder_block(x, f1, pdrop=0.25)
    e2, x = encoder_block(x, f2, pdrop=0.25)
    e3, x = encoder_block(x, f3, pdrop=0.5)
    e4, x = encoder_block(x, f4, pdrop=0.5)
    return x, [e1, e2, e3, e4]

def bottle_neck(x, fs, depth):
    b1 = Conv2D(fs, (1, 1), activation='relu', padding='same')(x)
    b2 = Conv2D(fs, (1, 1), activation='relu', padding='same')(b1)
    b3 = Conv2D(fs, (1, 1), activation='relu', padding='same')(b2)
    b4 = Conv2D(fs, (1, 1), activation='relu', padding='same')(b3)
    
    x = add([b1, b2, b3, b4])
    x = Dropout(0.5)(x)
    return x


def unet(input_shape, filter_sizes, learning_rate, loss):
    input_img = Input(input_shape)
    input_depth = Input((1,))

    x, concat_layers = encoder(input_img, filter_sizes)
    x = bottle_neck(x, filter_sizes[-1], input_depth)
    x = decoder(x, concat_layers, filter_sizes)
    x = Conv2D(1, (1, 1), activation='sigmoid')(x)

    model = Model([input_img, input_depth], x)
    parallel_model = multi_gpu_model(model, gpus=2)
    parallel_model.compile(loss=loss, optimizer=RMSprop(learning_rate), metrics=[binary_accuracy, dice_coef])
    return parallel_model
    