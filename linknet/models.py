from keras.layers import Input, Conv2D, Activation, BatchNormalization, add, \
                         MaxPool2D, GlobalAveragePooling2D, Dense, \
                         Conv2DTranspose, Cropping2D, Lambda, Multiply
from keras import Model
from keras.optimizers import Adam
 
from losses import dice_coef


def init_block(input_img):
    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same')(input_img)
    x = MaxPool2D((3, 3), strides=(2, 2), padding='same')(x)
    return x

def encode_block(input_tensor, filters, ksize=(3, 3)):
    f_in, f_out = filters
    
    x = Conv2D(f_out, (1, 1), strides=(2, 2), padding='same', kernel_initializer='he_normal')(input_tensor)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    
    x = Conv2D(f_out, ksize, strides=(1, 1), padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization(axis=3)(x)
    
    shortcut = Conv2D(f_out, (1, 1), strides=(2, 2), padding='same', kernel_initializer='he_normal')(input_tensor)
    shortcut = BatchNormalization(axis=3)(shortcut)
    
    x = add([x, shortcut])
    ec1 = Activation('relu')(x)
    
    x = Conv2D(f_out, ksize, strides=(1, 1), padding='same', kernel_initializer='he_normal')(ec1)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    
    x = Conv2D(f_out, ksize, strides=(1, 1), padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization(axis=3)(x)
    
    x = add([x, ec1])
    x = Activation('relu')(x)
    
    return x

def encoder(input_img):
    e1 = encode_block(input_img, [64, 64], )
    e2 = encode_block(e1, [64, 128])
    e3 = encode_block(e2, [128, 256])
    e4 = encode_block(e3, [256, 512])
    return [e1, e2, e3, e4]

def decode_block(input_tensor, fsizes, ksize=(3, 3)):
    f_in, f_out = fsizes
    x = Conv2D(int(f_in/4), (1, 1), strides=(1, 1), padding='same', kernel_initializer='he_normal')(input_tensor)
    x = Conv2DTranspose(int(f_in/4), ksize, strides=(2, 2), padding='same', kernel_initializer='he_normal')(x)
    x.set_shape(x._keras_shape)
    x = Conv2D(f_out, (1, 1), strides=(1, 1), padding='same', kernel_initializer='he_normal')(x)                   
    return x

def decoder(e1, e2, e3, e4):
    d4 = decode_block(e4, [512, 256])
    d4 = add([e3, d4])
    
    d3 = decode_block(d4, [256, 128])
    d3 = add([e2, d3])
    
    d2 = decode_block(d3, [128, 64])
    d2 = add([e1, d2])
    
    d1 = decode_block(d2, [64, 64])
    return d1

def output_block(input_tensor, ksize=(3, 3)):
    x = Conv2DTranspose(32, ksize, strides=(2, 2), padding='same', kernel_initializer='he_normal')(input_tensor)
    x.set_shape(x._keras_shape)

    x = Conv2D(32, ksize, strides=(1, 1), padding='same', kernel_initializer='he_normal')(x)
    x = Conv2DTranspose(1, (2, 2), strides=(2, 2), padding='same', kernel_initializer='he_normal')(x)
    x.set_shape(x._keras_shape)
    
    return x

def linknet(input_shape, learing_rate, loss):
    input_img = Input(input_shape)
    depth_input = Input((1,))
    
    x = init_block(input_img)

    e1, e2, e3, e4 = encoder(x)
    x = add([x, depth_input])
    x = decoder(e1, e2, e3, e4)

    y_pred = output_block(x)
    
    model = Model([input_img, depth_input], y_pred)
    model.compile(loss=loss, optimizer=Adam(learing_rate), metrics=[dice_coef])
    
    return model