import keras
import tensorflow as tf
from keras import layers
from keras import Model
from keras.applications.resnet50 import ResNet50
from keras.optimizers import Adam

from losses import dice_coef

def ResNetEncode(input_shape, freeze=False):
    
    encoder = ResNet50(weights='imagenet', include_top=False,
                   input_shape=input_shape)
            
    # extract concat layers
    concat_layers = []
    concat_layers_names = ['add_4', 'add_8', 'add_14']

    for ln in concat_layers_names:
        layer_out = encoder.get_layer(ln).output
        concat_layers.append(layer_out)
        
    # freeze model layers
    if freeze: freeze_layers(encoder.layers)
        
    # small pass through final layer for bottle neck
    final_layer = concat_layers[-1]
    final_layer = layers.MaxPooling2D((3, 3), strides=(2, 2))(final_layer)
        
    return encoder.inputs, concat_layers, final_layer

def freeze_layers(layers):
    for layer in layers:
        layer.trainable = False

def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2DTranspose(filters1, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a')(input_tensor)
    x.set_shape(x._keras_shape)
    
    x = layers.BatchNormalization(axis=3, name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2DTranspose(filters2, kernel_size,
                      padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b')(x)
    x.set_shape(x._keras_shape)
    
    x = layers.BatchNormalization(axis=3, name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2DTranspose(filters3, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2c')(x)
    x.set_shape(x._keras_shape)
    
    x = layers.BatchNormalization(axis=3, name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = layers.Activation('relu')(x)
    return x


def convT_block(input_tensor,
               kernel_size,
               filters,
               stage,
               block,
               strides=(2, 2)):
    """A block that has a conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        strides: Strides for the first conv layer in the block.
    # Returns
        Output tensor for the block.
    Note that from stage 3,
    the first conv layer at main path is with strides=(2, 2)
    And the shortcut should have strides=(2, 2) as well
    """
    filters1, filters2, filters3 = filters
 
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2DTranspose(filters1, (1, 1), strides=strides,
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a')(input_tensor)
    x.set_shape(x._keras_shape)
    
    x = layers.BatchNormalization(axis=3, name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2DTranspose(filters2, kernel_size, padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b')(x)
    x.set_shape(x._keras_shape)
    
    x = layers.BatchNormalization(axis=3, name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2DTranspose(filters3, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2c')(x)
    x.set_shape(x._keras_shape)
    
    x = layers.BatchNormalization(axis=3, name=bn_name_base + '2c')(x)

    shortcut = layers.Conv2DTranspose(filters3, (1, 1), strides=strides,
                             kernel_initializer='he_normal',
                             name=conv_name_base + '1')(input_tensor)
    x.set_shape(x._keras_shape)
    
    shortcut = layers.BatchNormalization(
        axis=3, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)
    return x


def trim_concat(x, concat_layer):
    # resize & concat
    shape = x.get_shape().as_list()
    rs_l = layers.Lambda(tf.image.resize_image_with_crop_or_pad, \
                         arguments={'target_height': shape[1], 'target_width': shape[2]})(concat_layer)
    x = layers.concatenate([rs_l, x])
    return x

# modified the keras impl of resnet
# found on keras' github
def ResNet50Decode(img_input, concat_layers):

    concat1, concat2, concat3 = concat_layers[0], concat_layers[1], concat_layers[2]

    x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(img_input)
    x = layers.Conv2DTranspose(64, (7, 7),
                      strides=(2, 2),
                      padding='same',
                      kernel_initializer='he_normal',
                      name='conv1')(x)
    x.set_shape(x._keras_shape)

    x = layers.BatchNormalization(axis=3, name='bn_conv1')(x)
    x = layers.Activation('relu')(x)
    
    x = convT_block(img_input, 3, [64, 64, 256], stage=2, block='a')
    x = convT_block(x, 3, [64, 64, 256], stage=2, block='a_2')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = trim_concat(x, concat1)
    
    x = convT_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = convT_block(x, 3, [128, 128, 512], stage=3, block='a_2')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')
    
    x = trim_concat(x, concat2)

    x = convT_block(x, 3, [256, 256, 1024], stage=4, block='a')
    
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')
    
    x = trim_concat(x, concat3)
    
    x = convT_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')
    
    x = layers.Conv2DTranspose(64, (7, 7),
                  strides=(2, 2),
                  padding='same',
                  kernel_initializer='he_normal',
                  name='conv1')(x)
    x.set_shape(x._keras_shape)
    return x


def ResBottleneck(x, filter_size, depth=6, kernel_size=(3, 3)):
    dilated_layers = []
    for _ in range(depth):
        x = layers.Conv2D(filter_size, kernel_size, activation='relu', padding='same')(x)
        dilated_layers.append(x)
        
    return layers.add(dilated_layers)

    
def get_unet(H, W, lr, loss):
    inputs, concat_layers, encode_out = ResNetEncode((H, W, 3), freeze=True)
    bottleneck = layers.Lambda(ResBottleneck, arguments={'filter_size': 128})(encode_out)
    decode = layers.Lambda(ResNet50Decode, arguments={'concat_layers': concat_layers})(bottleneck)

    # classification layer
    x = layers.Lambda(tf.image.resize_image_with_crop_or_pad, \
                     arguments={'target_height': H, 'target_width': W})(decode)
    classify = layers.Conv2D(3, (1, 1), activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=classify)
    model.compile(optimizer=Adam(lr), loss=loss, metrics=[dice_coef])
    
    return model
    
    
    