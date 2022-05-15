from keras.layers import Input
from keras.layers import Conv2D,Add
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import Concatenate
from keras.layers import add
from yolo3.utils import compose
from keras.layers import Lambda
from keras.layers import GlobalAveragePooling2D, Reshape, Dense, multiply, Permute
from keras import backend as K

def squeeze_excite_block(input, ratio=16):
    ''' Create a channel-wise squeeze-excite block
    Args:
        input: input tensor
        filters: number of output filters
    Returns: a keras tensor
    References
    -   [Squeeze and Excitation Networks](https://arxiv.org/abs/1709.01507)
    '''
    init = input
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = init._keras_shape[channel_axis]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    if K.image_data_format() == 'channels_first':
        se = Permute((3, 1, 2))(se)

    x = multiply([init, se])
    return x

def Conv_bn_relu(num_filters,
                 kernel_size,
                 batchnorm=True,
                 strides=(1, 1),
                 padding='same'):

    def layer(input_tensor):
        x = Conv2D(num_filters, kernel_size,
                   padding=padding, kernel_initializer='he_normal',
                   strides=strides)(input_tensor)
        if batchnorm:
            x = BatchNormalization()(x)
        x = Activation('relu')(x)
        return x
    return layer

def slice_layer(x, slice_num, channel_input):
    output_list = []
    single_channel = channel_input//slice_num
    for i in range(slice_num):
        #out = x[:, :, :, i*single_channel:(i+1)*single_channel]
        out = Lambda(lambda x: x[:, :, :, i*single_channel:(i+1)*single_channel])(x)
        output_list.append(out)
    return output_list



def res2net_block(num_filters, slice_num, expand):
    def layer(input_tensor):
        # short_cut = input_tensor
        short_cut = Conv_bn_relu(num_filters=expand*num_filters, kernel_size=(1, 1))(input_tensor) # shortcut
        x = Conv_bn_relu(num_filters=expand*num_filters, kernel_size=(1, 1))(input_tensor)
        slice_list = slice_layer(x, slice_num, x.shape[-1])
        side = Conv_bn_relu(num_filters=expand*num_filters//slice_num, kernel_size=(3, 3))(slice_list[1])
        
        z = Concatenate(axis=-1)([slice_list[0], side])   # for one and second stage

        for i in range(2, len(slice_list)):
            y = Conv_bn_relu(num_filters=expand*num_filters//slice_num, kernel_size=(3, 3))(add([side, slice_list[i]]))
            side = y
            z = Concatenate(axis=-1)([z, y])
        z = Conv_bn_relu(num_filters=expand*num_filters, kernel_size=(1, 1))(z)
        # z = squeeze_excite_block(z, ratio=16)
        out = Add()([z, short_cut])
        return out
    return layer



# x = Input((256, 256, 256))
# print(x.shape)
# x_conv_nor = Conv_bn_relu(512, (3, 3))(x)
# print(x_conv_nor.shape)
# out = slice_layer(x_conv_nor, 8, 512)
# print(out)
# print(len(out))
# x = res2net_block(512, 8, 2)(x_conv_nor)
# print(x.shape)