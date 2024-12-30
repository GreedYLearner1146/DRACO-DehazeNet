import tensorflow.keras as keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, SeparableConv2D, DepthwiseConv2D, Add
import tensorflow as tf

#################### Squeeze-and-excite block. #######################################

def squeeze_excite_block(filters,input):
    se = tf.keras.layers.GlobalAveragePooling2D()(input)
    se = tf.keras.layers.Reshape((1, filters))(se)
    se = tf.keras.layers.Dense(filters//16, activation='relu')(se)
    se = tf.keras.layers.Dense(filters, activation='sigmoid')(se)
    se = tf.keras.layers.multiply([input, se])
    return se


########## DDIRB. (1 module: Conv2D -> Depthwise Conv -> SE -> Conv2D) #############

def DDIRB(input_DDIRB):

    ###### Dilation rate = 1. ######

    convI = Conv2D(filters = 32, kernel_size = 1, strides = 1, padding = 'same', dilation_rate = (1,1), use_bias = True, activation = 'relu')(input_DDIRB)

    DepthconvI = DepthwiseConv2D(kernel_size = 3, strides = 1, padding = 'same', use_bias = True, activation = 'relu')(convI)

    SEI = squeeze_excite_block(32,DepthconvI)

    convII = Conv2D(filters = 32, kernel_size = 1, strides = 1, padding = 'same', dilation_rate = (1,1), use_bias = True, activation = 'relu')(SEI)

    # Add layers.

    Add1 = keras.layers.Add()([input_DDIRB,convII])

    ##### Dilation rate = 2. #####

    convIII = Conv2D(filters = 32, kernel_size = 1, strides = 1, padding = 'same', dilation_rate = (2,2), use_bias = True, activation = 'relu')(Add1)

    DepthconvII = DepthwiseConv2D(kernel_size = 3, strides = 1, padding = 'same', use_bias = True, activation = 'relu')(convIII)

    SEII = squeeze_excite_block(32,DepthconvII)

    convIV = Conv2D(filters = 32, kernel_size = 1, strides = 1, padding = 'same', dilation_rate = (2,2), use_bias = True, activation = 'relu')(SEII)

    # Add layers.

    Add2 = keras.layers.Add()([Add1,convIV])

    ##### Dilation rate = 5. #####

    convV = Conv2D(filters = 32, kernel_size = 1, strides = 1, padding = 'same', dilation_rate = (5,5), use_bias = True, activation = 'relu')(Add2)

    DepthconvIII = DepthwiseConv2D(kernel_size = 3, strides = 1, padding = 'same', use_bias = True, activation = 'relu')(convV)

    SEIII = squeeze_excite_block(32,DepthconvIII)

    convVI = Conv2D(filters = 32, kernel_size = 1, strides = 1, padding = 'same', dilation_rate = (5,5), use_bias = True, activation = 'relu')(SEIII)

    # Add layers.

    Add3 = keras.layers.Add()([Add2,convVI])
    return Add3
