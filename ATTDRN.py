def ATTDRN(input_ECPAB):

    ########################################### " The concatenation block " ####################################################################################
    convDI = Conv2D(filters = 32, kernel_size = 3, strides = 1, padding = 'same', use_bias = True, activation = 'relu')(input_ECPAB)

    convDII = Conv2D(filters = 32, kernel_size = 3, strides = 1, padding = 'same', dilation_rate=(2, 2), use_bias = True, activation = 'relu')(input_ECPAB)

    convDIII = Conv2D(filters = 32, kernel_size = 3, strides = 1, padding = 'same', dilation_rate=(5, 5), use_bias = True, activation = 'relu')(input_ECPAB)

    concatI = tf.keras.ops.concatenate([convDI,convDII,convDIII], axis = -1)

    ########################################### " The channel attention block" (Based on FFA-Net) ####################################################################################

    avg_pool_2d = tf.keras.layers.AveragePooling2D(pool_size=(1, 1),strides=(1, 1), padding='valid')(concatI)  # Incldue average Pool 2D

    convII = Conv2D(filters = 96, kernel_size = 3, strides = 1, padding = 'same', use_bias = True, activation = 'relu')(avg_pool_2d)

    convIII = Conv2D(filters = 1, kernel_size = 3, strides = 1, padding = 'same', use_bias = True, activation = 'relu')(convII)

    convIV = Conv2D(filters = 96, kernel_size = 3, strides = 1, padding = 'same', use_bias = True, activation = 'sigmoid')(convIII)

    AddCAB = keras.layers.Add()([concatI,convIV])  # Adding as skip connection.

    ########################################### " The pixel attention block" (Based on FFA-Net) ####################################################################################

    convV = Conv2D(filters = 96, kernel_size = 3, strides = 1, padding = 'same', use_bias = True, activation = 'relu')(AddCAB)

    convVI = Conv2D(filters = 1, kernel_size = 3, strides = 1, padding = 'same', use_bias = True, activation = 'relu')(convV)

    convVII = Conv2D(filters = 96, kernel_size = 3, strides = 1, padding = 'same', use_bias = True, activation = 'sigmoid')(convVI)

    MultiplyI = keras.layers.Multiply()([convV,convVII])  # Multiply skip connection.

    convVIII = Conv2D(filters = 32, kernel_size = 3, strides = 1, padding = 'same', use_bias = True, activation = 'sigmoid')(MultiplyI)


    AddPAB = keras.layers.Add()([input_ECPAB,convVIII])  # Adding as skip connection.

    ########################################################################################################################################################
    conv_final = Conv2D(filters = 3, kernel_size = 3, strides = 1, padding = 'same', use_bias = True, activation = 'relu')(AddPAB)

    return AddPAB
