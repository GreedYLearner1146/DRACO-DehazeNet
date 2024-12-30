
input_hazy = keras.layers.Input(shape=(256, 256, 3))  # hazy images.
input_clear = keras.layers.Input(shape=(256, 256, 3))  # clear images.

conv1 = Conv2D(filters = 32, kernel_size = 3, strides = 1, padding = 'same', use_bias = True, activation = 'relu')(input_hazy)

DDIRB1 = DDIRB(conv1)  # DDIRB

conv_intermediateI = Conv2D(filters = 3, kernel_size = 3, strides = 1, padding = 'same', use_bias = True, activation = 'relu')(DDIRB1)

TotalAddI = keras.layers.Add()([conv_intermediateI, input_hazy])  # Adding as skip connection.
anchorII = ReLU(max_value = 1.0)(tf.keras.ops.multiply(TotalAddI,input_hazy) - TotalAddI + 1.0) # AnchorII are dehazed images from DDIRB only.

ECPAB1 = ATTDRN(DDIRB1) # ATTDRN

conv_Final = Conv2D(filters = 3, kernel_size = 3, strides = 1, padding = 'same', use_bias = True, activation = 'relu')(ECPAB1)

TotalAddII = keras.layers.Add()([TotalAddI, conv_Final])  # Adding as skip connection.

# Atmospheric scattering model
anchor = ReLU(max_value = 1.0)(tf.keras.ops.multiply(TotalAddII,input_hazy) - TotalAddII + 1.0) # Anchor are dehazed images.

####################################### Quadruplet Network-based Contrastive Dehazing #############################################################

# Encoder input anchor.
xa = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(anchor)
xa = layers.MaxPooling2D((2, 2), padding="same")(xa)
xa = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(xa)
xa = layers.MaxPooling2D((2, 2), padding="same")(xa)

# Encoder input anchor 2.
xaII = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(anchorII)
xaII = layers.MaxPooling2D((2, 2), padding="same")(xaII)
xaII = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(xaII)
xaII = layers.MaxPooling2D((2, 2), padding="same")(xaII)


# Encoder input positive.
xp = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(input_hazy)
xp = layers.MaxPooling2D((2, 2), padding="same")(xp)
xp = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(xp)
xp = layers.MaxPooling2D((2, 2), padding="same")(xp)

# Encoder input negative.
xn = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(input_clear)
xn = layers.MaxPooling2D((2, 2), padding="same")(xn)
xn = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(xn)
xn = layers.MaxPooling2D((2, 2), padding="same")(xn)

# "encoded" is the encoded representation of the input.
# Encoder flatten anchor.

encoding_dim = 64  # Embedding dimension is 64.

encoded_anchor = layers.Dense(encoding_dim, activation='relu')(xa)
encoded_anchorII = layers.Dense(encoding_dim, activation='relu')(xaII)

# Encoder flatten positive.
encoded_pos = layers.Dense(encoding_dim, activation='relu')(xp)

# Encoder flatten negative.
encoded_neg = layers.Dense(encoding_dim, activation='relu')(xn)

embedding_anchor = encoded_anchor  # Embedding anchor.
embedding_anchorII = encoded_anchorII  # Embedding anchor.
embedding_positive = encoded_pos    # Embedding positive.
embedding_negative = encoded_neg      # Embedding negative.


################################ Model Summary ##############################################
DPE_Net_contrastive = Model([input_hazy,input_clear], [anchor,embedding_anchor,embedding_anchorII, embedding_positive,embedding_negative])
DPE_Net_contrastive.summary()
