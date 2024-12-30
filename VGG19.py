import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model

vgg = VGG19(weights='imagenet', include_top=False)

#def perceptual_loss(y_true, y_pred):
layer_name1 = 'block1_conv2'  # block5_conv4
layer_name2 = 'block1_pool'
intermediate_layer_model = Model(inputs=vgg.get_layer(layer_name1).input, outputs=vgg.get_layer(layer_name2).output)
#return keras.losses.MSE(intermediate_layer_model(y_true), intermediate_layer_model(y_pred))
intermediate_layer_model.summary()
