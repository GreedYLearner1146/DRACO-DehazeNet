import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model

vgg = VGG19(weights='imagenet', include_top=False)

layer_name1 = 'block1_conv2'  # block5_conv4
layer_name2 = 'block1_pool'
intermediate_layer_model = Model(inputs=vgg.get_layer(layer_name1).input, outputs=vgg.get_layer(layer_name2).output)
