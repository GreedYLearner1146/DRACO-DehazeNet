############### Import library package ###################

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import keras
import tensorflow as tf
import tensorflow.keras.backend as K
import os
import cv2
import numpy as np
import glob
from os import listdir
from numpy import asarray
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
import re
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, ReLU, Add, MaxPool2D, UpSampling2D, BatchNormalization, concatenate, Subtract, AveragePooling2D
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, ZeroPadding2D, Add, Activation, Conv2DTranspose, Conv1D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras import initializers
from tensorflow.python.keras.layers import Layer, InputSpec
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, Lambda
from tensorflow.keras import backend as K
from tensorflow.keras.activations import relu
from keras.applications.vgg19 import VGG19
import keras.backend as K
from keras.models import Model
from keras.optimizers import Adam
from keras import layers
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import numpy as np

################# load all images in a directory. #####################
from PIL import Image

def natural_sort_key(s, _nsre=re.compile('([0-9]+)')):
   return[int(text) if text.isdigit() else text.lower() for text in _nsre.split(s   )]


def load_images(path, size = (1024,1024)):
    data_list = list()# enumerate filenames in directory, assume all are images
    for filename in sorted(os.listdir(path),key=natural_sort_key):
      pixels = load_img(path + filename, target_size = size)# Convert to numpy array.
      pixels = img_to_array(pixels).astype('float32')
      pixels = cv2.resize(pixels,(256,256))# Need to resize images first, otherwise RAM will run out of space.
      pixels = pixels/255
      data_list.append(pixels)
    return asarray(data_list)

#################### For the directory in this repository, we use the O-HAZE datasets #################

path1 = 'data path for haze dataset'
path2 = 'data path for clear dataset'

data_haze1 = load_images(path1)
data_clear1 = load_images(path2)

############# Insert codes for storing the images into the train valid and test arrays ################

train_haze = []
train_clear = []

valid_haze = []
valid_clear = []

test_haze = []
test_clear = []

########################### E.g.: For O-HAZE ################################################### 


path1 = '/.../# O-HAZY NTIRE 2018/hazy/'
path2 = '/.../# O-HAZY NTIRE 2018/GT/'


data_haze1 = load_images(path1)
data_clear1 = load_images(path2)

############# Following O-HAZE train-valid-split ################

train_haze = []
train_clear = []

valid_haze = []
valid_clear = []

test_haze = []
test_clear = []

for h in data_haze1[0:35]:
  train_haze.append(h)

for i in data_clear1[0:35]:
  train_clear.append(i)

for j in data_haze1[35:40]:
  valid_haze.append(j)

for k in data_clear1[35:40]:
  valid_clear.append(k)

for m in data_haze1[40:45]:
  test_haze.append(m)

for n in data_clear1[40:45]:
  test_clear.append(n)
