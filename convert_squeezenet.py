#!/usr/bin/env python
from keras_squeezenet import SqueezeNet
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.preprocessing import image
import keras2caffe


#converting

keras_model = SqueezeNet()

keras2caffe.convert(keras_model, 'squ.prototxt', 'squ.caffemodel')

