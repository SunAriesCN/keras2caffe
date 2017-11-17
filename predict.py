#!/usr/bin/env python
import numpy as np
from keras_squeezenet import SqueezeNet
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.preprocessing import image
import time
import cv2


model = SqueezeNet()

img = image.load_img('./images/cat.jpeg', target_size=(227, 227))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
#x = preprocess_input(x)

out = model.predict(x)
print out.shape
preds = out.reshape(-1)
print preds[:10]

#print out == out.transpose((0,2,1,3))
