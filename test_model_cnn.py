#####################
#import dependencies#
#####################

import cv2
print (cv2.__version__)

import numpy as np
print (np.__version__) #not the latest version but with the 1.18 we have an error issue with tf

import os
#print("Tensorflow version "+tf.__version__)

import tensorflow as tf
print("Tensorflow version "+tf.__version__)

import keras
print("keras version "+keras.__version__)


####################################################################

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(150, 150,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# the model so far outputs 3D feature maps (height, width, features)

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

from keras.models import load_model
model.load_weights('first_try.h5')
print("Model loaded")
model.summary()

################
# Test picture #
################
from keras.preprocessing import image


repertoire = "../data/test/"
for nom in os.listdir(repertoire):
	img=cv2.imread(repertoire+"/"+nom)
	img=cv2.resize(img,(200,200))
	test_image = image.load_img(repertoire+"/"+nom, target_size=(150,150))

	test_image = image.img_to_array(test_image)
	test_image = np.expand_dims(test_image,axis=0)
	result = model.predict(test_image)

	if result[0][0] >=0.5:
		print('dog')
	else:
		print('car')
	cv2.imshow('Image',img)
	cv2.waitKey(0)

cv2.destroyAllWindows()






