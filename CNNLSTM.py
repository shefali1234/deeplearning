import os
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras import callbacks
from decimal import *
import dhash
from PIL import Image
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers.wrappers import TimeDistributed
from keras.layers.core import Dense, Dropout, Activation, Flatten, Permute, Reshape 
from keras.layers import Input ,Dense, Dropout, Activation, LSTM
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Reshape,GlobalAveragePooling1D
#from keras.layers import TimeDistributedConvolution2D, TimeDistributedMaxPooling2D, TimeDistributedFlatten
from keras.layers.recurrent import LSTM
timesteps=2

DEV = False
argvs = sys.argv
argc = len(argvs)

if argc > 1 and (argvs[1] == "--development" or argvs[1] == "-d"):
    DEV = True

if DEV:
    epochs = 4
else:
    epochs = 25


train_data_dir = './spoofing/face/train/'
validation_data_dir = './spoofing/face/train/'
img_width, img_height = 320,240
nb_train_samples = 6795
nb_validation_samples = 6795
nb_filters1 = 32
nb_filters2 = 64
conv1_size = 3
conv2_size = 2
pool_size = 2
classes_num = 5
batch_size = 10
lr = 0.0004
cnn = Sequential()
cnn.add(Conv2D(nb_filters1, (conv1_size, conv1_size), padding="same", input_shape=(img_width, img_height, 3)))
#cnn.add(Conv2D(nb_filters1, (conv1_size, conv1_size)))
cnn.add(Activation("relu"))
cnn.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
cnn.add(Flatten())

model = Sequential()
model.add(TimeDistributed(cnn, input_shape=(1,320,240,3)))
#model.add(Flatten())
model.add(LSTM(12,input_shape=(1,320,240,3), return_sequences=True))
model.add(Flatten())
model.add(Dense(5, activation='softmax'))
model.summary()

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)
 test_datagen = ImageDataGenerator(
    rescale=1. / 255)
 train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(320,240),
    batch_size=batch_size,
    class_mode='categorical')
 validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(320,240),
    batch_size=batch_size,
    class_mode='categorical')
model.fit_generator(
    train_generator,
    samples_per_epoch=nb_train_samples,
    epochs=epochs,
    validation_data=validation_generator,
    #callbacks=cbks,
    validation_steps=nb_validation_samples)
