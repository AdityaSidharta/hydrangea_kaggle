import numpy as np
from numpy.random import random, permutation
from scipy import misc, ndimage
from scipy.ndimage.interpolation import zoom

import keras
from keras import backend as K
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.models import Sequential, Model
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.layers import Input, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, RMSprop
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.engine.topology import get_source_inputs

from config import class_json_path

class VGG_16():
    def __init__(self, image, weights=None):
        self.vgg_16_mean = np.array([123.68, 116.779, 103.939]).reshape((3,1,1))
        self.model = Sequential()
        self.preprocessed_image = self.vgg_preprocess(image)
        self.input_shape = (3,224,224)
        self.weights = weights
        self.batch_size = 4
        self.default_classes = self.get_classes()
        self.classes = self.default_classes
        self.IDG = ImageDataGenerator()
        self.nb_train = 1000
        self.nb_test = 1000
        print "Constructing VGG 16 Model..."
        self.construct_vgg()

        if self.weights != None:
            print "Load attached VGG 16 weight..."
            self.model.load_weights(self.weights)
    # in VGG16, data for each channel must have a mean of zero.
    # Channels must be in BGR order. It's a reverse from Python default color channel, RGB

    def get_classes(self):
        with open(class_json_path) as f:
            class_dict = json.load(f)
        classes = [class_dict[str(i)][1] for i in range(len(class_dict))]
        return classes

    def construct_vgg(self):
        self.add_LambdaLayer()
        self.add_Convolution_Block(2, 64)
        self.add_Convolution_Block(2, 128)
        self.add_Convolution_Block(3, 256)
        self.add_Convolution_Block(3, 512)
        self.add_Convolution_Block(3, 512)
        self.add_FlattenLayer()
        self.add_FullyConnectedBlock()
        self.add_FullyConnectedBlock()
        self.add_DenseLayer(1000)

    def vgg_preprocess(self, image):
        image = image - self.vgg_16_mean
        image = image[:,::-1]
        return image

    def load_weights_from_path(self, path, filename):
        filepath = get_file(filename, path)
        self.model.load_weights(filepath)
        print "Loading model weights from path..."

    def save_weights(self, path, filename):
        self.model.save_weights(path+filename)
        print "Saving %s model in %s" %(filename, path)

    def change_batch_size(self, batch_size):
        self.batch_size = batch_size

    def pop_layer(self, return_info = False):
        self.model.pop()
        if return_info == True:
            print self.model.summary()

    def load_data_genbatch(self, data, labels, batch_size = self.batch_size, shuffle = True):
        batch = self.IDG.flow(data, labels, batch_size = batch_size, shuffle = shuffle)
        return batch

    def fit_model_gen(self, train_batch, nb_epoch=1, val_batch = None):
        self.model.fit_generator(train_batch, self.nb_train / self.batch_size, epochs=nb_epoch,
                                 validation_data=val_batch)

    def pred_gen(self, test_batch):
        full_pred_array = self.model.predict_generator(test_batch, self.nb_tests / self.batch_size)
        index = np.argmax(full_pred_array, axis=1)
        preds = [full_pred_array[i, index[i]] for i in range(len(index))]
        classes = [self.classes[x] for x in index]
        return np.array(preds), index, classes

    def pred_all(self, test_data):
        full_pred_array = self.model.predict(test_data, batch_size =self.batch_size, verbose =1)
        index = np.argmax(full_pred_array, axis = 1)
        preds = [full_pred_array[i, index[i]] for i in range(len(index))]
        classes = [self.classes[x] for x in index]
        return np.array(preds), index, classes

    def add_Convolution_Block(self, layers, filters):
        for i in range(layers):
            self.model.add(ZeroPadding2D(1,1))
            self.model.add(Convolution2D(filters, kernel_size=(3,3),strides=(3,3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    def add_FullyConnectedBlock(self):
        self.model.add(Dense(4096, activation = 'relu'))
        self.model.add(Dropout(0.5))

    def add_LambdaLayer(self):
        self.model.add(Lambda(self.vgg_preprocess), input_shape=self.input_shape)

    def add_FlattenLayer(self):
        self.model.add(Flatten())

    def add_DenseLayer(self, units):
        self.model.add(Dense(units=units, activation='softmax'))

    def compile_model(self, optimizer = RMSprop(lr=0.1), loss='categorical_crossentropy', metrics=['accuracy']):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

