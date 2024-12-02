'''
MLCommons
group: TinyMLPerf (https://github.com/mlcommons/tiny)

image classification on cifar10

keras_model.py: CIFAR10_ResNetv1 from eembc
'''

import numpy as np
import os

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Activation, Flatten, BatchNormalization
from tensorflow.keras.layers import Conv2D, AveragePooling2D, MaxPooling2D
from tensorflow.keras.regularizers import l2

@tf.custom_gradient
def my_sign(W):
  # Forward computation
  f = tf.math.sign(W)
  # Backward computation
  def grad(dy):
    return dy
  return f, grad

# Custom convolution layer
class quant_conv2D(tf.keras.layers.Layer):
  def __init__(self, filters, filter_size,strides,**kwargs):
    super(quant_conv2D, self).__init__(**kwargs)
    self.num_filters=filters
    self.filter_size=filter_size
    self.strides = strides
  def build(self, input_shape):
    self.kernel=self.add_weight("kernel",shape=[self.filter_size,self.filter_size,input_shape[3],self.num_filters])   #W_t
  def call(self,input):
    return tf.nn.conv2d(input, my_sign(self.kernel), strides=[1, self.strides, self.strides, 1], padding='SAME')
  def get_config(self):
    config = super(quant_conv2D, self).get_config()
    config.update({
        'filters': self.num_filters,
        'filter_size': self.filter_size,
        'strides':self.strides
    })
    return config

# Custom activation RELU
#TODO
class quant_activation(tf.keras.layers.Layer):
    def __init__(self):
        super(quant_activation,self).__init__()
    def call(self, input):
        return my_sign(input)

# Custom Dense
class quant_dense(tf.keras.layers.Layer):
    def __init__(self,units,**kwargs):
        super(quant_dense,self).__init__(**kwargs)
        self.units = units
    def build(self, input_shape):
        # Add weights
        self.kernel = self.add_weight(name = 'kernel',
                                      shape = [input_shape[-1],self.units],
                                      initializer = 'random_normal',
                                      trainable = True)
    def call(self,inputs):
        return tf.matmul(inputs, my_sign(self.kernel))
    def get_config(self):
        config = super(quant_dense, self).get_config()
        config.update({
            'units': self.units
        })
        return config
def get_model_name():
    if os.path.exists("trained_models/trainedResnet.h5"):
        return "trainedResnet"
    else:
        return "pretrainedResnet"

def get_quant_model_name():
    if os.path.exists("trained_models/trainedResnet.h5"):
        return "trainedResnet"
    else:
        return "pretrainedResnet"
def resnet_v1_eembc():
    # Resnet parameters
    input_shape=[32,32,3] # default size for cifar10
    num_classes=10 # default class number for cifar10
    num_filters = 16 # this should be 64 for an official resnet model

    # Input layer, change kernel size to 7x7 and strides to 2 for an official resnet
    inputs = Input(shape=input_shape)
    # Insert quantization for inputs and weights
    x = quant_conv2D(num_filters,
            filter_size = 3,
            strides = 1
            )(inputs)

   # x = Conv2D(num_filters,
   #                kernel_size=3,
   #                strides=1,
   #                padding='same',
   #                kernel_initializer='he_normal',
   #                kernel_regularizer=l2(1e-4))(inputs) #inputs

    x = BatchNormalization()(x)

    # Insert quantization for Inputs
    #TODO
    x = Activation('relu')(x)
    # First stack

    # Weight layers
    y = quant_conv2D(num_filters,
            filter_size = 3,
            strides = 1
            )(x)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    #y = quant_conv2D(num_filters,3)(y)
    y = quant_conv2D(num_filters,
            filter_size = 3,
            strides = 1
            )(y)
    y = BatchNormalization()(y)

    # Overall residual, connect weight layer and identity paths
    x = tf.keras.layers.add([x, y])
    x = Activation('relu')(x)
    # Second stack
    # Weight layers
    num_filters = 32 # Filters need to be double for each stack
   # y = quant_conv2D(num_filters,3)(x)
    y = quant_conv2D(num_filters,
            filter_size = 3,
            strides = 2
            )(x)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    #y = quant_conv2D(num_filters,3)(y)
    y = quant_conv2D(num_filters,
            filter_size = 3,
            strides = 1
            )(y)
    y = BatchNormalization()(y)

    # Adjust for change in dimension due to stride in identity
    #x = quant_conv2D(num_filters,1)(x)
    x = quant_conv2D(num_filters,
            filter_size = 1,
            strides = 2
            )(x)

    # Overall residual, connect weight layer and identity paths
    x = tf.keras.layers.add([x, y])
    x = Activation('relu')(x)
    # Third stack

    # Weight layers
    num_filters = 64
    #y = quant_conv2D(num_filters,3)(x)
    y = quant_conv2D(num_filters,
            filter_size = 3,
            strides = 2
            )(x)

    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    #y = quant_conv2D(num_filters,3)(y)
    y = quant_conv2D(num_filters,
            filter_size = 3,
            strides = 1
            )(y)

    y = BatchNormalization()(y)

    # Adjust for change in dimension due to stride in identity
    #x = quant_conv2D(num_filters,3)(x)
    x = quant_conv2D(num_filters,
            filter_size = 1,
            strides = 2
            )(x)

    # Overall residual, connect weight layer and identity paths
    x = tf.keras.layers.add([x, y])
    x = Activation('relu')(x)
    
    # Fourth stack.
    # Weight layers
    num_filters = 128
    #y = quant_conv2D(num_filters,3)(x)
    y = quant_conv2D(num_filters,
            filter_size = 3,
            strides = 2
            )(x)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    #y = quant_conv2D(num_filters,3)(y)
    y = quant_conv2D(num_filters,
            filter_size = 3,
            strides = 1
            )(y)
    y = BatchNormalization()(y)
    # Adjust for change in dimension due to stride in identity
    x = quant_conv2D(num_filters,
            filter_size = 1,
            strides = 2
            )(x)
    # Overall residual, connect weight layer and identity paths
    x = tf.keras.layers.add([x, y])
    x = Activation('relu')(x)
    # Final classification layer.
    # Do not need to quantize Average pooling layer
    pool_size = int(np.amin(x.shape[1:3]))
    x = AveragePooling2D(pool_size=pool_size)(x)

    # Do not need to quantize flatten layer
    y = Flatten()(x)

    # Add fully connected layer
    #outputs = Dense(num_classes,
    #                 activation='softmax',
    #                 kernel_initializer='he_normal')(y)
    outputs = quant_dense(num_classes)(y)
    outputs = Activation('softmax')(outputs)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)

    return model
