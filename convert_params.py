import math
import numpy as np
import keras
#from qkeras.utils import fold_batch_norm
import tensorflow as tf
from tensorflow.keras.models import load_model
from keras_model_quant_8bits import quant_conv2D,quant_activation, quant_dense
custom_objects = {'quant_conv2D':  quant_conv2D, 'quant_activation':quant_activation, 'quant_dense':quant_dense}

model = load_model('trained_models/trainedResnet_quant_8bits.h5', custom_objects = custom_objects)
#model = fold_batch_norm(model)
model.summary()

file = open("binary_file.bin", "wb")
#file.write("#This file contain parameter of model\n")

N_QuantBits = 8;
epsilon = 1e-6;

# Quantization function
def quantize(W):
  # Forward computation
  quantized_range = 2**(N_QuantBits-1)
  f = tf.round(W*quantized_range)/quantized_range
  f = int(quantized_range*(f))
  return f

def batch_fold(mu,var,gamma, beta, epsilon, weights, biases):
    
    param1 = gamma*math.sqrt(var*var+epsilon)
    param2 = -param1*mu + beta
    
    weights_new = weights*param1
    biases_new = biases*param1 + param2
    return weights_new,biases_new

class my_conv:
    def __init__(self, var1, var2):
        self.w = var1
        self.b = var2
    def set_params(self, var1, var2):
        self.w = var1
        self.b = var2
    def batch_fold(self,mu,var,gamma, beta, epsilon):
    
        param1 = gamma*np.sqrt(var*var+epsilon)
        param2 = -param1*mu + beta
        p = self.w
        weights_new = self.w*param1
        biases_new = self.b*param1 + param2
        self.set_params(weights_new,biases_new)
        #return weights_new,biases_new
    def delete_params(self):
        self.w = None
        self.b = None

quan_conv1 = my_conv(None, None)   # Using for batch_normalization folding
quan_conv2 = my_conv(None, None)
previous_layer = None
#print(quan_conv1.w, quan_conv1.b)

for layer in model.layers: 
    #print(layer.name)
    #print(previous_layer)
    
    # Save parameters for batch normalization
    if isinstance(layer, keras.layers.BatchNormalization):
        
       # Get the gamma, beta, moving_mean, and moving_variance parameters of the layer
        gamma, beta, moving_mean, moving_variance = layer.get_weights()
        #print("batch norm",gamma.shape)
       # Batch folding
        quan_conv1.batch_fold(moving_mean,moving_variance,gamma, beta,epsilon)
        
        if (quan_conv2.w is None) and (quan_conv2.b is None): 
            # apply quantization for all elements and convert to integer
            vec_quantize = np.vectorize(quantize)    # function for array input
            kernel_arr =  vec_quantize(quan_conv1.w)
            kernel_arr =  kernel_arr.astype("int"+"{}".format(N_QuantBits))
            
            bias_arr = vec_quantize(quan_conv1.b)
            bias_arr =  bias_arr.astype("int"+"{}".format(N_QuantBits))
            
            # Convert to binary
            vec_bin = np.vectorize(lambda x: np.binary_repr(x, width=N_QuantBits))
            kernel_bin = vec_bin(kernel_arr)
            bias_bin = vec_bin(bias_arr)
            # Save to file
            kernel_bin.tofile(file)
            bias_bin.tofile(file)
        else:
            # Save simultaneous 2 quant_conv to binary file
            
            # The first one
            # apply quantization for all elements and convert to integer
            vec_quantize = np.vectorize(quantize)    # function for array input
            kernel_arr =  vec_quantize(quan_conv1.w)
            kernel_arr =  kernel_arr.astype("int"+"{}".format(N_QuantBits))
            
            bias_arr = vec_quantize(quan_conv1.b)
            bias_arr =  bias_arr.astype("int"+"{}".format(N_QuantBits))
            
            # Convert to binary
            vec_bin = np.vectorize(lambda x: np.binary_repr(x, width=N_QuantBits))
            kernel_bin = vec_bin(kernel_arr)
            bias_bin = vec_bin(bias_arr)
            # Save to file
            kernel_bin.tofile(file)
            bias_bin.tofile(file)
            
            # The second one
            # apply quantization for all elements and convert to integer
            vec_quantize = np.vectorize(quantize)    # function for array input
            kernel_arr =  vec_quantize(quan_conv2.w)
            kernel_arr =  kernel_arr.astype("int"+"{}".format(N_QuantBits))
            
            bias_arr = vec_quantize(quan_conv2.b)
            bias_arr =  bias_arr.astype("int"+"{}".format(N_QuantBits))
            
            # Convert to binary
            vec_bin = np.vectorize(lambda x: np.binary_repr(x, width=N_QuantBits))
            kernel_bin = vec_bin(kernel_arr)
            bias_bin = vec_bin(bias_arr)
            # Save to file
            kernel_bin.tofile(file)
            bias_bin.tofile(file)
        # Delete parameters of 2 object quan_conv1 and quan_conv2 (for saving new parameters)
        quan_conv1.delete_params()
        quan_conv2.delete_params()
        continue
    #Save parameters for quant_conv2D
    if 'quant_conv2d' in layer.name:    
        
        kernel, bias = layer.get_weights() 
        # reshape kernel and bias to (,num_numfilters)
        kernel_arr = kernel.reshape((-1,kernel.shape[-1]))
        bias_arr = bias
        #print(kernel_arr.shape)
        
        # Check if there is exist 2 continuous layer
        if quan_conv1.w is None and quan_conv1.b is None:
            quan_conv1.set_params(kernel_arr,bias_arr)
        else:
            quan_conv2.set_params(kernel_arr,bias_arr)
        continue
        
    # Save parameters for layers
    if layer.get_weights():
        # Get the weights and biases of the first layer
        if hasattr(layer, 'bias') and layer.bias is not None:    # Check this line!!!
            kernel, bias = layer.get_weights()
            # Reshape parameters
            kernel_arr = kernel.reshape((-1,kernel.shape[-1]))
            bias_arr = bias
            # apply quantization for all elements and convert to integer
            vec_quantize = np.vectorize(quantize)    # function for array input
            kernel_arr =  vec_quantize(kernel_arr)
            kernel_arr =  kernel_arr.astype("int"+"{}".format(N_QuantBits))
            
            bias_arr = vec_quantize(bias_arr)
            bias_arr =  bias_arr.astype("int"+"{}".format(N_QuantBits))
            
            # Convert to binary
            vec_bin = np.vectorize(lambda x: np.binary_repr(x, width=N_QuantBits))
            kernel_bin = vec_bin(kernel_arr)
            bias_bin = vec_bin(bias_arr)
            # Save to file
            kernel_bin.tofile(file)
            bias_bin.tofile(file)  
        else:
            kernel = layer.get_weights()
            # Reshape parameters
            kernel_arr = kernel.reshape((-1,kernel.shape[-1]))
            # apply quantization for all elements and convert to integer
            vec_quantize = np.vectorize(quantize)    # function for array input
            kernel_arr =  vec_quantize(kernel_arr)
            kernel_arr =  kernel_arr.astype("int"+"{}".format(N_QuantBits))
    
            # Convert to binary
            vec_bin = np.vectorize(lambda x: np.binary_repr(x, width=N_QuantBits))
            kernel_bin = vec_bin(kernel_arr)
           
            # Save to file
            kernel_bin.tofile(file)
            
    previous_layer = layer.name
# Close the file
file.close()
print("Finish writing file !!!")

