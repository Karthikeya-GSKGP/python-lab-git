
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import matplotlib
matplotlib.use('Agg')


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# For multiple devices (GPUs: 4, 5, 6, 7)
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import matplotlib.pyplot as plt
import numpy as np
import math
import time, sys
import pickle
import timeit
import wandb
import shutil
import pandas as pd
import keras.backend as K
from tensorflow.keras import layers, constraints
import keras
os.environ["WANDB_API_KEY"] = "69d5dd673bb0e0228b9baba64a24ec65d8451b98"
plt.ioff()
from keras.constraints import Constraint

class minmax_constraint(Constraint):
    def __init__(self, min_value=-11.5, max_value=-2.25):
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, w):
        return tf.clip_by_value(w, self.min_value, self.max_value)

def update_progress(progress):
    barLength = 10 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\rPercent: [{0}] {1}% {2}".format( "#"*block + "-"*(barLength-block), progress*100, status)
    sys.stdout.write(text)
    sys.stdout.flush()

def x_Sigma_w_x_T(x, W_Sigma):
    batch_sz = x.shape[0]
    dim = tf.cast(tf.shape(x)[-1], tf.float32)
    x = x/dim
    xx_t = tf.reduce_sum(tf.multiply(x, x), axis=-1,
                         keepdims=True)  # [50, 17, 64]  -> [50, 17, 1] or [50, 64] - > [50, 1]
    # xx_t_e = tf.expand_dims(xx_t,axis=2)
    return tf.multiply(xx_t, W_Sigma)  # [50,17,64] or [50, 64] or [50, 10]

def w_t_Sigma_i_w(w_mu, in_Sigma):  # [64, 64]  , [50, 17, 64] or [64, 10], [50, 64]
    dim = tf.math.sqrt(tf.cast(w_mu.shape[0], tf.float32))
    w_mu = w_mu/dim
    Sigma_1 = tf.matmul(in_Sigma, tf.multiply(w_mu, w_mu))  # [50, 17, 64] or [50, 10]
    return Sigma_1

def tr_Sigma_w_Sigma_in(in_Sigma, W_Sigma):
    dim = tf.cast(W_Sigma.shape[-1], tf.float32)
    Sigma = tf.reduce_sum(in_Sigma, axis=-1, keepdims=True)  # [50,17, 1]
    return tf.multiply(Sigma, W_Sigma)/dim  # [50,17, 64]

def activation_Sigma(gradi, Sigma_in):
    grad1 = tf.multiply(gradi, gradi)  # [50,17,64] or [50, 10]
    dim = tf.cast(grad1.shape[-1], tf.float32)
    return tf.multiply(Sigma_in, grad1)/dim  # [50,17,64] or [50, 10]

def kl_regularizer_conv(mu, logvar):
    k = mu.shape[-1]
    mu = tf.reshape(mu, [-1, k])
    n= mu.shape[0]
    prior_var = 1.
    kl = - tf.math.reduce_mean( (1 + logvar - (tf.math.log(1+tf.math.exp(logvar))/prior_var)) - (tf.math.reduce_sum(tf.square(mu), axis=0)/(n*prior_var)))
    kl = tf.where(tf.math.is_nan(kl), tf.constant(1.0e-5, shape=kl.shape), kl)
    kl = tf.where(tf.math.is_inf(kl), tf.constant(1.0e-5, shape=kl.shape), kl)
    return kl

def kl_regularizer(mu, logvar):
    n= mu.shape[0]
    prior_var = 1.
    kl = - tf.math.reduce_mean( (1 + logvar - (tf.math.log(1+tf.math.exp(logvar))/prior_var)) - (tf.math.reduce_sum(tf.square(mu), axis=0)/(n*prior_var)))
    kl = tf.where(tf.math.is_nan(kl), tf.constant(1.0e-5, shape=kl.shape), kl)
    kl = tf.where(tf.math.is_inf(kl), tf.constant(1.0e-5, shape=kl.shape), kl)
    return kl

class VDP_first_Conv(keras.layers.Layer):
    def __init__(self, kernel_size=5, kernel_num=16, kernel_stride=1, padding="VALID",**kwargs):
        super(VDP_first_Conv, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.kernel_stride = kernel_stride
        self.padding = padding
    def build(self, input_shape):

        ini_sigma = -4.6
        self.w_mu = self.add_weight(shape=(self.kernel_size, self.kernel_size, input_shape[-1], self.kernel_num),
                                    initializer=tf.random_normal_initializer(mean=0.0, stddev=0.05, seed=None),name='w_mu',
                                    trainable=True,
                                    )
        self.w_sigma = self.add_weight(shape=(self.kernel_num,), initializer=tf.constant_initializer(ini_sigma), name='w_sigma',
                                       trainable=True, constraint=minmax_constraint()
                                       )
    def call(self, mu_in):
        batch_size = mu_in.shape[0]
        num_channel = mu_in.shape[-1]
        kl_conv = kl_regularizer_conv(self.w_mu, self.w_sigma)
        w_sigma_2 =   tf.math.log(1+tf.math.exp(self.w_sigma) )
        mu_out = tf.nn.conv2d(mu_in, self.w_mu, strides=[1, self.kernel_stride, self.kernel_stride, 1],
                              padding=self.padding, data_format='NHWC')
        x_train_patches = tf.image.extract_patches(mu_in, sizes=[1, self.kernel_size, self.kernel_size, 1],
                                                   strides=[1, self.kernel_stride, self.kernel_stride, 1],
                                                   rates=[1, 1, 1, 1],
                                                   padding=self.padding)  # shape=[batch_size, image_size, image_size, kernel_size*kernel_size*num_channel]
        x_train_matrix = tf.reshape(x_train_patches, [batch_size, -1, self.kernel_size * self.kernel_size * num_channel])  # shape=[batch_size, image_size*image_size, patch_size*patch_size*num_channel]
        x_dim = tf.cast(tf.shape(x_train_matrix)[-1], tf.float32)
        x_train_matrix = tf.math.reduce_sum(tf.math.square(x_train_matrix)/x_dim, axis=-1) 
        X_XTranspose = tf.ones([1, 1, self.kernel_num]) * tf.expand_dims(x_train_matrix, axis=-1)
        Sigma_out = tf.multiply(w_sigma_2,  X_XTranspose)  
        Sigma_out = tf.reshape(Sigma_out, [batch_size,mu_out.shape[1], mu_out.shape[1] , self.kernel_num])
        return mu_out, Sigma_out, kl_conv 

class VDP_intermediate_Conv(keras.layers.Layer):
    def __init__(self, kernel_size=5, kernel_num=16, kernel_stride=1, padding="VALID",**kwargs):
        super(VDP_intermediate_Conv, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.kernel_stride = kernel_stride
        self.padding = padding
     #   self.sigma_constraint = constraints.MaxNorm(max_value=1.0)
    def build(self, input_shape):
        ini_sigma = -4.6
        self.w_mu = self.add_weight(shape=(self.kernel_size, self.kernel_size, input_shape[-1], self.kernel_num),
                                    initializer=tf.random_normal_initializer(mean=0.0, stddev=0.05, seed=None), name='w_mu',
                                    trainable=True,
                                    )
        self.w_sigma = self.add_weight(shape=(self.kernel_num,),
                                       initializer=tf.constant_initializer(ini_sigma),  # tf.random_uniform_initializer(minval=min_sigma, maxval=ini_sigma,  seed=None),
                                       name='w_sigma', trainable=True, constraint=minmax_constraint()
                                       )
    def call(self, mu_in, Sigma_in): # [batch_size, image_size , image_size, channel]
        batch_size = mu_in.shape[0]
        num_channel = mu_in.shape[-1]  # shape=[batch_size, im_size, im_size, num_channel]
        kl_conv = kl_regularizer_conv(self.w_mu, self.w_sigma)
        w_sigma_2 =   tf.math.log(1+tf.math.exp(self.w_sigma) )
       # w_sigma_2 = tf.clip_by_value(t=w_sigma_2, clip_value_min=tf.constant(1e-6), clip_value_max=tf.constant(1e+6))
        mu_out = tf.nn.conv2d(mu_in, self.w_mu, strides=[1, self.kernel_stride, self.kernel_stride, 1],
                              padding=self.padding, data_format='NHWC')

        diag_sigma_patches = tf.image.extract_patches(Sigma_in, sizes=[1, self.kernel_size, self.kernel_size, 1],
                                                      strides=[1, self.kernel_stride, self.kernel_stride, 1],
                                                      rates=[1, 1, 1, 1], padding=self.padding) # shape=[batch_size, new_im_size, new_im_size, kernel_size*kernel_size*num_channel]

        diag_sigma_g = tf.reshape(diag_sigma_patches, [batch_size, -1, self.kernel_size * self.kernel_size * num_channel]) # shape=[batch_size, new_im_size*new_im_size,   self.kernel_size*self.kernel_size*num_channel ]
        mu_cov_square = tf.reshape(tf.math.multiply(self.w_mu, self.w_mu), [self.kernel_size * self.kernel_size * num_channel, self.kernel_num])  # shape[ kernel_size*kernel_size*num_channel,   kernel_num]
        mu_dim = tf.cast(tf.shape(mu_cov_square)[0], tf.float32)
        mu_wT_sigmags_mu_w = tf.matmul(diag_sigma_g, mu_cov_square/mu_dim)  # shape=[batch_size, new_im_size*new_im_size , kernel_num   ]
        trace = tf.math.reduce_sum(diag_sigma_g, 2, keepdims=True)  # shape=[batch_size,  new_im_size* new_im_size, 1]
        trace = tf.ones([1, 1, self.kernel_num]) * trace  # shape=[batch_size,  new_im_size*new_im_size, kernel_num]
        trace =tf.multiply(w_sigma_2, trace)  # shape=[batch_size, , new_im_size*new_im_size, kernel_num]

        mu_in_patches = tf.reshape(tf.image.extract_patches(mu_in, sizes=[1, self.kernel_size, self.kernel_size, 1],
                                                            strides=[1, self.kernel_stride, self.kernel_stride, 1],
                                                            rates=[1, 1, 1, 1], padding=self.padding),
                                                           [batch_size, -1, self.kernel_size * self.kernel_size * num_channel])# shape=[batch_size, new_im_size*new_im_size, self.kernel_size*self.kernel_size*num_channel]
        mu_gT_mu_g = tf.math.reduce_sum(tf.math.multiply(mu_in_patches, mu_in_patches)/mu_dim, axis=-1)  # shape=[batch_size, new_im_size*new_im_size]
        mu_gT_mu_g1 = tf.ones([1, 1,  self.kernel_num]) * tf.expand_dims(mu_gT_mu_g, axis=-1)     # shape=[batch_size, new_im_size*new_im_size, kernel_num]
        sigmaw_mu_gT_mu_g = tf.multiply(w_sigma_2, mu_gT_mu_g1) /mu_dim  # shape=[batch_size, new_im_size*new_im_size, kernel_num]

        Sigma_out = trace + mu_wT_sigmags_mu_w + sigmaw_mu_gT_mu_g  # # shape=[batch_size, new_im_size*new_im_size, kernel_num]
        Sigma_out = tf.reshape(Sigma_out, [batch_size,mu_out.shape[1], mu_out.shape[1] , self.kernel_num])
        return mu_out, Sigma_out, kl_conv

class VDP_MaxPooling(keras.layers.Layer):
    """VDP_MaxPooling"""
    def __init__(self, pooling_size=2, pooling_stride=2, pooling_pad='SAME',**kwargs):
        super(VDP_MaxPooling, self).__init__(**kwargs)
        self.pooling_size = pooling_size
        self.pooling_stride = pooling_stride
        self.pooling_pad = pooling_pad

    def call(self, mu_in, Sigma_in): # shape=[batch_size,,im_size, im_size, num_channel]
        batch_size = mu_in.shape[0]  # shape=[batch_size, im_size, im_size, num_channel]
        hw_in = mu_in.shape[1]
        num_channel = mu_in.shape[-1]
        mu_out, argmax_out = tf.nn.max_pool_with_argmax(mu_in, ksize=[1, self.pooling_size, self.pooling_size, 1],
                                                        strides=[1, self.pooling_stride, self.pooling_stride, 1],
                                                        padding=self.pooling_pad)  # shape=[batch_zise, new_size,new_size,num_channel]
        hw_out = mu_out.shape[1]
        argmax1 = tf.transpose(argmax_out, [0, 3, 1, 2])
        argmax2 = tf.reshape(argmax1, [batch_size, num_channel,
                                       -1])  # shape=[batch_size, num_channel, new_size*new_size]
        x_index = tf.math.floormod(tf.compat.v1.floor_div(argmax2, tf.constant(num_channel,
                                                                               shape=[batch_size, num_channel,
                                                                                      hw_out * hw_out], dtype='int64')),
                                   tf.constant(hw_in, shape=[batch_size, num_channel, hw_out * hw_out], dtype='int64'))
        aux = tf.compat.v1.floor_div(tf.compat.v1.floor_div(argmax2, tf.constant(num_channel,
                                                                                 shape=[batch_size, num_channel,
                                                                                        hw_out * hw_out],  dtype='int64')),
                                     tf.constant(hw_in, shape=[batch_size, num_channel, hw_out * hw_out], dtype='int64'))
        y_index = tf.math.floormod(aux,  tf.constant(hw_in, shape=[batch_size, num_channel, hw_out * hw_out], dtype='int64'))
        index = tf.multiply(y_index, hw_in) + x_index  # shape=[batch_size, num_channel,new_size*new_size]
        Sigma_in1 = tf.transpose(tf.reshape(Sigma_in, [batch_size, -1, num_channel ]), [0, 2, 1])
        Sigma_out = tf.gather(Sigma_in1, index, batch_dims=2,     axis=-1)  # shape=[batch_size,num_channel,new_size*new_size]
        Sigma_out = tf.reshape(tf.transpose(Sigma_out,    [0, 2, 1]), [batch_size, mu_out.shape[1], mu_out.shape[1],num_channel])  # shape=[batch_size,new_size, new_size, num_channel]
        return mu_out, Sigma_out

# mywork

class LinearFirst(keras.layers.Layer):
    """y = w.x + b"""
    def __init__(self, units,**kwargs):
        super(LinearFirst, self).__init__(**kwargs)
        self.units = units
      #  self.sigma_constraint = constraints.MaxNorm(max_value=1.0)
    def build(self, input_shape):
        ini_sigma = -4.6
        self.w_mu = self.add_weight(shape=(input_shape[-1], self.units),
                                    initializer=tf.random_normal_initializer(mean=0.0, stddev=0.05, seed=None), name='w_mu',
                                    trainable=True)
        self.w_sigma = self.add_weight(shape=(self.units,),
                                       initializer=tf.constant_initializer(ini_sigma), constraint=minmax_constraint(),
                                       name='w_sigma',
                                       trainable=True)
    def call(self, inputs):  # [50,17,64]
        # Mean
        # print(self.w_mu.shape)
        kl_fc = kl_regularizer(self.w_mu, self.w_sigma)
        mu_out = tf.matmul(inputs, self.w_mu)  # + self.b_mu       [50, 17, 64]             # Mean of the output
        # Varinace
        W_Sigma = tf.math.log(1+tf.math.exp(self.w_sigma) ) # [64]                        # Construct W_Sigma from w_sigmas
        #W_Sigma = tf.clip_by_value(t=W_Sigma, clip_value_min=tf.constant(1e-6), clip_value_max=tf.constant(1e+6))
        Sigma_out = x_Sigma_w_x_T(inputs,
                                  W_Sigma)  # [50, 17, 64]            + tf.math.log(1. + tf.math.exp(self.b_sigma)) #tf.linalg.diag(self.b_sigma)
        Sigma_out = tf.where(tf.math.is_nan(Sigma_out), tf.constant(1.0e-5, shape=Sigma_out.shape), Sigma_out)
        Sigma_out = tf.where(tf.math.is_inf(Sigma_out), tf.constant(1.0, shape=Sigma_out.shape), Sigma_out)
        ##Sigma_out = tf.abs(Sigma_out)
        return mu_out, Sigma_out, kl_fc

# # mywork



class LinearNotFirst(keras.layers.Layer):
    """y = w.x + b"""
    def __init__(self, units,**kwargs):
        super(LinearNotFirst, self).__init__(**kwargs)
        self.units = units
      #  self.sigma_constraint = constraints.MaxNorm(max_value=1.0)
    def build(self, input_shape):
        ini_sigma = -4.6
        # min_sigma = -4.5
      #  tau = 0.5  # 1. /input_shape[-1]
        self.w_mu = self.add_weight(shape=(input_shape[-1], self.units),
                                    # [64 , 64] or or [64, 10] or [10, 10]
                                    initializer=tf.random_normal_initializer(mean=0.0, stddev=0.05, seed=None), name='w_mu',
                                    trainable=True, )
        self.w_sigma = self.add_weight(shape=(self.units,),
                                       initializer=tf.constant_initializer(ini_sigma), constraint=minmax_constraint(),
                                       name='w_sigma',
                                       trainable=True, )

    def call(self, mu_in, Sigma_in):  # [50,17,64],  [50,17,64]   or [50, 64] or [50, 10]
        mu_out = tf.matmul(mu_in, self.w_mu)  # + self.b_mu  [50, 17, 64]
        kl_fc = kl_regularizer(self.w_mu, self.w_sigma)
        W_Sigma =  tf.math.log(1+ tf.math.exp(self.w_sigma))
       # W_Sigma = tf.clip_by_value(t=W_Sigma, clip_value_min=tf.constant(1e-6), clip_value_max=tf.constant(1e+6))
        #W_Sigma = tf.math.log(1. + tf.math.exp(self.w_sigma))  # [64]
        Sigma_1 = w_t_Sigma_i_w(self.w_mu, Sigma_in)  # [50,17,64]
        Sigma_2 = x_Sigma_w_x_T(mu_in, W_Sigma)  # [50, 17, 64]
        Sigma_3 = tr_Sigma_w_Sigma_in(Sigma_in, W_Sigma)  # [50, 17, 64]
        Sigma_out = Sigma_1 + Sigma_2 + Sigma_3  # + tf.linalg.diag(tf.math.log(1. + tf.math.exp(self.b_sigma)))  #[50, 17, 64]

        Sigma_out = tf.where(tf.math.is_nan(Sigma_out), tf.constant(1.0e-5, shape=Sigma_out.shape), Sigma_out)
        Sigma_out = tf.where(tf.math.is_inf(Sigma_out), tf.constant(1.0, shape=Sigma_out.shape), Sigma_out)
        #Sigma_out = tf.abs(Sigma_out)


        return mu_out, Sigma_out , kl_fc # mu_out=[50,17,64], Sigma_out = [50,17,64

class mysoftmax(keras.layers.Layer):
    def __init__(self,**kwargs):
        super(mysoftmax, self).__init__(**kwargs)
    def call(self, mu_in, Sigma_in):
        mu_dim = tf.cast(tf.shape(mu_in)[-1], tf.float32)
        mu_out = tf.nn.softmax(mu_in)
        grad = (mu_out - mu_out**2)**2
        Sigma_out = tf.multiply(grad, Sigma_in )/mu_dim

        return mu_out, Sigma_out

class VDP_ReLU(keras.layers.Layer):
    """ReLU"""
    def __init__(self,**kwargs):
        super(VDP_ReLU, self).__init__(**kwargs)

    def call(self, mu_in, Sigma_in):
        mu_out = tf.nn.relu(mu_in)
        with tf.GradientTape() as g:
            g.watch(mu_in)
            out = tf.nn.relu(mu_in)
        gradi = g.gradient(out, mu_in)
        Sigma_out = activation_Sigma(gradi, Sigma_in)
        Sigma_out = tf.where(tf.math.is_nan(Sigma_out), tf.constant(1.0e-5, shape=Sigma_out.shape), Sigma_out)
        Sigma_out = tf.where(tf.math.is_inf(Sigma_out), tf.constant(1.0, shape=Sigma_out.shape), Sigma_out)
        #Sigma_out = tf.abs(Sigma_out)
        return mu_out, Sigma_out

class VDP_GeLU(keras.layers.Layer):
    def __init__(self,**kwargs):
        super(VDP_GeLU, self).__init__(**kwargs)
    def call(self, mu_in, Sigma_in):  # mu_in = [50,17,64], Sigma_in= [50,17,64]
        mu_out = tf.nn.gelu(mu_in)  # [50,17,64]
        with tf.GradientTape() as g:
            g.watch(mu_in)
            out = tf.nn.gelu(mu_in)
        gradi = g.gradient(out, mu_in)  # [50,17,64]
        Sigma_out = activation_Sigma(gradi, Sigma_in)
        return mu_out, Sigma_out  # [50,2,17,64], [50,2,17,64,64]

class Density_prop_DNN(tf.keras.Model):
    def __init__(self, units1, units2, units3, name=None):
        super(Density_prop_DNN, self).__init__()

        self.units1 = units1
        self.units2 = units2
        self.units3 = units3

        self.fc_1 = LinearFirst(self.units1)
        self.fc_2 = LinearNotFirst(self.units2)
        self.fc_3 = LinearNotFirst(self.units3)
        self.relu = VDP_ReLU()
        self.mysoftma = mysoftmax()

    def call(self, inputs, training=True):
        batch_size = inputs.shape[0]
        inputs = tf.reshape(inputs, [batch_size, -1])
        print('inputs', inputs.shape)
        mu1, sigma1, kl1 = self.fc_1(inputs)
        mu1, sigma1 = self.relu(mu1, sigma1)
        mu2, sigma2, kl2 = self.fc_2(mu1, sigma1)
        mu2, sigma2 = self.relu(mu2, sigma2)
        mu3, sigma3, kl3 = self.fc_3(mu2, sigma2)
        # mu3, sigma3 = self.relu(mu3, sigma3)
        print('mu3', mu3.shape)
        print('sigma3', sigma3.shape)
        kl_total = kl1 + kl2 + kl3

        mu_out, Sigma_out = self.mysoftma(mu3, sigma3)

        Sigma_out = tf.where(tf.math.is_nan(Sigma_out), tf.zeros_like(Sigma_out), Sigma_out)
        Sigma_out = tf.where(tf.math.is_inf(Sigma_out), tf.zeros_like(Sigma_out), Sigma_out)

        return mu_out, Sigma_out, kl_total

    # I have changed the test4 to tes5 name for github assignment3
        # I have changed the test4 to tes5 name for github assignment3
wandb.init(project="Test5" ) #, entity="your_wandb_username")

def main_function(image_size=28, units1=200, units2=100, units3=10, epochs=20,  lr=0.0001, lr_end=0.00001,
                  batch_size=50, num_classes=10,kl_factor=0.01,Training = True, Testing= False, Random_noise=False, gaussain_noise_std=0.25):
    PATH = './latest_run/saved_models/VDP_cnn_epoch_{}/'.format(epochs)
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train = tf.expand_dims(x_train, -1)
    x_test = tf.expand_dims(x_test, -1)
    one_hot_y_train = tf.one_hot(y_train.astype(np.float32), depth=num_classes)
    one_hot_y_test = tf.one_hot(y_test.astype(np.float32), depth=num_classes)
    tr_dataset = tf.data.Dataset.from_tensor_slices((x_train, one_hot_y_train)).batch(batch_size)
    val_dataset = tf.data.Dataset.from_tensor_slices((x_test, one_hot_y_test)).batch(batch_size)

    trans_model = Density_prop_DNN(units1=units1, units2=units2, units3=units3)

    num_train_steps = epochs * int(x_train.shape[0] / batch_size)
    #    step = min(step, decay_steps)
    #    ((initial_learning_rate - end_learning_rate) * (1 - step / decay_steps) ^ (power) ) + end_learning_rate
    learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=lr,
                                                                     decay_steps=num_train_steps,
                                                                     end_learning_rate=lr_end, power=1.)
    optimizer = tf.keras.optimizers.Adam(learning_rate= learning_rate_fn)#, clipvalue=1.0)
    #loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    def nll_gaussian(y_test, y_pred_mean, y_pred_sd):
        mu = loss_fn(y_test, y_pred_mean) #y_test - y_pred_mean
        #mu_2 = mu ** 2
        y_pred_sd = y_pred_sd + 1e-4
        s = tf.math.divide_no_nan(1., y_pred_sd)
        #loss = tf.math.reduce_mean(tf.math.reduce_sum(tf.math.multiply(mu, s), axis=-1)) #with scaling with sigma
        loss1 = tf.math.reduce_mean(tf.math.reduce_sum(mu)) #without scaling with sigma
        loss2 = tf.math.reduce_mean(tf.math.reduce_sum(tf.math.log(y_pred_sd), axis=-1))
        loss = tf.math.reduce_mean(tf.math.add(loss1, loss2))
        loss = tf.where(tf.math.is_nan(loss), tf.zeros_like(loss), loss)
        loss = tf.where(tf.math.is_inf(loss), tf.zeros_like(loss), loss)
        return loss
    @tf.function  # Make it fast.
    def train_on_batch(x, y):
        with tf.GradientTape() as tape:
            mu_out, sigma, kl = trans_model(x, training=True)
            trans_model.trainable = True
            #loss_final = loss_fn(y, mu_out)
            loss_final = 0.01*nll_gaussian(y, mu_out, tf.clip_by_value(t=sigma, clip_value_min=tf.constant(1e-8),
                                                                  clip_value_max=tf.constant(1e+2)))

            regularization_loss= kl #+ tf.reduce_mean(tf.norm(tf.clip_by_value(t=sigma, clip_value_min=tf.constant(1e-6), clip_value_max=tf.constant(1e+5)), ord='euclidean', axis=-1))
            loss = 0.5*(loss_final + kl_factor*regularization_loss )#/tf.cast( tf.shape( tf.reshape(y, [-1])), tf.float32)
        gradients = tape.gradient(loss, trans_model.trainable_weights)
        gradients = [(tf.where(tf.math.is_nan(grad), tf.constant(1.0e-5, shape=grad.shape), grad)) for grad in gradients]
        gradients = [(tf.where(tf.math.is_inf(grad), tf.constant(1.0e-5, shape=grad.shape), grad)) for grad in gradients]
        optimizer.apply_gradients(zip(gradients, trans_model.trainable_weights))
        return loss, mu_out, sigma, gradients, regularization_loss, loss_final
    @tf.function
    def validation_on_batch(x, y):
        mu_out, sigma, kl = trans_model(x, training=False)
        trans_model.trainable = False
      #  vloss = loss_fn(y, mu_out)
        vloss = 0.01*nll_gaussian(y, mu_out, tf.clip_by_value(t=sigma, clip_value_min=tf.constant(1e-8),
                                                         clip_value_max=tf.constant(1e+2)))

        regularization_loss= kl  #+ tf.reduce_mean(tf.norm(tf.clip_by_value(t=sigma, clip_value_min=tf.constant(1e-6), clip_value_max=tf.constant(1e+5)), ord='euclidean', axis=-1))
        total_vloss = 0.5*(vloss + kl_factor*regularization_loss)#/tf.cast( tf.shape( tf.reshape(y, [-1])), tf.float32)
        return total_vloss, mu_out, sigma
    @tf.function
    def test_on_batch(x, y, trans_model):
        trans_model.trainable = False
        mu_out, sigma, kl = trans_model(x, training=False)
        return mu_out, sigma
    @tf.function
    def create_adversarial_pattern(input_image, input_label):
          with tf.GradientTape() as tape:
            tape.watch(input_image)
            trans_model.trainable = False
            prediction, sigma, kl = trans_model(input_image, training=False)
            #loss_final = loss_fn(input_label, prediction)
            loss_final = 0.01*nll_gaussian(input_label, prediction,
                                      tf.clip_by_value(t=sigma, clip_value_min=tf.constant(1e-8),   clip_value_max=tf.constant(1e+2)))
            regularization_loss= kl  #+ tf.reduce_mean(tf.norm(tf.clip_by_value(t=sigma, clip_value_min=tf.constant(1e-6), clip_value_max=tf.constant(1e+5)), ord='euclidean', axis=-1))
            loss = 0.5*(loss_final + kl_factor*regularization_loss)

            #loss =  loss_final
          # Get the gradients of the loss w.r.t to the input image.
          gradient = tape.gradient(loss, input_image)
          # Get the sign of the gradients to create the perturbation
          signed_grad = tf.sign(gradient)
          return signed_grad
    if Training:
        if os.path.exists(PATH):
            shutil.rmtree(PATH)
        os.makedirs(PATH)
        train_acc = np.zeros(epochs)
        valid_acc = np.zeros(epochs)
        train_err = np.zeros(epochs)
        valid_error = np.zeros(epochs)
        start = timeit.default_timer()
        for epoch in range(epochs):
            print('Epoch: ', epoch+1, '/' , epochs)
            acc1 = 0
            acc_valid1 = 0
            err1 = 0
            err_valid1 = 0
            tr_no_steps = 0
            va_no_steps = 0
            #-------------Training--------------------
            acc_training = np.zeros(int(x_train.shape[0] / (batch_size)))
            err_training = np.zeros(int(x_train.shape[0] / (batch_size)))
#            if epoch==200:
#               kl_factor=0.0001
            for step, (x, y) in enumerate(tr_dataset):
                update_progress(step/int(x_train.shape[0]/(batch_size)) )
                loss, mu_out, sigma, gradients, regularization_loss, loss_final = train_on_batch(x, y)
                err1+= loss.numpy()
                corr = tf.equal(tf.math.argmax(mu_out, axis=1),tf.math.argmax(y,axis=1))
                accuracy = tf.reduce_mean(tf.cast(corr,tf.float32))
                acc1+=accuracy.numpy()
                if step % 50 == 0:
                    print('\n gradient', np.mean(gradients[0].numpy()))
                    print('\n Matrix Norm', np.mean(sigma))
                    print("\n Step:", step, "Loss:" , float(err1/(tr_no_steps + 1.)))
                    print("Total Training accuracy so far: %.3f" % float(acc1/(tr_no_steps + 1.)))
                tr_no_steps+=1
                wandb.log({"Average Variance value": tf.reduce_mean(sigma).numpy(),
                            "Total Training Loss": loss.numpy() ,
                            "Training Accuracy per minibatch": accuracy.numpy() ,
                            "gradient per minibatch": np.mean(gradients[0]),
                           # 'epoch': epoch,
                            "Regularization_loss": regularization_loss.numpy(),
                            "Log-Likelihood Loss": np.mean(loss_final.numpy())
                   })
            train_acc[epoch] = acc1/tr_no_steps
            train_err[epoch] = err1/tr_no_steps
            print('Training Acc  ', train_acc[epoch])
            print('Training error  ', train_err[epoch])
            #---------------Validation----------------------
            for step, (x, y) in enumerate(val_dataset):
                update_progress(step / int(x_test.shape[0] / (batch_size)) )
                total_vloss, mu_out, sigma   = validation_on_batch(x, y)
                err_valid1+= total_vloss.numpy()
                corr = tf.equal(tf.math.argmax(mu_out, axis=-1),tf.math.argmax(y,axis=-1))
                va_accuracy = tf.reduce_mean(tf.cast(corr,tf.float32))
                acc_valid1+=va_accuracy.numpy()

                if step % 50 == 0:
                    print("Step:", step, "Loss:", float(total_vloss))
                    print("Total validation accuracy so far: %.3f" % va_accuracy)
                va_no_steps+=1
                wandb.log({"Average Variance value (validation Set)": tf.reduce_mean(sigma).numpy(),
                               "Total Validation Loss": total_vloss.numpy() ,
                               "Validation Acuracy per minibatch": va_accuracy.numpy()
                                 })
            valid_acc[epoch] = acc_valid1/va_no_steps
            valid_error[epoch] = err_valid1/va_no_steps
            stop = timeit.default_timer()

            if np.max(valid_acc) == valid_acc[epoch]:
                trans_model.save(PATH+'vdp_cnn_model_best.keras')
                trans_model.save_weights(PATH + 'vdp_trans_model.weights.h5')
            wandb.log({"Average Training Loss":  train_err[epoch],
                       "Average Training Accuracy": train_acc[epoch],
                       "Average Validation Loss": valid_error[epoch],
                       "Average Validation Accuracy": valid_acc[epoch],
                       'epoch': epoch
                      })
            print('Total Training Time: ', stop - start)
            print('Training Acc  ', train_acc[epoch])
            print('Validation Acc  ', valid_acc[epoch])
            print('------------------------------------')
            print('Training error  ', train_err[epoch])
            print('Validation error  ', valid_error[epoch])
        #-----------------End Training--------------------------


        f = open(PATH + 'training_validation_acc_error12.pkl', 'wb')
        pickle.dump([train_acc, valid_acc, train_err, valid_error], f)
        f.close()

        textfile = open(PATH + 'Related_hyperparameters.txt','w')
        textfile.write(' Input Dimension : ' + str(image_size))
        # textfile.write('\n Hidden units : ' + str(mlp_dim))
        textfile.write('\n Number of Classes : ' + str(num_classes))
        textfile.write('\n No of epochs : ' + str(epochs))
        textfile.write('\n Initial Learning rate : ' + str(lr))
        textfile.write('\n Ending Learning rate : ' + str(lr_end))
        textfile.write('\n batch size : ' + str(batch_size))
        textfile.write('\n KL term factor : ' + str(kl_factor))
        textfile.write("\n---------------------------------")
        if Training:
            textfile.write('\n Total run time in sec : ' + str(stop - start))
            if (epochs == 1):
                textfile.write("\n Averaged Training  Accuracy : " + str(train_acc))
                textfile.write("\n Averaged Validation Accuracy : " + str(valid_acc))

                textfile.write("\n Averaged Training  error : " + str(train_err))
                textfile.write("\n Averaged Validation error : " + str(valid_error))
            else:
                textfile.write("\n Averaged Training  Accuracy : " + str(np.mean(train_acc[epoch])))
                textfile.write("\n Averaged Validation Accuracy : " + str(np.mean(valid_acc[epoch])))

                textfile.write("\n Averaged Training  error : " + str(np.mean(train_err[epoch])))
                textfile.write("\n Averaged Validation error : " + str(np.mean(valid_error[epoch])))
        textfile.write("\n---------------------------------")
        textfile.write("\n--------------------------------")
        textfile.close()
    if Testing:
        test_path = 'test_results/'
        if Random_noise:
            print(f'Random Noise: {gaussain_noise_std}')
            test_path = 'test_results_random_noise_{}/'.format(gaussain_noise_std)
        full_test_path = PATH + test_path
        if os.path.exists(full_test_path):
            # Remove the existing test path and its contents
            shutil.rmtree(full_test_path)
        os.makedirs(PATH + test_path)
        #trans_model.load_weights(PATH + 'vdp_trans_model.weights.h5')
        #trans_model = tf.keras.models.load_model(PATH + 'vdp_cnn_model_best.keras')

        test_no_steps = 0
        true_x = np.zeros([int(x_test.shape[0] / (batch_size)), batch_size, image_size, image_size, 1])
        true_y = np.zeros([int(x_test.shape[0] / (batch_size)), batch_size, num_classes])
        mu_out_ = np.zeros([int(x_test.shape[0] / (batch_size)), batch_size, num_classes])
        sigma_ = np.zeros([int(x_test.shape[0] / (batch_size)), batch_size, num_classes])
        acc_test = np.zeros(int(x_test.shape[0] / (batch_size)))
        for step, (x, y) in enumerate(val_dataset):
            update_progress(step / int(x_test.shape[0] / (batch_size)))
            true_x[test_no_steps, :, :, :, :] = x
            true_y[test_no_steps, :, :] = y
            if Random_noise:
                noise = tf.random.normal(shape=[batch_size, image_size, image_size, 1], mean=0.0,
                                         stddev=gaussain_noise_std, dtype=x.dtype)
                x = x + noise
            mu_out, sigma = test_on_batch(x, y, trans_model=trans_model)
            mu_out_[test_no_steps, :, :] = mu_out
            sigma_[test_no_steps, :, :] = sigma
            corr = tf.equal(tf.math.argmax(mu_out, axis=1), tf.math.argmax(y, axis=1))
            accuracy = tf.reduce_mean(tf.cast(corr, tf.float32))
            acc_test[test_no_steps] = accuracy.numpy()
            if step % 100 == 0:
                print("Total running accuracy so far: %.3f" % acc_test[test_no_steps])
            test_no_steps += 1

        test_acc = np.mean(acc_test)
        print('Test accuracy : ', test_acc)

        pf = open(PATH + test_path + 'uncertainty_info.pkl', 'wb')
        pickle.dump([mu_out_, sigma_, true_x, true_y, test_acc], pf)
        pf.close()

        var = np.zeros([int(x_test.shape[0] / (batch_size)), batch_size])
        if Random_noise:
            snr_signal = np.zeros([int(x_test.shape[0] / (batch_size)), batch_size])
            for i in range(int(x_test.shape[0] / (batch_size))):
                for j in range(batch_size):
                    noise = tf.random.normal(shape=[image_size, image_size, 1], mean=0.0, stddev=gaussain_noise_std,
                                             dtype=x.dtype)
                    snr_signal[i, j] = 10 * np.log10(
                        np.sum(np.square(true_x[i, j, :, :, :])) / np.sum(np.square(noise)))

                predicted_out = np.argmax(mu_out_[i, j, :])
                var[i, j] = sigma_[i, j, int(predicted_out)]
            print('SNR', np.mean(snr_signal))

        print('Average Output Variance', np.mean(var))
        textfile = open(PATH + test_path + 'Related_hyperparameters.txt', 'w')
        textfile.write(' Input Dimension : ' + str(image_size))
        textfile.write('\n Number of Classes : ' + str(num_classes))
        textfile.write('\n No of epochs : ' + str(epochs))
        textfile.write('\n Initial Learning rate : ' + str(lr))
        textfile.write('\n Ending Learning rate : ' + str(lr_end))
        textfile.write('\n batch size : ' + str(batch_size))
        textfile.write('\n KL term factor : ' + str(kl_factor))
        textfile.write("\n---------------------------------")
        textfile.write("\n Test Accuracy : " + str(test_acc))
        textfile.write("\n Output Variance: " + str(np.mean(np.abs(var))))
        textfile.write("\n---------------------------------")
        if Random_noise:
            textfile.write('\n Random Noise std: ' + str(gaussain_noise_std))
            textfile.write("\n SNR: " + str(np.mean(snr_signal)))
        textfile.write("\n---------------------------------")
        textfile.close()

if __name__ == '__main__':
    main_function()

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline

import matplotlib.pyplot as plt
import pickle

# Set the number of epochs (replace with the actual number if known)
epochs = epochs  # or use the actual number of epochs you trained on

# Load training and validation accuracy/error data
file_path = f'./latest_run/saved_models/VDP_cnn_epoch_{epochs}/training_validation_acc_error12.pkl'

# Load the data directly (assuming the file exists at the specified path)
with open(file_path, 'rb') as f:
    train_acc, valid_acc, train_err, valid_error = pickle.load(f)

# Plot Training and Validation Accuracy
plt.figure(figsize=(12, 6))
plt.plot(train_acc, label='Training Accuracy', color='blue')
plt.plot(valid_acc, label='Validation Accuracy', color='red')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.grid(True)

# Plot Training and Validation Error
plt.figure(figsize=(12, 6))
plt.plot(train_err, label='Training Error', color='blue')
plt.plot(valid_error, label='Validation Error', color='red')
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.title('Training and Validation Error')
plt.legend()
plt.grid(True)

plt.show()  # Single show() call to display both plots

