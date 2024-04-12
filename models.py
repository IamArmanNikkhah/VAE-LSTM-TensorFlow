from base import BaseModel
from keras.layers import Conv2D
from keras.layers import Flatten
from tensorflow.keras import layers

import os
import numpy as np
import matplotlib.pylab as plt
from matplotlib.pyplot import savefig
import tensorflow as tf
import tensorflow_probability as tfp


tfd = tfp.distributions



################################ DATASET ##############################

class DatasetBuilder:
  def __init__(self, config):
    self.config = config
    self.seed = tf.keras.Input(shape=(), dtype=tf.int64, name='seed')

  def create_dataset(self, data):
    # Create a source dataset from the input data
    dataset = tf.data.Dataset.from_tensor_slices(data)
    
    # Apply dataset transformations to preprocess the data
    dataset = dataset.shuffle(buffer_size=60000, seed=self.seed)
    dataset = dataset.repeat(8000)
    dataset = dataset.batch(self.config['batch_size'], drop_remainder=True)
    
    return dataset


############################### Sampling ###############################

class Sampling(layers.Layer):
  """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

  def call(self, inputs):
    z_mean, z_log_var = inputs
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon



################################# Encoder ###############################
  
class Encoder(layers.Layer):
  
  def __init__(self, config, name='encoder', **kwargs):
    super(Encoder, self).__init__(name=name, **kwargs) 

    self.config = config

    self.input_dims = self.config['l_win'] * self.config['n_channel']
    self.encoder    = self.build_encoder()
    self.sampling   = Sampling()


  
  def build_encoder(self):
    
    init         = tf.keras.initializers.GlorotUniform()
    inputs       = tf.keras.Input(shape=(self.config['l_win'], self.config['n_channel']))
  
      
    ##########  24 Encoder ###########

    if self.config['l_win'] == 24:
        
          
        
        self.conv_1 = tf.keras.layers.Conv2D(filters=self.config['num_hidden_units'] // 16,
                                        kernel_size=(3, self.config['n_channel']),
                                        strides=(2, 1),
                                        padding='same',
                                        activation='leaky_relu',
                                        kernel_initializer=init)

        
        
        self.conv_2 = tf.keras.layers.Conv2D(filters=self.config['num_hidden_units'] // 8,
                                        kernel_size=(3, self.config['n_channel']),
                                        strides=(2, 1),
                                        padding='same',
                                        activation='leaky_relu',
                                        kernel_initializer=init)
        
        
        self.conv_3 = tf.keras.layers.Conv2D(filters=self.config['num_hidden_units'] // 4,
                                        kernel_size=(3, self.config['n_channel']),
                                        strides=(2, 1),
                                        padding='same',
                                        activation='leaky_relu',
                                        kernel_initializer=init)
        
        
        self.conv_4 = tf.keras.layers.Conv2D(filters=self.config['num_hidden_units'],
                                        kernel_size=(4, self.config['n_channel']),
                                        strides=1,
                                        padding='valid',
                                        activation='leaky_relu',
                                        kernel_initializer=init)
        
      
    ###########  48 Encoder ###########

    elif self.config['l_win'] == 48:
        
        self.conv_1 = tf.keras.layers.Conv2D(filters=self.config['num_hidden_units'] // 16,
                                        kernel_size=(3, self.config['n_channel']),
                                        strides=(2, 1),
                                        padding='same',
                                        activation='leaky_relu',
                                        kernel_initializer=init)
    

        self.conv_2 = tf.keras.layers.Conv2D(filters=self.config['num_hidden_units'] // 8,
                                        kernel_size=(3, self.config['n_channel']),
                                        strides=(2, 1),
                                        padding='same',
                                        activation='leaky_relu',
                                        kernel_initializer=init)
        

        self.conv_3 = tf.keras.layers.Conv2D(filters=self.config['num_hidden_units'] // 4,
                                        kernel_size=(3, self.config['n_channel']),
                                        strides=(2, 1),
                                        padding='same',
                                        activation='leaky_relu',
                                        kernel_initializer=init)
        

        self.conv_4 = tf.keras.layers.Conv2D(filters=self.config['num_hidden_units'],
                                        kernel_size=(6, self.config['n_channel']),
                                        strides=1,
                                        padding='valid',
                                        activation='leaky_relu',
                                        kernel_initializer=init)
        
      
 
    #############  144 Encoder #############
        
    elif self.config['l_win'] == 144:
        
        self.conv_1 = tf.keras.layers.Conv2D(filters=self.config['num_hidden_units'] // 16,
                                        kernel_size=(3, self.config['n_channel']),
                                        strides=(4, 1),
                                        padding='same',
                                        activation='leaky_relu',
                                        kernel_initializer=init)
        
        self.conv_2 = tf.keras.layers.Conv2D(filters=self.config['num_hidden_units'] // 8,
                                        kernel_size=(3, self.config['n_channel']),
                                        strides=(4, 1),
                                        padding='same',
                                        activation='leaky_relu',
                                        kernel_initializer=init)
        
        
        self.conv_3 = tf.keras.layers.Conv2D(filters=self.config['num_hidden_units'] // 4,
                                        kernel_size=(3, self.config['n_channel']),
                                        strides=(3, 1),
                                        padding='same',
                                        activation='leaky_relu',
                                        kernel_initializer=init)
        

        self.conv_4 = tf.keras.layers.Conv2D(filters=self.config['num_hidden_units'],
                                        kernel_size=(3, self.config['n_channel']),
                                        strides=1,
                                        padding='valid',
                                        activation='leaky_relu',
                                        kernel_initializer=init)
        

    
    
    ########  CODE MEAN + STD  #########
    
    
    self.flatten = tf.keras.layers.Flatten()
      
      
    self.dense = tf.keras.layers.Dense(units=self.config['code_size'] * 4,
                                             activation=tf.nn.leaky_relu,
                                             kernel_initializer=init)

    self.dense_mean = tf.keras.layers.Dense(units=self.config['code_size'],
                                             activation=None,
                                             kernel_initializer=init,
                                             name='code_mean')
                                             
    self.dense_std = tf.keras.layers.Dense(units=self.config['code_size'],
                                                activation=tf.nn.relu,
                                                kernel_initializer=init,
                                                name='code_std_dev')
    
  

def call(self, inputs):

  if self.config['l_win'] == 24:
    paddings = tf.constant([[0, 0], [4, 4], [0, 0], [0, 0]])  
    inputs   = tf.pad(input, paddings, "SYMMETRIC")
  
  x = self.conv_1(inputs)
  x = self.conv_2(x)
  x = self.conv_3(x)
  x = self.conv_4(x)

  x = tf.keras.backend.flatten(x)
  x = self.flatten(x)

  x = self.dense(x)

  z_mean = self.dense_mean(x)
  
  z_std  = self.dense_std(x)
  z_std  = z_std + 1e-2

  z = self.sampling((z_mean, z_std))

  return z_mean, z_std, z





################################## DECODER ##############################


class Decoder(layers.Layer):
  def __init__(self, config, name='encoder', **kwargs):
    super(Decoder, self).__init__(name=name, **kwargs)

    self.config = config
    
    self.window_size  = self.config['l_win']
    self.num_channels = self.config['n_channels']
    self.model        = self.build_decoder()

  
  def build_decoder(self):
        
        
        if self.window_size == 24:
            # Architecture for window size 24
            inputs = layers.Input(shape=(None,))
            x = layers.Dense(4 * 4 * 128, activation='relu')(inputs)
            x = layers.Reshape((4, 4, 128))(x)
            x = layers.Conv2DTranspose(128, 3, strides=2, padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
            x = layers.Dropout(0.3)(x)
            x = layers.Conv2DTranspose(64, 3, strides=3, padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
            x = layers.Dropout(0.3)(x)
            outputs = layers.Conv2DTranspose(self.num_channels, 3, activation='sigmoid', padding='same')(x)
            return tf.keras.Model(inputs=inputs, outputs=outputs)
        
        
        elif self.window_size == 48:
            # Architecture for window size 48
            inputs = layers.Input(shape=(None,))
            x = layers.Dense(4 * 4 * 256, activation='relu')(inputs)
            x = layers.Reshape((4, 4, 256))(x)
            x = layers.Conv2DTranspose(256, 3, strides=2, padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
            x = layers.Dropout(0.3)(x)
            x = layers.Conv2DTranspose(128, 3, strides=2, padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
            x = layers.Dropout(0.3)(x)
            x = layers.Conv2DTranspose(64, 5, strides=2, padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
            x = layers.Dropout(0.3)(x)
            outputs = layers.Conv2DTranspose(self.num_channels, 5, activation='sigmoid', padding='same')(x)
            return tf.keras.Model(inputs=inputs, outputs=outputs)
        
        
        
        elif self.window_size == 144:
            # Architecture for window size 144
            inputs = layers.Input(shape=(None,))
            x = layers.Dense(6 * 6 * 512, activation='relu')(inputs)
            x = layers.Reshape((6, 6, 512))(x)
            x = layers.Conv2DTranspose(512, 3, strides=2, padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
            x = layers.Dropout(0.3)(x)
            x = layers.Conv2DTranspose(256, 3, strides=2, padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
            x = layers.Dropout(0.3)(x)
            x = layers.Conv2DTranspose(128, 5, strides=2, padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
            x = layers.Dropout(0.3)(x)
            x = layers.Conv2DTranspose(64, 5, strides=2, padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
            x = layers.Dropout(0.3)(x)
            outputs = layers.Conv2DTranspose(self.num_channels, 7, activation='sigmoid', padding='same')(x)
            return tf.keras.Model(inputs=inputs, outputs=outputs)
        

        else:
            raise ValueError(f"Unsupported window size: {self.window_size}")
        
  
  def call(self, inputs):
        # Process the inputs through the decoder model
        return self.model(inputs)  




################################## VAE MODEL #############################

class VAEmodel(tf.keras.Model):
  def __init__(self, config):
    super().__init__(config)
    
    self.config = config
    
    self.input_dims = self.config['l_win'] * self.config['n_channel']

    # Define the encoder, decoder and sigma2 as separate callable methods
    self.encoder = self.build_encoder()
    self.decoder = self.build_decoder()
    self.sigma2 = self.build_sigma2()

    
    self.init_checkpoint()
    self.lr        = tf.Variable(self.config['learning_rate_vae'], trainable=False)
    self.optimizer = tf.optimizers.Adam(learning_rate=self.lr, beta_1=0.9, beta_2=0.95)


  
  #################### ENCODER ##################
  
  def build_encoder(self):
    
    init = tf.keras.initializers.GlorotUniform()
    
    inputs = tf.keras.Input(shape=(self.config['l_win'], self.config['n_channel']))
    input_tensor = tf.expand_dims(inputs, -1)
      
      
      
    ##########  24 Encoder ###########

    if self.config['l_win'] == 24:
        paddings = tf.constant([[0, 0], [4, 4], [0, 0], [0, 0]])  # Step 3
        input_tensor_padded = tf.pad(input_tensor, paddings, "SYMMETRIC")  # Step 3
        conv_1 = tf.keras.layers.Conv2D(filters=self.config['num_hidden_units'] // 16,
                                        kernel_size=(3, self.config['n_channel']),
                                        strides=(2, 1),
                                        padding='same',
                                        activation='leaky_relu',
                                        kernel_initializer=init)(input_tensor_padded)  # Step 4
        print("conv_1: {}".format(conv_1))
        conv_2 = tf.keras.layers.Conv2D(filters=self.config['num_hidden_units'] // 8,
                                        kernel_size=(3, self.config['n_channel']),
                                        strides=(2, 1),
                                        padding='same',
                                        activation='leaky_relu',
                                        kernel_initializer=init)(conv_1)  # Step 4
        print("conv_2: {}".format(conv_2))
        conv_3 = tf.keras.layers.Conv2D(filters=self.config['num_hidden_units'] // 4,
                                        kernel_size=(3, self.config['n_channel']),
                                        strides=(2, 1),
                                        padding='same',
                                        activation='leaky_relu',
                                        kernel_initializer=init)(conv_2)  # Step 4
        print("conv_3: {}".format(conv_3))
        conv_4 = tf.keras.layers.Conv2D(filters=self.config['num_hidden_units'],
                                        kernel_size=(4, self.config['n_channel']),
                                        strides=1,
                                        padding='valid',
                                        activation='leaky_relu',
                                        kernel_initializer=init)(conv_3)  # Step 4
        print("conv_4: {}".format(conv_4))
      
    ###########  48 Encoder ###########

    elif self.config['l_win'] == 48:
        conv_1 = tf.keras.layers.Conv2D(filters=self.config['num_hidden_units'] // 16,
                                        kernel_size=(3, self.config['n_channel']),
                                        strides=(2, 1),
                                        padding='same',
                                        activation='leaky_relu',
                                        kernel_initializer=init)(input_tensor)  # Step 4
        print("conv_1: {}".format(conv_1))
        conv_2 = tf.keras.layers.Conv2D(filters=self.config['num_hidden_units'] // 8,
                                        kernel_size=(3, self.config['n_channel']),
                                        strides=(2, 1),
                                        padding='same',
                                        activation='leaky_relu',
                                        kernel_initializer=init)(conv_1)  # Step 4
        print("conv_2: {}".format(conv_2))
        conv_3 = tf.keras.layers.Conv2D(filters=self.config['num_hidden_units'] // 4,
                                        kernel_size=(3, self.config['n_channel']),
                                        strides=(2, 1),
                                        padding='same',
                                        activation='leaky_relu',
                                        kernel_initializer=init)(conv_2)  # Step 4
        print("conv_3: {}".format(conv_3))
        conv_4 = tf.keras.layers.Conv2D(filters=self.config['num_hidden_units'],
                                        kernel_size=(6, self.config['n_channel']),
                                        strides=1,
                                        padding='valid',
                                        activation='leaky_relu',
                                        kernel_initializer=init)(conv_3)  # Step 4
        print("conv_4: {}".format(conv_4))
      
      
    #############  144 Encoder #############
        
    elif self.config['l_win'] == 144:
        conv_1 = tf.keras.layers.Conv2D(filters=self.config['num_hidden_units'] // 16,
                                        kernel_size=(3, self.config['n_channel']),
                                        strides=(4, 1),
                                        padding='same',
                                        activation='leaky_relu',
                                        kernel_initializer=init)(input_tensor)  # Step 4
        print("conv_1: {}".format(conv_1))
        conv_2 = tf.keras.layers.Conv2D(filters=self.config['num_hidden_units'] // 8,
                                        kernel_size=(3, self.config['n_channel']),
                                        strides=(4, 1),
                                        padding='same',
                                        activation='leaky_relu',
                                        kernel_initializer=init)(conv_1)  # Step 4
        print("conv_2: {}".format(conv_2))
        conv_3 = tf.keras.layers.Conv2D(filters=self.config['num_hidden_units'] // 4,
                                        kernel_size=(3, self.config['n_channel']),
                                        strides=(3, 1),
                                        padding='same',
                                        activation='leaky_relu',
                                        kernel_initializer=init)(conv_2)  # Step 4
        print("conv_3: {}".format(conv_3))
        conv_4 = tf.keras.layers.Conv2D(filters=self.config['num_hidden_units'],
                                        kernel_size=(3, self.config['n_channel']),
                                        strides=1,
                                        padding='valid',
                                        activation='leaky_relu',
                                        kernel_initializer=init)(conv_3)  # Step 4
        print("conv_4: {}".format(conv_4))

    
    
    ########  CODE MEAN + CODE STD DEV #########
      
    # Step 5: Convert SymbolicTensor to concrete tensor before passing to Flatten
    
    conv_4_flat = tf.keras.backend.flatten(conv_4)
    
    encoded_signal = tf.keras.layers.Flatten()(conv_4_flat)  # Step 5
      
      
    encoded_signal = tf.keras.layers.Dense(units=self.config['code_size'] * 4,
                                             activation=tf.nn.leaky_relu,
                                             kernel_initializer=init)(encoded_signal)  # Step 6

    code_mean = tf.keras.layers.Dense(units=self.config['code_size'],
                                             activation=None,
                                             kernel_initializer=init,
                                             name='code_mean')(encoded_signal)  # Step 6
                                             
    code_std_dev = tf.keras.layers.Dense(units=self.config['code_size'],
                                                activation=tf.nn.relu,
                                                kernel_initializer=init,
                                                name='code_std_dev')(encoded_signal)  # Step 6
    code_std_dev = code_std_dev + 1e-2

    code_mean = tf.convert_to_tensor(code_mean)
    code_std_dev = tf.convert_to_tensor(code_std_dev)
    
    ##### DEBUG ####
    print(code_mean.shape)
    print(code_std_dev.shape)
    ##### END ######

    mvn = tfp.distributions.MultivariateNormalDiag(loc=code_mean, scale_diag=code_std_dev)
    code_sample = mvn.sample()
    
    print("finish encoder: \n{}".format(self.pppppppp))
    print("\n")

    # Return the encoder as a callable function
    return tf.keras.Model(inputs=inputs, outputs=[code_mean, code_std_dev, code_sample])


  
  
  ##############  Decoder ################

  def build_decoder(self):
    
    init = tf.keras.initializers.GlorotUniform()  # Step 1
    code_input = tf.keras.Input(shape=(self.config['code_size'],))
    
  
    encoded = tf.keras.layers.Dense(units=self.config['num_hidden_units'],
                                    activation=tf.nn.leaky_relu,
                                    kernel_initializer=init)(code_input)  # Step 6
    decoded_1 = tf.keras.layers.Reshape((1, 1, self.config['num_hidden_units']))(encoded)
      
      
    #######  24 Decoder ########

    if self.config['l_win'] == 24:
        decoded_2 = tf.keras.layers.Conv2D(filters=self.config['num_hidden_units'],
                                           kernel_size=1,
                                           padding='same',
                                           activation=tf.nn.leaky_relu)(decoded_1)  # Step 6
        decoded_2 = tf.reshape(decoded_2, [-1, 4, 1, self.config['num_hidden_units'] // 4])
        print("decoded_2 is: {}".format(decoded_2))
        
        decoded_3 = tf.keras.layers.Conv2DTranspose(filters=self.config['num_hidden_units'] // 4,
                                                    kernel_size=(3, 1),
                                                    strides=1,
                                                    padding='same',
                                                    activation=tf.nn.leaky_relu,
                                                    kernel_initializer=init)(decoded_2)  # Step 7
        decoded_3 = tf.nn.depth_to_space(input=decoded_3,
                                         block_size=2)
        decoded_3 = tf.reshape(decoded_3, [-1, 8, 1, self.config['num_hidden_units'] // 8])
        print("decoded_3 is: {}".format(decoded_3))


        decoded_4 = tf.keras.layers.Conv2DTranspose(filters=self.config['num_hidden_units'] // 8,
                                                    kernel_size=(3, 1),
                                                    strides=1,
                                                    padding='same',
                                                    activation=tf.nn.leaky_relu,
                                                    kernel_initializer=init)(decoded_3)  # Step 7
        decoded_4 = tf.nn.depth_to_space(input=decoded_4,
                                         block_size=2)
        decoded_4 = tf.reshape(decoded_4, [-1, 16, 1, self.config['num_hidden_units'] // 16])
        print("decoded_4 is: {}".format(decoded_4))


        decoded_5 = tf.keras.layers.Conv2DTranspose(filters=self.config['num_hidden_units'] // 16,
                                                    kernel_size=(3, 1),
                                                    strides=1,
                                                    padding='same',
                                                    activation=tf.nn.leaky_relu,
                                                    kernel_initializer=init)(decoded_4)  # Step 7
        decoded_5 = tf.nn.depth_to_space(input=decoded_5,
                                         block_size=2)
        decoded_5 = tf.reshape(decoded_5, [-1, self.config['num_hidden_units'] // 16, 1, 16])
        print("decoded_5 is: {}".format(decoded_5))


        decoded = tf.keras.layers.Conv2DTranspose(filters=self.config['n_channel'],
                                                  kernel_size=(9, 1),
                                                  strides=1,
                                                  padding='valid',
                                                  activation=None,
                                                  kernel_initializer=init)(decoded_5)  # Step 7
        print("decoded_6 is: {}".format(decoded))
        self.decoded = tf.reshape(decoded, [-1, self.config['l_win'], self.config['n_channel']])
      
      
    ###########  48 Decoder ###########
        
    elif self.config['l_win'] == 48:
        decoded_2 = tf.keras.layers.Conv2DTranspose(filters=256 * 3,
                                                    kernel_size=1,
                                                    padding='same',
                                                    activation=tf.nn.leaky_relu)(decoded_1)  # Step 7
        decoded_2 = tf.reshape(decoded_2, [-1, 3, 1, 256])
        print("decoded_2 is: {}".format(decoded_2))
        
        decoded_3 = tf.keras.layers.Conv2DTranspose(filters=256,
                                                    kernel_size=(3, 1),
                                                    strides=1,
                                                    padding='same',
                                                    activation=tf.nn.leaky_relu,
                                                    kernel_initializer=init)(decoded_2)  # Step 7
        decoded_3 = tf.nn.depth_to_space(input=decoded_3,
                                         block_size=2)
        decoded_3 = tf.reshape(decoded_3, [-1, 6, 1, 128])
        print("decoded_3 is: {}".format(decoded_3))
        
        decoded_4 = tf.keras.layers.Conv2DTranspose(filters=128,
                                                    kernel_size=(3, 1),
                                                    strides=1,
                                                    padding='same',
                                                    activation=tf.nn.leaky_relu,
                                                    kernel_initializer=init)(decoded_3)  # Step 7
        decoded_4 = tf.nn.depth_to_space(input=decoded_4,
                                         block_size=2)
        decoded_4 = tf.reshape(decoded_4, [-1, 24, 1, 32])
        print("decoded_4 is: {}".format(decoded_4))
        
        decoded_5 = tf.keras.layers.Conv2DTranspose(filters=32,
                                                    kernel_size=(3, 1),
                                                    strides=1,
                                                    padding='same',
                                                    activation=tf.nn.leaky_relu,
                                                    kernel_initializer=init)(decoded_4)  # Step 7
        decoded_5 = tf.nn.depth_to_space(input=decoded_5,
                                         block_size=2)
        decoded_5 = tf.reshape(decoded_5, [-1, 48, 1, 16])
        print("decoded_5 is: {}".format(decoded_5))
        
        decoded = tf.keras.layers.Conv2DTranspose(filters=1,
                                                  kernel_size=(5, self.config['n_channel']),
                                                  strides=1,
                                                  padding='same',
                                                  activation=None,
                                                  kernel_initializer=init)(decoded_5)  # Step 7
        print("decoded_6 is: {}".format(decoded))
        self.decoded = tf.reshape(decoded, [-1, self.config['l_win'], self.config['n_channel']])
      
      
    ########  144 Decoder ########
        
    elif self.config['l_win'] == 144:
        decoded_2 = tf.keras.layers.Conv2DTranspose(filters=32 * 27,
                                                    kernel_size=1,
                                                    strides=1,
                                                    padding='same',
                                                    activation=tf.nn.leaky_relu)(decoded_1)  # Step 7
        decoded_2 = tf.reshape(decoded_2, [-1, 3, 1, 32 * 9])
        print("decoded_2 is: {}".format(decoded_2))
        decoded_3 = tf.keras.layers.Conv2DTranspose(filters=32 * 9,
                                                    kernel_size=(3, 1),
                                                    strides=1,
                                                    padding='same',
                                                    activation=tf.nn.leaky_relu,
                                                    kernel_initializer=init)(decoded_2)  # Step 7
        decoded_3 = tf.nn.depth_to_space(input=decoded_3, block_size=3)
        
        decoded_3 = tf.reshape(decoded_3, [-1, 9, 1, 32 * 3])
        print("decoded_3 is: {}".format(decoded_3))
        decoded_4 = tf.keras.layers.Conv2DTranspose(filters=32 * 3,
                                                    kernel_size=(3, 1),
                                                    strides=1,
                                                    padding='same',
                                                    activation=tf.nn.leaky_relu,
                                                    kernel_initializer=init)(decoded_3)  # Step 7
        decoded_4 = tf.nn.depth_to_space(input=decoded_4,
                                        block_size=2)
        decoded_4 = tf.reshape(decoded_4, [-1, 36, 1, 24])
        print("decoded_4 is: {}".format(decoded_4))

        decoded_5 = tf.keras.layers.Conv2DTranspose(filters=24,
                                                    kernel_size=(3, 1),
                                                    strides=1,
                                                    padding='same',
                                                    activation=tf.nn.leaky_relu,
                                                    kernel_initializer=init)(decoded_4)  # Step 7
        decoded_5 = tf.nn.depth_to_space(input=decoded_5, block_size=2)
        decoded_5 = tf.reshape(decoded_5, [-1, 144, 1, 6])
        print("decoded_5 is: {}".format(decoded_5))

        decoded = tf.keras.layers.Conv2DTranspose(filters=1,
                                                  kernel_size=(9, self.config['n_channel']),
                                                  strides=1,
                                                  padding='same',
                                                  activation=None,
                                                  kernel_initializer=init)(decoded_5)  # Step 7
        print("decoded_6 is: {}".format(decoded))
        self.decoded = tf.reshape(decoded, [-1, self.config['l_win'], self.config['n_channel']])
        print("finish decoder: \n{}".format(self.decoded))
        print('\n')

        return tf.keras.Model(inputs=[code_input], outputs=decoded)


  
  
  ############### SIGMA ############
  def build_sigma2(self):
    self.sigma2_offset = tf.constant(self.config['sigma2_offset']) 
    
    if self.config['TRAIN_sigma'] == 1:
      
      sigma = tf.Variable(tf.cast(self.config['sigma'], tf.float32), dtype=tf.float32, trainable=True)
    
    else:
      
      sigma = tf.cast(self.config['sigma'], tf.float32)

    sigma2 = tf.square(sigma)

    if self.config['TRAIN_sigma'] == 1:
      sigma2 = sigma2 + self.sigma2_offset

    print("sigma2: \n{}\n".format(sigma2))
    return sigma2

 
 
 
 
  ############ LOSS ############

  def calculate_loss(self, code_mean, code_std_dev, original_signal, decoded):
    # KL divergence loss - analytical result
    kl_loss = 0.5 * (tf.reduce_sum(tf.square(code_mean), 1)
                     + tf.reduce_sum(tf.square(code_std_dev), 1)
                     - tf.reduce_sum(tf.math.log(tf.square(code_std_dev)), 1)
                     - self.config['code_size'])
    kl_loss = tf.reduce_mean(kl_loss)

    # norm 1 of standard deviation of the sample-wise encoder prediction
    std_dev_norm = tf.reduce_mean(code_std_dev, axis=0)

    weighted_reconstruction_error = tf.reduce_sum(
        tf.square(original_signal - decoded), [1, 2])
    weighted_reconstruction_error = tf.reduce_mean(weighted_reconstruction_error)
    weighted_reconstruction_error /= (2 * self.sigma2)

    # least squared reconstruction error
    ls_reconstruction_error = tf.reduce_sum(
        tf.square(original_signal - decoded), [1, 2])
    ls_reconstruction_error = tf.reduce_mean(ls_reconstruction_error)

    # sigma regularisor - input elbo
    sigma_regularisor = self.input_dims / 2 * tf.math.log(self.sigma2)
    two_pi = self.input_dims / 2 * tf.constant(2 * np.pi)

    elbo_loss = two_pi + sigma_regularisor + \
                0.5 * weighted_reconstruction_error + kl_loss

    return kl_loss, std_dev_norm, weighted_reconstruction_error, ls_reconstruction_error, elbo_loss

  
  ############## CALL #############

  def call(self, inputs, training=None):
    original_signal = inputs
    
    # Pass the input through the encoder to obtain the code mean, std_dev, and sample
    code_mean, code_std_dev, code_sample = self.encoder(original_signal)
    
    # Pass the code sample and the boolean tensor to the decoder
    decoded = self.decoder(code_sample)
    
    # Reshape the decoded output to match the original signal shape
    decoded = tf.reshape(decoded, [-1, self.config['l_win'], self.config['n_channel']])
    
    # Calculate the different loss components
    kl_loss, std_dev_norm, weighted_reconstruction_error, ls_reconstruction_error, elbo_loss = self.calculate_loss(
        code_mean, code_std_dev, original_signal, decoded)
    
    # Add the losses to the model's loss dictionary
    self.add_loss(weighted_reconstruction_error)
    self.add_metric(kl_loss, name='kl_loss', aggregation='mean')
    self.add_metric(std_dev_norm, name='std_dev_norm', aggregation='mean')
    self.add_metric(weighted_reconstruction_error, name='weighted_reconstruction_error', aggregation='mean')
    self.add_metric(ls_reconstruction_error, name='ls_reconstruction_error', aggregation='mean')
    self.add_metric(elbo_loss, name='elbo_loss', aggregation='mean')
    
    return decoded

  ############# train_step #############
  @tf.function
  def train_step(self, x):
    """Perform a single training step on a batch of data.

    Args:
        data: A tuple containing the input data and target data.

    Returns:
        A dictionary of metric values for the current training step.
    """

    with tf.GradientTape() as tape:
        # Forward pass through the model using the call function
        decoded = self(x)

        # Compute the total loss
        total_loss = self.losses[0]

    # Compute gradients
    gradients = tape.gradient(total_loss, self.trainable_variables)

    # Apply gradients to the model's variables
    self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

    # Update the metrics
    self.compiled_metrics.update_state(x, decoded)

    # Return a dictionary of metric values
    return {m.name: m.result() for m in self.metrics}

############################ LSTM ####################################
class lstmKerasModel(tf.keras.Model):
  
  def __init__(self, config):
    super().__init__()
    self.config     = config
    self.lstm_model = self.create_lstm_model()
    

  def create_lstm_model(self):
    lstm_input = tf.keras.layers.Input(shape=(self.config['l_seq'] - 1, self.config['code_size']))
    LSTM1 = tf.keras.layers.LSTM(self.config['num_hidden_units_lstm'], return_sequences=True)(lstm_input)
    LSTM2 = tf.keras.layers.LSTM(self.config['num_hidden_units_lstm'], return_sequences=True)(LSTM1)
    lstm_output = tf.keras.layers.LSTM(self.config['code_size'], return_sequences=True, activation=None)(LSTM2)
    lstm_model = tf.keras.Model(lstm_input, lstm_output)
    lstm_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.config['learning_rate_lstm']),
                       loss='mse',
                       metrics=['mse'])
    return lstm_model


  def call(self, inputs):
    """
    The call function for the lstmKerasModel.

    This function takes the input sequences and passes them through the LSTM layers
    to generate the output sequences. The input and output sequences have the same shape.

    Args:
        inputs (tf.Tensor): Input tensor of shape (batch_size, l_seq - 1, code_size).

    Returns:
        tf.Tensor: Output tensor of shape (batch_size, l_seq - 1, code_size).
    """
    # Pass the input through the LSTM layers
    lstm_output = self.lstm_model(inputs)

    # The lstm_model returns the output sequences for all time steps
    # We only need the output sequences for the first l_seq - 1 time steps
    # as the input sequences have length l_seq - 1
    output = lstm_output[:, :self.config['l_seq'] - 1, :]

    return output

  
  
  def produce_embeddings(self, model_vae, data):
    self.embedding_lstm_train = np.zeros((data.n_train_lstm, self.config['l_seq'], self.config['code_size']))
    
    for i in range(data.n_train_lstm):
        print(f'Producing embedding : {i}')
        self.embedding_lstm_train[i] = model_vae.encoder(data.train_set_lstm['data'][i])[0].numpy()
    
    print("Finish processing the embeddings of the entire dataset.")
    print("The first a few embeddings are\n{}".format(self.embedding_lstm_train[0, 0:5]))
    
    self.x_train = self.embedding_lstm_train[:, :self.config['l_seq'] - 1]
    self.y_train = self.embedding_lstm_train[:, 1:]

    self.embedding_lstm_test = np.zeros((data.n_val_lstm, self.config['l_seq'], self.config['code_size']))
    
    for i in range(data.n_val_lstm):
        self.embedding_lstm_test[i] = model_vae.encoder(data.val_set_lstm['data'][i])[0].numpy()
    
    self.x_test = self.embedding_lstm_test[:, :self.config['l_seq'] - 1]
    self.y_test = self.embedding_lstm_test[:, 1:]

  
  
  def load_model(self, checkpoint_path):
    print(self.config['checkpoint_dir_lstm'] + 'checkpoint')
    if os.path.isfile(self.config['checkpoint_dir_lstm'] + 'checkpoint'):
      self.lstm_model.load_weights(checkpoint_path)
      print("LSTM model loaded.")
    else:
      print("No LSTM model loaded.")

  
  
  def train(self, cp_callback):
    self.lstm_model.fit(self.x_train, self.y_train,
                   validation_data=(self.x_test, self.y_test),
                   batch_size=self.config['batch_size_lstm'],
                   epochs=self.config['num_epochs_lstm'],
                   callbacks=[cp_callback])

  
  
  def plot_reconstructed_lt_seq(self, idx_test, model_vae, data, lstm_embedding_test):
    """
    This function plots the reconstructed sequences from both the VAE and LSTM models 
    alongside the ground truth data for comparison. It visualizes how well each model
    is able to reconstruct the original time series data. 
    """

    decoded_seq_vae = np.squeeze(model_vae.decoder(self.embedding_lstm_test[idx_test], 
                                                   tf.ones(self.embedding_lstm_test[idx_test].shape[0], dtype=tf.bool),
                                                   self.embedding_lstm_test[idx_test]).numpy())
    print("Decoded seq from VAE: {}".format(decoded_seq_vae.shape))

    decoded_seq_lstm = np.squeeze(model_vae.decoder(lstm_embedding_test[idx_test],
                                                    tf.ones(lstm_embedding_test[idx_test].shape[0], dtype=tf.bool),
                                                    lstm_embedding_test[idx_test]).numpy())
    print("Decoded seq from lstm: {}".format(decoded_seq_lstm.shape))

    fig, axs = plt.subplots(self.config['n_channel'], 2, figsize=(15, 4.5 * self.config['n_channel']), edgecolor='k')
    fig.subplots_adjust(hspace=.4, wspace=.4)
    axs = axs.ravel()
    for j in range(self.config['n_channel']):
      for i in range(2):
        axs[i + j * 2].plot(np.arange(0, self.config['l_seq'] * self.config['l_win']),
                            np.reshape(data.val_set_lstm['data'][idx_test, :, :, j],
                                       (self.config['l_seq'] * self.config['l_win'])))
        axs[i + j * 2].grid(True)
        axs[i + j * 2].set_xlim(0, self.config['l_seq'] * self.config['l_win'])
        axs[i + j * 2].set_xlabel('samples')
      if self.config['n_channel'] == 1:
        axs[0 + j * 2].plot(np.arange(0, self.config['l_seq'] * self.config['l_win']),
                            np.reshape(decoded_seq_vae, (self.config['l_seq'] * self.config['l_win'])), 'r--')
        axs[1 + j * 2].plot(np.arange(self.config['l_win'], self.config['l_seq'] * self.config['l_win']),
                            np.reshape(decoded_seq_lstm, ((self.config['l_seq'] - 1) * self.config['l_win'])), 'g--')
      else:
        axs[0 + j * 2].plot(np.arange(0, self.config['l_seq'] * self.config['l_win']),
                            np.reshape(decoded_seq_vae[:, :, j], (self.config['l_seq'] * self.config['l_win'])), 'r--')
        axs[1 + j * 2].plot(np.arange(self.config['l_win'], self.config['l_seq'] * self.config['l_win']),
                            np.reshape(decoded_seq_lstm[:, :, j], ((self.config['l_seq'] - 1) * self.config['l_win'])), 'g--')
      axs[0 + j * 2].set_title('VAE reconstruction - channel {}'.format(j))
      axs[1 + j * 2].set_title('LSTM reconstruction - channel {}'.format(j))
      for i in range(2):
        axs[i + j * 2].legend(('ground truth', 'reconstruction'))
      savefig(self.config['result_dir'] + "lstm_long_seq_recons_{}.pdf".format(idx_test))
      fig.clf()
      plt.close()

  
  
  def plot_lstm_embedding_prediction(self, idx_test, model_vae, data, lstm_embedding_test):

    """ 
    This function visualizes the LSTM embedding predictions and compares them to 
    the original VAE embeddings. It helps to understand how the LSTM model is 
    transforming the latent space representation of the time series data.
    """

    self.plot_reconstructed_lt_seq(idx_test, model_vae, data, lstm_embedding_test)

    fig, axs = plt.subplots(2, self.config['code_size'] // 2, figsize=(15, 5.5), edgecolor='k')
    fig.subplots_adjust(hspace=.4, wspace=.4)
    axs = axs.ravel()
    for i in range(self.config['code_size']):
      axs[i].plot(np.arange(1, self.config['l_seq']), np.squeeze(self.embedding_lstm_test[idx_test, 1:, i]))
      axs[i].plot(np.arange(1, self.config['l_seq']), np.squeeze(lstm_embedding_test[idx_test, :, i]))
      axs[i].set_xlim(1, self.config['l_seq'] - 1)
      axs[i].set_ylim(-2.5, 2.5)
      axs[i].grid(True)
      axs[i].set_title('Embedding dim {}'.format(i))
      axs[i].set_xlabel('windows')
      if i == self.config['code_size'] - 1:
        axs[i].legend(('VAE\nembedding', 'LSTM\nembedding'))
    savefig(self.config['result_dir'] + "lstm_seq_embedding_{}.pdf".format(idx_test))
    fig.clf()
    plt.close()
