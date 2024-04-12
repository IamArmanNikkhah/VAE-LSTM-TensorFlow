
from tensorflow.keras import layers
import matplotlib.pylab as plt
from matplotlib.pyplot import savefig
import tensorflow_probability as tfp

import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import tensorflow as tf
import keras
from keras import layers


tfd = tfp.distributions



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


class VAE(keras.Model):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        
        self.config  = config
        
        self.encoder = Encoder(self.config)
        self.decoder = Decoder(self.config)
        
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    
    @tf.function
    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction),
                    axis=(1, 2),
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
              }   























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













