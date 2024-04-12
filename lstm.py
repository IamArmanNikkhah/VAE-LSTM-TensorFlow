
import matplotlib.pylab as plt
from matplotlib.pyplot import savefig
import os
import numpy as np
import tensorflow as tf

os.environ["KERAS_BACKEND"] = "tensorflow"




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