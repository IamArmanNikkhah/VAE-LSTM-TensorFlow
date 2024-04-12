import tensorflow as tf


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
