# VAE-LSTM-TensorFlow

This repository provides a TensorFlow 2 implementation of a hybrid Variational Autoencoder (VAE) combined with Long Short-Term Memory (LSTM) layers for modeling sequential or time-series data.

## Features

- **Hybrid VAE–LSTM architecture** – uses an LSTM encoder and decoder within a VAE framework to learn meaningful latent representations of sequences.
- **TensorFlow 2 / Keras API** – built with TensorFlow 2.x and the Keras API for ease of use and customization.
- **Dataset loader** – includes a simple dataset loader (`dataset.py`) for preparing sequence data.
- **Modular code** – separate modules for model definitions (`vae.py`, `lstm.py`), dataset handling, and training scripts.
- **MIT licensed** – free to use in research and commercial projects.

## Project Structure

```
├── dataset.py   # dataset loader and pre-processing
├── lstm.py      # defines LSTM encoder/decoder components
├── vae.py       # defines VAE architecture using LSTM layers
├── README.md    # project documentation
└── LICENSE      # license information
```

## Getting Started

1. **Clone the repository**

```bash
git clone https://github.com/IamArmanNikkhah/VAE-LSTM-TensorFlow.git
cd VAE-LSTM-TensorFlow
```

2. **Install dependencies**

Make sure you have Python 3.7+ and TensorFlow 2.x installed. You can install the required packages using:

```bash
pip install tensorflow tensorflow-probability numpy
```

3. **Run the training script**

Currently the training script is contained in `vae.py` and can be executed to train the model on your dataset. You may need to adapt the code to your specific data format.

## Usage

You can import the model classes into your own projects and train on your own sequence data:

```python
from vae import VAE_LSTM

model = VAE_LSTM(input_dim=128, latent_dim=64, lstm_units=256)
model.compile(optimizer="adam")
# prepare your dataset...
model.fit(dataset, epochs=50)
```

Refer to the source files for more details and configuration options.

## Contributing

Pull requests are welcome. Please open an issue to discuss your suggestions or improvements before submitting a PR.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
