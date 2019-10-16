# autoencoder_builder

Small helper library to build various autoencoder networks with ease, run inference with them and store to file system.

Autoencoders are fairly commonly used networks and rather simple structured so this is just to reduce boilerplate.

This is mainly for my own use for I research AEs quite a lot, so the API might change a lot. Also still quite barren, will add features.

No proper documentation yet, check code, comments and examples.

## Features:
* Build autoencoder configurations with builder object
* Good defaults from Keras, only required arguments for builder:
    * Input data shape
    * Encoded latent dimension size (middle layer)
    * Hidden layer units for encoder (decoder is mirrored by default)
* Configure, build, compile and train all three networks with a few lines of code
* Store configurations and weights to file system or load from it

## TODO:
* Different types of layers (CNN, LSTM)
* Automatic architecture search
* Variational autoencoders (VAE)

## Examples:


#### Build autoencoder to reconstruct MNIST digits.
```python
from autoencoder_builder.autoencoder import AutoencoderBuilder
from keras.datasets.mnist import load_data

(x_train, y_train), (x_test, y_test) = load_data()

input_shape = x_train.shape[1:]

# Set up builder with the configuration you want, or leave them as default.
builder = AutoencoderBuilder(input_shape=input_shape
                             , encoded_dim=64
                             , l1=1e-6
                             , layers=[700, 300]
                             , early_stopping_patience=0)

# Build, compile and train the network and build Autoencoder object.
autoencoder = builder.build_and_train(input_data=x_train[:5000]
                                      , epochs=100
                                      , validation_split=.15)

# Print Keras summary of the full network.
autoencoder.full_model.summary()

# Encode test data!
encoded_test = autoencoder.encode(x_test[0])

# Decode the encoded test into original shape.
decoded_test = autoencoder.decode(encoded_test)

# Or do it all in one go, to evaluate the network.
reconstructed_tests = autoencoder.reconstruct(x_test[1:5]) 

# Store to file system.
AutoencoderBuilder.save_to_files(autoencoder, "test_config", "test_weights")
```

#### Load previous model from file system.
```python
# Note that the files do not store the builder configuration, that's why it is needed.
builder = AutoencoderBuilder(input_shape=input_shape
                             , encoded_dim=64
                             , l1=1e-6
                             , layers=[700, 300]
                             , early_stopping_patience=0)

# Load trained model from file system.                             
autoencoder = builder.load_from_files("test_config", "test_weights")

# Get the underlying encoder network Keras model.
encoder_keras = autoencoder.encoder_model

# Continue working with it without retraining it...
```