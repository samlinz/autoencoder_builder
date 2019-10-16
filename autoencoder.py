from enum import Enum
from typing import Optional, List, Union, Callable, Tuple

import numpy as np
import keras


def _get_encoder_layer_name(layer):
    return f"encoder_{layer}"


def _get_decoder_layer_name(layer):
    return f"decoder_{layer}"


_ENCODED_LAYER_NAME = "encoded"
_DECODED_LAYER_NAME = "decoded"


class Autoencoder:
    """Represents a trained and ready-to-use autoencoder network."""

    def __init__(self
                 , full_model: keras.models.Model
                 , encoder_model: keras.models.Model
                 , decoder_model: keras.models.Model):
        self.__encoder = encoder_model
        self.__decoder = decoder_model
        self.__full = full_model

    def __predict(self, input, network_part, model):
        given_input_shape = input.shape
        expected_shape = model.layers[0].input_shape

        # If given only one sample, expand dimensions to have batch dimension of one.
        singular_input = False
        if len(given_input_shape) == len(
                expected_shape) - 1 and given_input_shape == expected_shape[1:]:
            input = np.expand_dims(input, axis=0)
            given_input_shape = input.shape
            singular_input = True

        if given_input_shape[1:] != expected_shape[1:]:
            raise ValueError(
                f"Data shape {given_input_shape} does not match {network_part} input"
                f" {expected_shape}")

        result = model.predict(input)
        return result[0] if singular_input else result

    def decode(self, input):
        """Run the input through decoder."""
        return self.__predict(input, "decoder", self.__decoder)

    def encode(self, input):
        """Encode the input."""
        return self.__predict(input, "encoder", self.__encoder)

    def reconstruct(self, input):
        """Run the input through entire network, returning the reconstructed representation."""
        return self.__predict(input, "full_model", self.__full)

    @property
    def full_model(self):
        return self.__full

    @property
    def encoder_model(self):
        return self.__encoder

    @property
    def decoder_model(self):
        return self.__decoder


class AutoencoderBuilder:
    """Builder for Autoencoder objects. Also some static utility functions."""

    class LAYER_TYPES(Enum):
        Dense = 1

    def __init__(self
                 , *
                 , input_shape
                 , encoded_dim: Union[int, Tuple]
                 , type: LAYER_TYPES = LAYER_TYPES.Dense
                 , loss: Union[Callable, str] = keras.losses.mse
                 , activation: Callable = keras.activations.relu
                 , output_activation: Callable = None
                 , use_batch_normalization: bool = True
                 , early_stopping_patience: Optional[int] = None
                 , l1: Optional[float] = 1e-6
                 , l2: Optional[float] = None
                 , optimizer: keras.optimizers.Optimizer = keras.optimizers.Adam(lr=0.001)
                 , layers: Optional[List[int]] = None):
        """
        Construct the builder with given properties. The builder can then build, compile and train
        Autoencoder objects.

        :param input_shape: Input shape for data that the (full) network should take in.
        :param encoded_dim: Dimensions of the encoded latent dimensions (number of units).
        :param type: Type of the network from the enum, dense, cnn etc. Problem specific.
        :param loss: Loss function to use, either callable or Keras hardcoded string.
        :param activation: Activation function to use in all hidden layers. Callable or string.
        :param output_activation: Activation function to use in decoder's last layer.
        :param use_batch_normalization: If true, add batch normalization layers between hidden
        layers.
        :param early_stopping_patience: If not None, use early stopping callback with this patience.
        :param l1: If not None, use Lasso regression regularization with this parameter.
        :param l2:If not None, use Ridge regression regularization with this parameter.
        :param optimizer: Keras optimizer to use to train the network.
        :param layers: If present, the list of hidden units in encoder layers, decoder will have
        the same but mirrored. Also determined the number of hidden layers. No encoded layer!
        """

        self.set_type(type)
        self.set_optimizer(optimizer)
        self.set_activation(activation, output_activation)
        self.set_encoded_dim(encoded_dim)
        self.set_batch_normalization(use_batch_normalization)
        self.set_regularization(l1, l2)
        self.set_loss(loss)
        self.set_early_stopping(early_stopping_patience)
        if layers:
            self.set_layers(layers)

        self.__trained = False
        self.set_input_shape(input_shape)

    def set_type(self, type: LAYER_TYPES):
        self.__type = type

    def set_early_stopping(self, patience: Optional[int]):
        self.__early_stopping = patience

    def set_encoded_dim(self, encoded_dim: int):
        self.__encoded_dim = encoded_dim

    def set_loss(self, loss: Union[Callable, str]):
        if type(loss) == "str":
            loss = keras.losses.get(loss)
            if not loss: raise ValueError(f"Failed to get loss function {loss}")
        self.__loss = loss

    def set_layers(self, sizes=List[int]):
        self.set_encoder_layers(sizes)
        self.set_decoder_layers(list(reversed(sizes)))

    def set_optimizer(self, optimizer: Union[keras.optimizers.Optimizer, str]):
        if type(optimizer) == "str":
            instance = keras.optimizers.get(optimizer)
            if not instance: raise ValueError(f"Failed to get activation {instance}")
        else:
            instance = optimizer
        self.__optimizer = instance

    def set_activation(self, activation: Union[Callable, str],
                       output_activation: Optional[Union[Callable, str]] = None):
        if type(activation) == "str":
            instance = keras.activations.get(activation)
            if not instance: raise ValueError(f"Failed to get activation {activation}")
        else:
            instance = activation

        self.__activation = instance

        output_instance = None
        if output_activation:
            if type(output_activation) == "str":
                output_instance = keras.activations.get(output_activation)
                if not output_instance: raise ValueError(
                    f"Failed to get activation {output_activation}")
            else:
                output_instance = output_activation

        self.__output_activation = output_instance

    def set_encoder_layers(self
                           , sizes=List[int]):
        self.__encoder_layers = sizes

    def set_decoder_layers(self
                           , sizes=List[int]):
        self.__decoder_layers = sizes

    def set_input_shape(self, shape: Union[int, tuple]):
        if isinstance(shape, int):
            shape = shape,
        self.__input_shape = shape

    def set_regularization(self, l1: Optional[float], l2: Optional[float]):
        self.__l1 = l1
        self.__l2 = l2

    def set_batch_normalization(self, value=True):
        self.__use_batch_normalization = value

    def __get_layer(self, units, direction="encoder", name=None, activation=None):
        activity_regularizer = None
        if self.__l1:
            activity_regularizer = keras.regularizers.l1(self.__l1)
        if self.__l2:
            activity_regularizer = keras.regularizers.l2(self.__l2)
        if self.__l1 and self.__l2:
            activity_regularizer = keras.regularizers.l1_l2(self.__l1, self.__l2)

        if self.__type == AutoencoderBuilder.LAYER_TYPES.Dense:
            return keras.layers.Dense(units
                                      ,
                                      activation=self.__activation if activation is None else
                                      activation
                                      , activity_regularizer=activity_regularizer
                                      , name=name)

        raise ValueError(f"Invalid type {self.__type}")

    def __compile(self, model):
        model.compile(optimizer=self.__optimizer, loss=self.__loss)

    def _build_models(self) -> Tuple[keras.models.Model, keras.models.Model, keras.models.Model]:
        if not self.__input_shape: raise ValueError("Input shape not set")
        input_layer = keras.layers.Input(shape=self.__input_shape)

        x = input_layer

        # Build encoder.
        if not self.__encoder_layers:
            raise ValueError("Encoder layers not set")

        encoder_layer_name = None
        encoder_layers = self.__encoder_layers + [self.__encoded_dim]
        encoder_layers_count = len(encoder_layers)

        # If network is MLP and input is not flat, it has to be flattened before further processing.
        input_dim = np.prod(self.__input_shape)
        flattened = False
        if self.__type == self.LAYER_TYPES.Dense and len(self.__input_shape) != 1:
            flattened = True
            x = keras.layers.Flatten()(x)

        for i, encoder_layer in enumerate(encoder_layers):
            encoded = i == encoder_layers_count - 1
            encoder_layer_name = _get_encoder_layer_name(i) if not encoded else _ENCODED_LAYER_NAME
            x = self.__get_layer(encoder_layer, direction="encoder", name=encoder_layer_name)(x)

            # TODO: In which parts should batch normalization be applied?
            # Should encoded layer have normalization?
            if self.__use_batch_normalization and not encoded:
                x = keras.layers.BatchNormalization()(x)

        encoded_layer = x

        # Build decoder.
        if not self.__decoder_layers:
            raise ValueError("Decoder layers not set")

        decoder_layers = self.__decoder_layers + [input_dim]
        decoder_layers_count = len(decoder_layers)
        for i, decoder_layer in enumerate(decoder_layers):
            decoded = i == decoder_layers_count - 1
            decoder_layer_name = _get_decoder_layer_name(i) if not decoded else _DECODED_LAYER_NAME
            activation = None

            if decoded and self.__output_activation:
                activation = self.__output_activation

            x = self.__get_layer(decoder_layer
                                 , direction="decoder"
                                 , name=decoder_layer_name
                                 , activation=activation)(x)

            if self.__use_batch_normalization and not decoded:
                x = keras.layers.BatchNormalization()(x)

        # If input was flattened before network, it has to be reshaped back to input shape.
        if flattened:
            x = keras.layers.Reshape(self.__input_shape)(x)

        decoded_layer = x

        # Compile full model.
        full_model = keras.models.Model(input_layer, decoded_layer)
        self.__compile(full_model)

        # Create encoder model.
        encoder_model = keras.models.Model(input_layer, encoded_layer)

        decoder_input_shape = full_model.get_layer(encoder_layer_name).output_shape
        decoder_input = keras.layers.Input(shape=decoder_input_shape[1:])

        __previous_layer = None

        # Create decoder by replacing the input with one that has encoded dimensions.
        found_decoder = False
        for layer in full_model.layers:
            # Iterate until the decoder is found.
            if layer.name == _get_decoder_layer_name(0): found_decoder = True
            if not found_decoder: continue

            if __previous_layer is not None:
                __previous_layer = layer(__previous_layer)
            else:
                __previous_layer = layer(decoder_input)

        decoder_model = keras.models.Model(decoder_input, __previous_layer)

        return full_model, encoder_model, decoder_model

    def build_and_train(self, return_history=False, **kwargs):
        """Build the Keras model for network, train in and return the Autoencoder object."""
        autoencoder = self.build(**kwargs)
        history = self.train(autoencoder.full_model, **kwargs)
        return autoencoder if not return_history else (autoencoder, history)

    def build(self, *, full_model=None, encoder_model=None, decoder_model=None,
              **kwargs) -> Autoencoder:
        """Build and return the Autoencoder object with *untrained* network, have to call train(
        autoencoder.full_model) later."""
        models_count = sum(x is not None for x in [full_model, encoder_model, decoder_model])
        if models_count == 0:
            full_model, encoder_model, decoder_model = self._build_models()
        elif models_count != 3:
            raise ValueError("If you provide existing models, you have to provide all three")

        built_autoencoder = Autoencoder(full_model, encoder_model, decoder_model)

        return built_autoencoder

    def train(self
              , full_model: keras.models.Model
              , input_data
              , epochs=100
              , validation_data=None
              , validation_split=None
              , batch_size=32
              , **kwargs):
        """Train the given autoencoder model with the builder's configuration."""

        given_input_shape = input_data.shape[1:]
        if given_input_shape != self.__input_shape:
            raise ValueError(f"Input shape {given_input_shape} does not match {self.__input_shape}")

        callbacks = []

        if self.__early_stopping is not None and (
                validation_data is not None or validation_split is not None):
            callbacks.append(keras.callbacks.EarlyStopping(patience=self.__early_stopping))

        return full_model.fit(input_data
                              , input_data
                              , epochs=epochs
                              , batch_size=batch_size
                              , callbacks=callbacks
                              , validation_data=validation_data
                              , validation_split=validation_split
                              , shuffle=True
                              , **kwargs)

    def load_from_files(self, config_file, weight_file):
        """
        Load autoencoder object from files and compile it with builder's settings.
        Names of files will be *config_file_SUFFIX.json* and *weight_file_SUFFIX.h5*.
        See example.

        :param config_file: Config files' main name part, without suffix and extension.
        :param weight_file: Weights files' main name part, without suffix and extension.
        :return: Autoencoder model loaded from files.
        """

        def load_model(config, weigths):
            with open(config, "r") as f:
                model = keras.models.model_from_json(f.read())
            model.load_weights(weigths)
            return model

        full_model = load_model(*self._get_file_names(config_file, weight_file, 'full'))
        encoder_model = load_model(*self._get_file_names(config_file, weight_file, 'encoder'))
        decoder_model = load_model(*self._get_file_names(config_file, weight_file, 'decoder'))

        self.__compile(full_model)

        return Autoencoder(full_model, encoder_model, decoder_model)

    @staticmethod
    def save_to_files(autoencoder: Autoencoder, config_file, weight_file):
        """
        Persist Autoencoder model into six separate files, containing configurations and weights
        for all three networks. Can then be loaded with *load_from_files* or with Keras
        independently.

        :param autoencoder: *Trained* Autoencoder object to save.
        :param config_file: Config files' main name part, without suffix and extension.
        :param weight_file: Weights files' main name part, without suffix and extension.
        """

        def save_model(model, config, weigths):
            with open(config, "w") as f:
                f.write(model.to_json())
            model.save_weights(weigths)

        save_model(autoencoder.full_model
                   , *AutoencoderBuilder._get_file_names(config_file, weight_file, 'full'))
        save_model(autoencoder.encoder_model
                   , *AutoencoderBuilder._get_file_names(config_file, weight_file, 'encoder'))
        save_model(autoencoder.decoder_model
                   , *AutoencoderBuilder._get_file_names(config_file, weight_file, 'decoder'))

    @staticmethod
    def _get_file_names(config_file, weight_file, suffix):
        return f'{config_file}_{suffix}.json', f'{weight_file}_{suffix}.h5'
