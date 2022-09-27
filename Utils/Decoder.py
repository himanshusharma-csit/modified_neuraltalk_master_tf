import numpy as np
import numpy.random
import tensorflow as tf


# This class deals with the decoder that converts input word indices into a multimodal vector and then generates their decoding
# After passing them through a time-dependent neural model
class Decoder(tf.keras.Model):
    hidden_units = None
    maximum_caption_length = None
    batch_size = None

    def __init__(self, batch_size, hidden_units, embedding_size, vocabulary_size, maximum_caption_length):
        super(Decoder, self).__init__()
        self.hidden_units = hidden_units
        self.maximum_caption_length = maximum_caption_length
        # Create an input layer for the decoder
        self.input_layer = tf.keras.layers.InputLayer(input_shape=(batch_size, 1))
        # Pass the input layer indices through the embedding layer
        self.embedding = tf.keras.layers.Embedding(input_dim=vocabulary_size,
                                                   output_dim=embedding_size,
                                                   input_length=maximum_caption_length)
        # All the required parameters have been obtained, now build the decoder model
        self.gru = tf.keras.layers.GRU(self.hidden_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        # Collect the recurrent data using a hidden layer
        self.hidden_layer = tf.keras.layers.Dense(self.hidden_units)
        # Now, generate the index predictions on the word vocabulary
        self.output_layer = tf.keras.layers.Dense(vocabulary_size)
        print(">>> Decoder generation complete...")

    # Executes the decoder over a given sequential data
    def call(self, decoder_input, image_features, hidden):
        # Pass the input word index through the input layer
        input_context = self.input_layer(decoder_input)
        # Generate the word embeddings for the input word indices
        word_embeddings = self.embedding(input_context)
        # Both the image and word is now having a multimodal representation, now concat both of them to feed to the network
        x = tf.concat([image_features, word_embeddings], axis=-1)
        # Compute the recurrent input of the input multimodal representation
        output, state = self.gru(x)
        # Pass the recurrent generated output through the hidden layer
        x = self.hidden_layer(output)
        # Pass the hidden layer output through the vocabulary sized dense layer
        x = self.output_layer(x)
        # Return the output probabilities of vocabulary along with GRU's internal states for this input
        return x, state

    # Resets a given tensor for upcoming processing
    def reset_states(self, batch_size):
        return tf.zeros((batch_size, self.hidden_units))

    # Initialize the model training layers and then load the pre-saved weights
    def load_model_weights(self, filepath=None, batch_size=None, embedding_size=None):
        # Initialize the instance of decoder with random weights
        # Hidden shape: (batch_size, embedding_size)
        hidden = self.reset_states(batch_size=batch_size)
        # Decoder_input: (batch_size, 1) {<startsen> index is 2, so using it here. Even though we can initialize with any index}
        dec_input = tf.expand_dims([2] * batch_size, 1)
        # Features: (batch_size, 1, embedding_size)
        features = tf.random.normal(shape=(batch_size, 1, embedding_size))
        # Now, initialize the decoder
        self(dec_input, features, hidden)
        # Now, load the pre-saved weights to the model
        self.load_weights(filepath=filepath)


# This method generates a decoder based on user defined configurations
def generate_decoder(batch_size=None,
                     hidden_units=None,
                     embedding_size=None,
                     vocabulary_size=None,
                     maximum_caption_length=None):
    # Create a decoder instance with the required user defined configurations
    decoder = Decoder(batch_size, hidden_units, embedding_size, vocabulary_size, maximum_caption_length)
    # Decoder weights have been initialized, now return the instance
    return decoder
