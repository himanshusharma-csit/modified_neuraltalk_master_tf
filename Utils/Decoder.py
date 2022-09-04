import numpy.random
import tensorflow as tf
import keras


# This class deals with the decoder that converts input word indices into a multimodal vector and then generates their decoding
# After passing them through a time-dependent neural model
class Decoder(tf.keras.Model):
    hidden_units = None
    recurrent_model = None
    gru = None
    hidden_layer = None
    output_layer = None

    def __init__(self, hidden_units, embedding_size, vocabulary_size):
        super(Decoder, self).__init__()
        self.hidden_units = hidden_units
        self.embedding = tf.keras.layers.Embedding(vocabulary_size, embedding_size)
        # All the required parameters have been obtained, now build the decoder model
        self.gru = tf.keras.layers.GRU(self.hidden_units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')
        # Collect the recurrent data using a hidden layer
        self.hidden_layer = tf.keras.layers.Dense(self.hidden_units)
        # Now, generate the index predictions on the word vocabulary
        self.output_layer = tf.keras.layers.Dense(vocabulary_size)

        print(">>> Decoder generation complete...")

    # Executes the decoder over a given sequential data
    def call(self, features, hidden):
        features = self.embedding(features)
        # embedding = tf.concat([tf.expand_dims()])
        output, state = self.recurrent_model(features)
        x = self.hidden_layer(output)
        x = self.output_layer(x)
        return x, state

    # Resets a given tensor for upcoming processing
    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.hidden_units))


# This method generates a decoder based on user defined configurations
def generate_decoder(hidden_units, embedding_size, vocabulary_size):
    decoder = Decoder(hidden_units, embedding_size, vocabulary_size)
    # decoder.execute(numpy.random.randint(low=-100, high=100, size=(1, 2048)))
    return decoder
