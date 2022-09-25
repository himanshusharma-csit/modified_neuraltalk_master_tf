import numpy.random
import tensorflow as tf
import keras


# This class deals with the decoder that converts input word indices into a multimodal vector and then generates their decoding
# After passing them through a time-dependent neural model
class Decoder(tf.keras.Model):
    hidden_units = None
    maximum_caption_length = None
    recurrent_model = None
    gru = None
    hidden_layer = None
    output_layer = None

    def __init__(self, hidden_units, embedding_size, vocabulary_size, maximum_caption_length):
        super(Decoder, self).__init__()
        self.hidden_units = hidden_units
        self.maximum_caption_length = maximum_caption_length
        self.embedding = tf.keras.layers.Embedding(input_dim=vocabulary_size, output_dim=embedding_size, input_length=maximum_caption_length)
        # All the required parameters have been obtained, now build the decoder model
        self.gru = tf.keras.layers.GRU(self.hidden_units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')
        # Collect the recurrent data using a hidden layer
        self.hidden_layer = tf.keras.layers.Dense(self.hidden_units)
        # Now, generate the index predictions on the word vocabulary
        self.output_layer = tf.keras.layers.Dense(vocabulary_size)
        print(">>> Decoder generation complete...")

    # Executes the decoder over a given sequential data
    def call(self, decoder_input, image_features, hidden):
        # Generate the word embeddings for the input indices
        word_embeddings = self.embedding(decoder_input)
        # Concat both the multimodalities
        # x = tf.concat([tf.expand_dims(image_features, 1), word_embeddings], axis=-1)
        x = tf.concat([image_features, word_embeddings], axis=-1)
        # x = tf.expand_dims(x, 1)
        output, state = self.gru(x)
        x = self.hidden_layer(output)
        x = self.output_layer(x)
        return x, state

    # Resets a given tensor for upcoming processing
    def reset_states(self, batch_size):
        return tf.zeros((batch_size, self.hidden_units))


# This method generates a decoder based on user defined configurations
def generate_decoder(hidden_units, embedding_size, vocabulary_size, maximum_caption_length):
    decoder = Decoder(hidden_units, embedding_size, vocabulary_size, maximum_caption_length)
    # decoder.execute(numpy.random.randint(low=-100, high=100, size=(1, 2048)))
    return decoder
