import tensorflow as tf


# This class implements the semantic attention for the image captioning pipeline while training and prediction
class Attention(tf.keras.Model):
    def __init__(self, hidden_units=None):
        super(Attention, self).__init__()
        print(">>> Attention instance initialized...")
        self.hidden_one = tf.keras.layers.Dense(hidden_units)
        self.hidden_two = tf.keras.layers.Dense(hidden_units)
        self.attention_output = tf.keras.layers.Dense(1)

    def call(self, batch_features=None, hidden_activations=None):
        hidden_with_time_axis = tf.expand_dims(hidden_activations, 1)
        # Pass the (batch_size, 25*256) through the first hidden layer
        attention_activations = tf.nn.tanh(self.hidden_one(batch_features) + self.hidden_two(hidden_with_time_axis))
        score = self.attention_output(attention_activations)
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * batch_features
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return tf.expand_dims(context_vector, axis=1), attention_weights

    # This method load the pre-saved model weights into existing attention instance
    def load_model_weights(self, filepath=None, batch_size=None, embedding_size=None):
        # Initialize the instance of batch features
        batch_features = tf.random.normal(shape=(batch_size, 25, embedding_size))
        # Initialize the instance of hidden context
        hidden = tf.random.normal(shape=(batch_size, 1, embedding_size))
        # Initialize the attention module with random weights
        self(batch_features=batch_features, hidden_activations=hidden)
        # Now, load the pre-saved weights to the model
        self.load_weights(filepath=filepath)


# This method generates an instance of attention class based on the user configurations and returns it
def generate_attention_instance(hidden_units=None):
    attention = Attention(hidden_units=hidden_units)
    # Attention instance initiated, now return
    return attention
