import tensorflow as tf


# This class encodes the numpy extracted features into a multimodal hidden embedding
# Feature encoder is itself inherited from the tf.keras.Model and thus it itself is a Model instance
class FeatureEncoder(tf.keras.Model):

    # Generates a fully connected layer of the user defined embedding size
    def __init__(self, embedding_dim, input_size):
        super(FeatureEncoder, self).__init__()
        self.input_layer = tf.keras.layers.InputLayer(input_shape=(1, input_size))
        self.output_layer = tf.keras.layers.Dense(embedding_dim)
        print(">>> Feature encoder generation complete...")

    # Calculate the embedding of the given image feature
    def call(self, context_vector):
        # Pass the features through the input layer
        context_vector = self.input_layer(context_vector)
        # Calculate the image encoding using the image feature
        context_vector = self.output_layer(context_vector)
        # Apply the activation function over the calculated encoding
        context_vector = tf.nn.relu(context_vector)
        # Return the activation
        return context_vector

    # Initialize the model training layers and then load the pre-saved weights
    def load_model_weights(self, filepath=None, batch_size=None, feature_size=None):
        # Initialize the instance of feature encoder with random weights
        self(tf.random.normal(shape=(batch_size, 1,feature_size)))  # Batch features are of shape: (batch_size, 1, feature_size) e.g. (128, 1, 2048)
        # Now, load the pre-saved weights to the model
        self.load_weights(filepath=filepath)

# This method returns an instance of the feature encoder based on the user defined image encoding size
def generate_feature_encoder(embedding_size=None, input_size=None):
    # Initialize the feature encoder class using user defined embedding size
    feature_encoder = FeatureEncoder(embedding_size, input_size)
    # Return the initialized generated feature encoder
    return feature_encoder
