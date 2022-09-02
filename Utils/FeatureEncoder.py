import tensorflow as tf

import keras


# This class encodes the numpy extracted features into a multimodal hidden embedding
class FeatureEncoder:
    embedding_layer = None

    # Generates a fully connected layer of the user defined embedding size
    def __init__(self, embedding_size):
        self.embedding_layer = tf.keras.layers.Dense(embedding_size)

    # Calculate the embedding of the given image feature
    def generate_image_encoding(self, image_feature):
        # Calculate the image encoding using the image feature
        image_encoding = self.embedding_layer(image_feature)
        # Apply the activation function over the calculated encoding
        image_encoding = tf.nn.relu(image_encoding)
        # Return the activation
        return image_encoding


# This method returns an instance of the feature encoder based on the user defined image encoding size
def generate_feature_encoder(embedding_size):
    # Initialize the feature encoder class using user defined embedding size
    feature_encoder = FeatureEncoder(embedding_size)
    # Return the generated feature encoder
    return feature_encoder
