import os
from time import strftime, gmtime

import tensorflow as tf
import pickle as pkl


# This class encodes the numpy extracted features into a multimodal hidden embedding
# Feature encoder is itself inherited from the tf.keras.Model and thus it itself is a model kind
class FeatureEncoder(tf.keras.Model):
    hidden_layer = None

    # Generates a fully connected layer of the user defined embedding size
    def __init__(self, embedding_size):
        super(FeatureEncoder, self).__init__()
        self.hidden_layer = tf.keras.layers.Dense(embedding_size)

    # Calculate the embedding of the given image feature
    def call(self, image_feature):
        # Calculate the image encoding using the image feature
        image_encoding = self.hidden_layer(image_feature)
        # Apply the activation function over the calculated encoding
        image_encoding = tf.nn.relu(image_encoding)
        # Return the activation
        return image_encoding

    # Saves the state of our model for future use
    # Save the python instance into an external file for future visualization pruposes
    def save_model(self, model_saving_directory):
        # Epoch number to be inserted as a name of the json file
        file_name = 'FeatureEncoder'

        # Time stamp of the completion to be added to the file name
        time_stamp = strftime("%d_%b_%H_%M_%S", gmtime())

        # Generate the overall name of the file
        file_name = os.path.join(model_saving_directory + file_name + time_stamp + '.json')

        # Create a new json file with the above specified filename and dump the history there
        with open(file_name, "wb") as history_file:
            pkl.dump(self, history_file)


# This method returns an instance of the feature encoder based on the user defined image encoding size
def generate_feature_encoder(batch_size, embedding_size):
    # Initialize the feature encoder class using user defined embedding size
    feature_encoder = FeatureEncoder(embedding_size)
    # Build the model for initializing it for saving
    feature_encoder.build(input_shape=(batch_size,1,2048))
    # Return the generated feature encoder
    return feature_encoder
