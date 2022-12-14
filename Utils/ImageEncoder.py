import os
import tensorflow as tf


class ImageEncoder:
    def __init__(self):
        print('Successfully Initialized Caption Loader')


# This method generates an image encoder for the image captioning pipeline
def generate_inception_feature_extractor():
    # Load the preexisting cnn model
    inceptionv3 = tf.keras.applications.inception_v3.InceptionV3(include_top=False, weights='imagenet', input_shape=(224, 224, 3) )

    # Now, generate a new encoder on top of preexisting model layers
    # The input layer of the existing model acts as the input layer for our encoder too
    input_layers = inceptionv3.input
    # The output layer of our model however varies as we are not using the last layer but the second last layer to extract features
    output_layer = inceptionv3.layers[-1].output

    # Now generate a new image encoder using the above input and hidden layers information
    image_encoder = tf.keras.Model(inputs=input_layers, outputs=output_layer, name="InceptionV3_Feature_Extractor")

    # Keep this information with us, as this feature size will be used to define the InputLayer() of feature encoder
    feature_size = output_layer.shape[3]

    # The model has been generated, now return
    print('>>> Inception v3 feature extractor initialized...')
    return image_encoder, feature_size
