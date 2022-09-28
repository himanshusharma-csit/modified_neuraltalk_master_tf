import os
import tensorflow as tf


class ImageEncoder:
    def __init__(self):
        print('Successfully Initialized Caption Loader')


# This method generates an image encoder for the image captioning pipeline
def generate_inception_feature_extractor(image_encoder_path):
    # Load the preexisting cnn model
    inceptionv3 = tf.keras.applications.inception_v3.InceptionV3(include_top=False,
                                                                 weights='imagenet',
                                                                 input_shape=(224, 224, 3),
                                                                 pooling='avg')

    # Now, generate a new encoder on top of preexisting model layers
    # The input layer of the existing model acts as the input layer for our encoder too
    input_layers = inceptionv3.input
    # The output layer of our model however varies as we are not using the last layer but the second last layer to extract features
    output_layer = inceptionv3.layers[-1].output

    # Now generate a new image encoder using the above input and hidden layers information
    image_encoder = tf.keras.Model(inputs=input_layers, outputs=output_layer, name="InceptionV3_Feature_Extractor")

    # Keep this information with us, as this feature size will be used to define the InputLayer() of feature encoder
    feature_size = output_layer.shape[1]

    # Load the encoder weights from the directory. If the directory is empty, this means, the model has never been saved, so save it.
    load_image_encoder_weights(image_encoder, image_encoder_path)

    # The model has been generated, now return
    print('>>> Inception v3 feature extractor initialized...')
    return image_encoder, feature_size


# This method either saves the encoder weights or loads them in case the directory is already containing weights
def load_image_encoder_weights(image_encoder, image_encoder_path):
    # Fetch the directory name from the absolute path
    directory_name = image_encoder_path.replace("\\IE_Checkpoint", "")
    # List all the files of the image encoder path
    encoder_directory = os.listdir(directory_name)
    # Check if the weights are already stored there
    if len(encoder_directory) == 0:
        print(">>> No image encoder weights found in the directory... Saving the weights now...")
        # This means the directory is empty, so save the weights here now
        image_encoder.save_weights(filepath=image_encoder_path, overwrite=True)
        print(">>> Image encoder weights saved successfully...")
    else:
        print(">>> Image encoder weights found in the directory... Loading the weights now...")
        image_encoder.load_weights(filepath=image_encoder_path)
        print(">>> Image encoder weights loaded successfully...")

