import os.path
import tensorflow as tf


class Dataset:
    def __init__(self):
        print('Successfully Initialized Caption Loader')


# This method returns a list of testing and training samples
def split_dataset(image_names):
    # Randomize the data
    tf.random.shuffle(image_names)
    # Find the total number of images and split the data into the ratio of 80:20 for training and testing
    data_length = len(image_names)
    # Use 80% of images for training
    training_data_size = int(data_length * 0.8)

    return image_names[:training_data_size], image_names[training_data_size:]


# This method loads the list of images into the main memory and returns both the datasets namely training_image_names and training_images
def load_images(directory_name, training_image_names, image_encoder):
    # Read the input specifications of the image encoder
    _, x, y, channels = image_encoder.input_shape

    # Create a tf.dataset of the training_image_names for loading the actual images into main memory
    training_image_names = tf.data.TextLineDataset.from_tensor_slices(training_image_names)

    # Now, map this training_image_name tf.dataset with the actual images and generate their new tf.dataset as well
    training_images = list(map(lambda image_name: load_image(directory_name, image_name.decode(), x, y, channels), training_image_names.as_numpy_iterator()))
    training_image_dataset = tf.data.Dataset.from_tensor_slices(training_images)

    # Once both the datasets have been prepared, return the dataset
    return training_image_names, training_image_dataset


# This method loads the image referenced by its path into the main memory
def load_image(directory_name, image_name, x, y, channels):
    # Generate the absolute image path
    absolute_image_path = os.path.join(directory_name, str(image_name))

    # Read the image file into the main memory
    image_file = tf.io.read_file(absolute_image_path)

    # Convert the image file into tensors of three RGB channel format
    image = tf.image.decode_image(image_file, channels=channels)

    # Resize the image to the
    return tf.image.resize(image, (x, y))


# This method extracts the features of the training images using the image encoder model
def extract_features(image_encoder, training_images):
    feature_list = list(training_images.map(lambda image: tf.keras.applications.inception_v3.preprocess_input(image)))

    # Create a tf.Dataset for the feature list and return
    return tf.data.Dataset.from_tensor_slices(feature_list)
