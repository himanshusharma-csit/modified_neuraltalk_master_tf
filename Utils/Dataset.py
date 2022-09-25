import os.path
import numpy as np
from tqdm import tqdm
import tensorflow as tf


class Dataset:
    def __init__(self):
        print('Successfully Initialized Caption Loader')


# This method returns a list of testing and training samples
def split_dataset(image_names):
    # Randomize the data
    print('>>> Randomizing the dataset...')
    image_names = tf.random.shuffle(image_names)
    # Find the total number of images and split the data into the ratio of 80:20 for training and testing
    data_length = len(image_names)
    # Use 80% of images for training, 10% for validation and 10% for testing
    training_data_size = int(data_length * 0.01)
    validation_data_size = int(data_length * 0.1)

    # Calculate the splitting index based on the training, validation and testing lengths
    validation_data_start_index, validation_data_end_index = training_data_size, training_data_size+validation_data_size
    testing_data_start_index = validation_data_end_index

    # Return the fist 6472 images as training data and the rest as testing
    print('>>> Dataset splitting completed...')
    return image_names[:training_data_size], \
           image_names[validation_data_start_index:validation_data_end_index], \
           image_names[testing_data_start_index:]


# This method loads the list of images into the main memory and returns both the datasets namely training_image_names and training_images
def load_images(directory_name, training_image_names, image_encoder):
    # Read the input specifications of the image encoder
    _, x, y, channels = image_encoder.input_shape

    # Create a tf.dataset of the training_image_names for loading the actual images into main memory
    training_image_names = tf.data.TextLineDataset.from_tensor_slices(training_image_names)

    # Now, map this training_image_name tf.dataset with the actual images and generate their new tf.dataset as well
    # training_image_dataset = list(map(lambda image_name: load_image(directory_name, image_name, x, y, channels), training_image_names.as_numpy_iterator()))
    training_image_dataset = list(
        map(lambda image_name: load_image(directory_name, image_name.decode(), x, y, channels),
            tqdm(training_image_names.as_numpy_iterator())))

    # Once the training images have been retrieved, generate a tf.Dataset for image name and their individual images and return
    print('>>> Training image retrieved from the user defined directory...')
    return training_image_dataset


# This method loads the image referenced by its path into the main memory
def load_image(directory_name, image_name, x, y, channels):
    # Generate the absolute image path
    absolute_image_path = os.path.join(directory_name, image_name)
    # Read the image file into the main memory
    image_file = tf.io.read_file(absolute_image_path)
    # Convert the image file into tensors of three RGB channel format
    image = tf.image.decode_image(image_file, channels=channels)
    # Resize the image to the required dimension
    image = tf.image.resize(image, (x, y))
    # Return the image_name along with the actual image tensor
    return image_name, image


# This method extracts the features of the training images using the image encoder model
def extract_features(image_features_path, image_encoder, training_image_dataset):
    # Extract all the image tensors from our training image dataset
    training_image_tensors = list(map(lambda image_element: image_element[1], training_image_dataset))

    # Recreate a batched version of tf.Dataset for feature extraction
    # We want to process each image one by one, so setting batch size to 1
    batched_dataset = tf.data.Dataset.from_tensor_slices(training_image_tensors).batch(1)

    # Batched dataset have been created, now map each image tensor of this dataset with the image encoder to extract their features
    for index, image_tensor in enumerate(batched_dataset.as_numpy_iterator()):
        # Extract the features of the incoming image
        image_features = image_encoder(image_tensor)
        # Generate the filename to store image features
        image_features_filename = get_image_features_filename(image_features_path, training_image_dataset[index])
        # Save the image features into the local disk
        np.save(image_features_filename, image_features.numpy())


# This method generates a .npy filename for the image features for saving them into the disk
def get_image_features_filename(image_features_path, image_element):
    # The image element received has both image_name and image_tensor, unpack it
    image_name = image_element[0]
    # Remove the .jpg from the image name and append .npy instead
    image_name = image_name.replace('.jpg', '.npy')
    # Generate the absolute image feature path and return
    return os.path.join(image_features_path, image_name)


# This method generates a batched dataset version of the training image and labels
def generate_batched_dataset(batch_size, buffer_size, train_X, train_Y):
    image_feature_dataset = list()
    # Load the extracted features of the images now, we require them for training
    for image_location in tqdm(train_X):
        # Load the features of the current image
        image_feature = load_features(image_location)
        image_feature_dataset.append(image_feature)

    # Now, generate a new tf.dataset instance from the above list slices
    image_feature_dataset = tf.data.Dataset.from_tensor_slices((image_feature_dataset, train_Y))

    # Generate a batched dataset by shuffling the dataset into a user defined buffer size and then filtering the batch from there
    image_feature_dataset = image_feature_dataset.shuffle(buffer_size=buffer_size).batch(batch_size=batch_size)
    # Execute the prefetch function to keep next training samples in memory. This yields optimization
    image_feature_dataset = image_feature_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    # Dataset preparation has been complete, now return
    print('>>> Batched dataset generated...')
    return image_feature_dataset


# This method loads the features of the given image filename
def load_features(image_name):
    # Replace the .jpg extension with the npy extension for loading the
    image_name = image_name = image_name.replace('.jpg', '.npy')
    image_feature = np.load(image_name)
    return image_feature
