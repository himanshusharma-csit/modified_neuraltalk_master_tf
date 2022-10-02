import json
import os.path
import time
import pickle as pkl
from tqdm import tqdm
import tensorflow as tf


class Dataset:
    def __init__(self):
        print('Successfully Initialized Caption Loader')


# This method returns a list of testing and training samples
def split_dataset(image_names):
    # Randomize the data
    print('>>> Randomizing the dataset...[has been disabled for now]')
    # image_names = tf.random.shuffle(image_names)

    # Find the total number of images and split the data into the ratio of 80:20 for training and testing
    data_length = len(image_names)

    # Use 80% of images for training, 10% for validation and 10% for testing
    training_data_size = int(data_length * 0.8)
    validation_data_size = int(data_length * 0.1)

    # Calculate the splitting index based on the training, validation and testing lengths
    validation_data_start_index, validation_data_end_index = training_data_size, training_data_size + validation_data_size
    testing_data_start_index = validation_data_end_index

    # Return the fist 6472 images as training data and the rest as testing
    print('>>> Dataset splitting completed...')
    return image_names[:training_data_size], \
           image_names[validation_data_start_index:validation_data_end_index], \
           image_names[testing_data_start_index:]


# This method loads the list of images into the main memory and returns both the datasets namely training_image_names and training_images
def load_images(directory_name, training_image_names, input_shape):
    # Read the input specifications of the image encoder
    _, x, y, channels = input_shape

    # Now, map this training_image_name tf.dataset with the actual images and generate their new tf.dataset as well
    # training_image_dataset = list(map(lambda image_name: load_image(directory_name, image_name, x, y, channels), training_image_names.as_numpy_iterator()))
    training_image_dataset = list(map(lambda image_name: load_image(directory_name, image_name, x, y, channels),
                                      tqdm(training_image_names, desc="LOADING IMAGES >>> ", ascii=False, ncols=100)))

    # Once the training images have been retrieved, generate a tf.Dataset for image name and their individual images and return
    print('>>> Images loaded from the user defined directory...')
    # Return the tensor representations of the images
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
    # Normalize the input pixel values into the range of (0-1)
    image = tf.keras.applications.inception_v3.preprocess_input(image)
    # Return the image_name along with the actual image tensor
    return image


# This method extracts the features of the training images using the image encoder model
def extract_image_features(image_encoder, training_image_tensors):
    # Holding 6472 images in the dynamic environment crashes runtime, so batching the training image tensors
    batch_size = 500
    iterations = len(training_image_tensors) // batch_size
    extracted_features = list()
    for index in tqdm(range(iterations + 1)):
        start_index = index * batch_size
        end_index = index * batch_size + batch_size
        start_time = time.time()
        print(f">>> Extraction features of images: {start_index} to {end_index}")
        # Extract the batch features and store them in a new list
        features = list(image_encoder(tf.Variable(training_image_tensors[start_index:end_index])))
        extracted_features = extracted_features + features
        print(f">>> Done extraction for: {start_index} to {end_index} in time: {time.time() - start_time} seconds")
        del features
    # Image features have been extracted, now return
    return extracted_features


# This method generates a batched dataset version of the training image and labels
def generate_batched_dataset(batch_size, buffer_size, train_x, train_y, training_image_features):
    image_feature_dataset = list()
    # Load the extracted features of the images now, we require them for training
    for image_name in tqdm(train_x, desc="BATCHING DATASET >>> ", ascii=False, ncols=100):
        # Load the features of the current image
        image_feature = training_image_features[image_name]
        image_feature_dataset.append(image_feature)

    # Now, generate a new tf.dataset instance from the above list slices
    image_feature_dataset = tf.data.Dataset.from_tensor_slices((image_feature_dataset, train_y))

    # Generate a batched dataset by shuffling the dataset into a user defined buffer size and then filtering the batch from there
    image_feature_dataset = image_feature_dataset.shuffle(buffer_size=buffer_size).batch(batch_size=batch_size)

    # Execute the prefetch function to keep next training samples in memory. This yields optimization
    image_feature_dataset = image_feature_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    # Dataset preparation has been complete, now return
    print('>>> Batched dataset generated...')
    return image_feature_dataset


# This method prepares the testing image dataset by loading the image features into the main memory
def prepare_testing_dataset(image_features_path):
    # Load the image features from the storage
    image_features = load_features(filepath=image_features_path)
    # Return the features
    return image_features


# This method saves the model configuration into an external json file
def save_configurations(filepath=None, configurations=None):
    with open(filepath, "w+") as json_file:
        json.dump(configurations, json_file, indent=6)
        print(">>> Configurations dumped successfully...")


# This method saves the vocabulary into an external file
def save_vocabulary(filepath=None, vocabulary=None):
    with open(filepath, 'wb') as vocabulary_file:
        pkl.dump(vocabulary, vocabulary_file)
        print(">>> Vocabulary dumped successfully...")


# This method loads the vocabulary object from an external file
def load_vocabulary(filepath=None):
    with open(filepath, 'rb') as vocabulary_file:
        vocabulary = pkl.load(vocabulary_file)
        print(">>> Vocabulary loaded successfully...")
        return vocabulary


# This method saves the image features into an external file
def save_features(filepath=None, image_features=None):
    with open(filepath, 'wb') as features_file:
        pkl.dump(image_features, features_file)
        print(">>> Image features dumped successfully...")


# This method loads the image features object from an external file
def load_features(filepath=None):
    with open(filepath, 'rb') as features_file:
        image_features = pkl.load(features_file)
        print(">>> Image features loaded successfully...")
        return image_features


# This method saves the user configuration into a json file for further use
def model_configuration_dictionary(batch_size=None,
                                   buffer_size=None,
                                   embedding_size=None,
                                   hidden_units=None,
                                   image_feature_size=None,
                                   maximum_caption_length=None,
                                   encoder_path=None,
                                   decoder_path=None,
                                   training_image_features_path=None,
                                   validation_image_features_path=None,
                                   testing_image_features_path=None,
                                   image_directory_path=None,
                                   caption_file_path=None,
                                   vocabulary_size=None,
                                   vocabulary_path=None,
                                   embedding_checkpoint_path=None,
                                   attention_checkpoint_path=None):
    # Create a dict with the following received configurations
    configurations = {"batch_size": batch_size,
                      "buffer_size": buffer_size,
                      "embedding_size": embedding_size,
                      "hidden_units": hidden_units,
                      "image_feature_size": image_feature_size,
                      "maximum_caption_length": maximum_caption_length,
                      "encoder_path": encoder_path,
                      "decoder_path": decoder_path,
                      "training_image_features_path": training_image_features_path,
                      "validation_image_features_path": validation_image_features_path,
                      "testing_image_features_path": testing_image_features_path,
                      "image_directory_path": image_directory_path,
                      "caption_file_path": caption_file_path,
                      "vocabulary_size": vocabulary_size,
                      "vocabulary_path": vocabulary_path,
                      "embedding_checkpoint_path": embedding_checkpoint_path,
                      "attention_checkpoint_path": attention_checkpoint_path}
    # Configuration dictionary has been created, now return
    return configurations
