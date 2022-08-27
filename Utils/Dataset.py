import numpy as np


class Dataset:
    def __init__(self):
        print('Successfully Initialized Caption Loader')


# This method returns a list of testing and training samples
def split_dataset(image_names):
    # Randomize the data
    np.random.shuffle(image_names)

    # Find the total number of images and split the data into the ratio of 80:20 for training and testing
    data_length = len(image_names)
    # Use 80% of images for training
    training_data_size = int(data_length * 0.8)

    return image_names[:training_data_size], image_names[training_data_size:]
