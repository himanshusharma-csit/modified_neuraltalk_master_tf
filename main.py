import os
from Utils.CaptionLoader import load_captions, process_captions
from Utils.Dataset import split_dataset
from Utils.ImageEncoder import generate_inception_image_encoder

# 1. PREPROCESS THE CAPTION DATASET
# The address of the caption file that contains all the captions associated with the dataset


caption_filepath = os.path.abspath('Data\Flickr_8K\captions.txt')

# Read the captions of the flickr 8k dataset into the main memory
# The captions contains labeled dataset where each image id is associated with five captions. See example below
# (1000268201_693b08cb0e.jpg,A child in a pink dress is climbing up a set of stairs in an entry way .)
unprocessed_image_captions = load_captions(caption_filepath)

# The captions file as read above is unpacked version as every caption has its own image identifier
# Now, we associate five captions of single image with it using a dictionary where the structure will be as follows
# dict{image_id: [caption_1, caption_2, caption_3, caption_4, caption_5], image_id: [caption/s]}
processed_image_captions = process_captions(unprocessed_image_captions)

# Create training and testing image split
training_images, testing_images = split_dataset(list(processed_image_captions.keys()))

# 2. PREPROCESS THE IMAGES
# Generate the image encoding model
image_encoder = generate_inception_image_encoder()



