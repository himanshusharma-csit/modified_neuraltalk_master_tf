import os
import json
from PIL import Image
from Utils.Decoder import generate_decoder
from Utils.FeatureEncoder import generate_feature_encoder
from Utils.PredictionManager import generate_prediction_manager
from Utils.CaptionLoader import load_captions, process_captions
from Utils.Dataset import split_dataset, prepare_testing_dataset, load_vocabulary, load_images

# ------------------------------------------------------------------------------------------------------------------
# 0.0 DIRECTORY AND FILE PATHS (FOR LOADING DATASET FILES)
# ------------------------------------------------------------------------------------------------------------------
configuration_directory = os.path.abspath("D:\modified_neuraltalk_master_tf\Configurations\configurations.json")

# ------------------------------------------------------------------------------------------------------------------
# 0.1 LOAD THE CONFIGURATIONS INTO THE DICTIONARY
# ------------------------------------------------------------------------------------------------------------------
with open(configuration_directory, "r") as json_file:
    configurations = json.load(json_file)

# Configurations have been loaded, now unpack the variables for use
embedding_size = int(configurations["embedding_size"])
image_feature_size = int(configurations["image_feature_size"])
batch_size = int(configurations["batch_size"])
hidden_units = int(configurations["hidden_units"])
vocabulary_size = int(configurations["vocabulary_size"])
maximum_caption_length = int(configurations["maximum_caption_length"])
testing_image_features_path = configurations["testing_image_features_path"]
encoder_path = configurations["encoder_path"]
decoder_path = configurations["decoder_path"]
vocabulary_path = configurations["vocabulary_path"]
image_directory_path = configurations["image_directory_path"]
attention_path = configurations["attention_checkpoint_path"]
embedding_path = configurations["embedding_checkpoint_path"]
# ------------------------------------------------------------------------------------------------------------------
# 0.0 (OPTIONAL) INPUT THE IMAGE TO BE CAPTIONED FROM THE USER
# ------------------------------------------------------------------------------------------------------------------
# image_name = input(">>> Enter the absolute image path...")
# testing_images = list(image_name)

# ------------------------------------------------------------------------------------------------------------------
# 1.0 INITIALIZE THE TESTING IMAGE NAMES AND FEATURES
# ------------------------------------------------------------------------------------------------------------------
# Read all the image_names and captions again
unprocessed_image_captions = load_captions(configurations["caption_file_path"])
# Associate the captions to the individual images in a form of dictionary
processed_image_captions = process_captions(unprocessed_image_captions)
# Save the memory and delete unused dynamic objects
del unprocessed_image_captions
# Get the names of all the testing images based on the previous split
_, validation_image_names, testing_image_names = split_dataset(list(processed_image_captions.keys()))
# ------------------------------------------------------------------------------------------------------------------


# ------------------------------------------------------------------------------------------------------------------
# 2.0 LOAD THE PRE-SAVED FEATURES OF THE TESTING IMAGES
# ------------------------------------------------------------------------------------------------------------------
testing_image_features = prepare_testing_dataset(testing_image_features_path)


# ------------------------------------------------------------------------------------------------------------------
# 3.0 INITIALIZE THE PRE-SAVED FEATURE ENCODER AND MULTIMODAL DECODER
# ------------------------------------------------------------------------------------------------------------------
# Initialize the encoder
feature_encoder = generate_feature_encoder(embedding_size=embedding_size, input_size=image_feature_size)
# Initialize the feature encoder with saved weights
feature_encoder.load_model_weights(filepath=encoder_path, batch_size=batch_size, feature_size=image_feature_size)

# Initialize the decoder
decoder = generate_decoder(batch_size=batch_size, hidden_units=hidden_units, embedding_size=embedding_size,
                           vocabulary_size=vocabulary_size, maximum_caption_length=maximum_caption_length)
# Initialize the decoder with saved weights
decoder.load_model_weights(filepath=decoder_path, batch_size=batch_size, embedding_size=embedding_size)
# Initialize the pre-learnt word embeddings
decoder.load_word_embeddings(filepath=embedding_path)
# Initialize the pre-learnt attention module
decoder.attention.load_model_weights(filepath=attention_path, batch_size=batch_size, embedding_size=embedding_size)

# ------------------------------------------------------------------------------------------------------------------
# 4.0 INITIALIZE THE PREDICTION MANAGER WITH ALL THE NECESSARY CONFIGURATIONS AND GENERATE CAPTIONS
# ------------------------------------------------------------------------------------------------------------------
prediction_manger = generate_prediction_manager(configurations, feature_encoder, decoder)
predicted_caption = prediction_manger.predict(testing_image_features[10], load_vocabulary(vocabulary_path))
image_absolute_path = os.path.join(os.path.abspath("D:\modified_neuraltalk_master_tf\Data\Flickr_8K\Images"), testing_image_names[10])
image = Image.open(image_absolute_path)
# image.show()
print(
    '\n------------------------------------------------------------------------------------------------------------------')
print("Original Caption: ", processed_image_captions[testing_image_names[10]])
print("Generated Caption: ", *predicted_caption)
print(
    '\n------------------------------------------------------------------------------------------------------------------')
