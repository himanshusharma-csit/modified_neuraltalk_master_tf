import os
import tensorflow as tf
from Utils.Decoder import generate_decoder
from Utils.Visualizations import epoch_vs_loss_plot
from Utils.TextDataset import preprocess_text_dataset
from Utils.Training import initialize_pipeline_training
from Utils.FeatureEncoder import generate_feature_encoder
from Utils.CaptionLoader import load_captions, process_captions
from Utils.ImageEncoder import generate_inception_feature_extractor
from Utils.Dataset import split_dataset, load_images, generate_batched_dataset, model_configuration_dictionary, \
    save_configurations
from Utils.ModelDetails import ModelDetails, generate_sparse_categorical_crossentropy_loss, generate_adam_optimizer

# ------------------------------------------------------------------------------------------------------------------
# 0.0 SETTING THE DIRECTORY AND FILE PATHS (FOR LOADING, SAVING AND STORING DATASET FILES)
# ------------------------------------------------------------------------------------------------------------------
caption_file_path = os.path.abspath('D:\modified_neuraltalk_master_tf\Data\Flickr_8K\captions.txt')
image_directory_path = os.path.abspath('D:\modified_neuraltalk_master_tf\Data\Flickr_8K\Images')
image_features_path = os.path.abspath('D:\modified_neuraltalk_master_tf\Features\Flickr_8K')
encoder_checkpoint_path = os.path.abspath("D:\modified_neuraltalk_master_tf\FeatureEncoderCheckpoints\FE_Checkpoint")
decoder_checkpoint_path = os.path.abspath("D:\modified_neuraltalk_master_tf\DecoderCheckpoints\D_Checkpoint")
configuration_json_path = os.path.abspath("D:\modified_neuraltalk_master_tf\Configurations\configurations.json")

# ------------------------------------------------------------------------------------------------------------------
# 0.1 SETTING THE HYPER-PARAMETERS FOR OUR MODEL
# ------------------------------------------------------------------------------------------------------------------
# Batch size: The number of training samples in each training sample
batch_size = 128
# Buffer size: The sampling buffer size from where the batch is sampled
buffer_size = 1000
# Multimodal embedding size of model
embedding_size = 256
# Number of hidden units in the decoder
hidden_units = 256
# Inception v3 output dimension
inception_model_output_size = None

# ------------------------------------------------------------------------------------------------------------------
# 0.1 (OPTIONAL) LOGGING ALL THE CPU/GPU TASKS OVER CONSOLE
# ------------------------------------------------------------------------------------------------------------------
# Log all the information about whether the operation is using CPU or GPU
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# tf.debugging.set_log_device_placement(True)
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# ------------------------------------------------------------------------------------------------------------------
# 1. PREPROCESS THE CAPTION DATASET
# ------------------------------------------------------------------------------------------------------------------
# Read the captions of the flickr 8k dataset into the main memory
# The captions contains labeled dataset where each image id is associated with five captions. See example below
# (1000268201_693b08cb0e.jpg,A child in a pink dress is climbing up a set of stairs in an entry way .)
unprocessed_image_captions = load_captions(caption_file_path)

# The captions file as read above is unpacked version as every caption has its own image identifier
# Now, we associate five captions of single image with it using a dictionary where the structure will be as follows
# dict{image_id: [caption_1, caption_2, caption_3, caption_4, caption_5], image_id: [caption/s]}
processed_image_captions = process_captions(unprocessed_image_captions)

# Create training and testing image split
training_image_names, validation_image_names, testing_image_names = split_dataset(list(processed_image_captions.keys()))

# ------------------------------------------------------------------------------------------------------------------
# 2. PREPROCESS THE IMAGES (FEATURE EXTRACTION)
# ------------------------------------------------------------------------------------------------------------------
# Generate the image encoding model
image_encoder, inception_model_output_size = generate_inception_feature_extractor()

# Load the training images into the main memory of the predefined input size
with tf.device('/CPU:0'):
    training_image_dataset = load_images(image_directory_path, training_image_names, image_encoder.input_shape)
    # validation_image_dataset = load_images(image_directory_path, validation_image_names, image_encoder.input_shape)
    # testing_image_dataset = load_images(image_directory_path, testing_image_names, image_encoder.input_shape)

# Extract the features of the training images
# Image features have been extracted for now, so commenting this code section
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#     extract_features(image_features_path, image_encoder, training_image_dataset)
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# ------------------------------------------------------------------------------------------------------------------
# 3. PREPROCESS THE CAPTIONS DATASET (CAPTION PREPROCESSING)
# ------------------------------------------------------------------------------------------------------------------
# Preprocess the text by converting  loit into lowercase and appending start and end tokens to it
# Then generate the labeled dataset in the form of X and Y for our problem
train_X, train_Y, text_vectorization, maximum_caption_length = preprocess_text_dataset(image_features_path,
                                                                                       training_image_names,
                                                                                       processed_image_captions)
# ------------------------------------------------------------------------------------------------------------------
# 4. DATASET ABSTRACTION (GENERATE A TF.DATASET VERSION OF DATA)
# ------------------------------------------------------------------------------------------------------------------
image_captioning_dataset = generate_batched_dataset(batch_size, buffer_size, train_X, train_Y)

# ------------------------------------------------------------------------------------------------------------------
# 5. GENERATE THE IMAGE ENCODER FOR ENCODING IMAGE FEATURES TO A MULTIMODAL EMBEDDING
# ------------------------------------------------------------------------------------------------------------------
feature_encoder = generate_feature_encoder(embedding_size=embedding_size, input_size=inception_model_output_size)
# Load the pre-saved training weights if required
# feature_encoder.load_model_weights(filepath=encoder_checkpoint_path,
#                                    batch_size=batch_size,
#                                    feature_size=inception_model_output_size)
# ------------------------------------------------------------------------------------------------------------------
# 6. GENERATE THE GRU BASED DECODER FOR THE CAPTIONS
# ------------------------------------------------------------------------------------------------------------------
decoder = generate_decoder(batch_size=batch_size,
                           hidden_units=hidden_units,
                           embedding_size=embedding_size,
                           vocabulary_size=text_vectorization.vocabulary_size(),
                           maximum_caption_length=maximum_caption_length)
# Load the pre-saved training weights if required
# decoder.load_model_weights(filepath=decoder_checkpoint_path,
#                            batch_size=batch_size,
#                            embedding_size=embedding_size)
# ------------------------------------------------------------------------------------------------------------------
# 7. GENERATE THE OPTIMIZERS AND LOSS FUNCTION AND SAVE ALL THE DETAILS IN THE MODEL_MANAGER
# ------------------------------------------------------------------------------------------------------------------
model_manager = ModelDetails(loss=generate_sparse_categorical_crossentropy_loss(),
                             optimizer=generate_adam_optimizer(),
                             batch_size=batch_size,
                             training_sample_count=len(train_X),
                             encoder_path=encoder_checkpoint_path,
                             decoder_path=decoder_checkpoint_path)

# Initialize the model weights with the checkpoints
# feature_encoder, decoder = load_pipeline_weights(model_manager, feature_encoder, decoder)

# ------------------------------------------------------------------------------------------------------------------
# 8. MODEL TRAINING (TRAIN THE MODEL ON IMAGE FEATURES TO PREDICT THE CAPTIONS AND APPLY BACKPROPAGATION ACCORDINGLY TO ADJUST WEIGHTS)
# ------------------------------------------------------------------------------------------------------------------
loss_plot = initialize_pipeline_training(image_captioning_dataset=image_captioning_dataset,
                                         feature_encoder=feature_encoder,
                                         decoder=decoder,
                                         text_vectorizer=text_vectorization,
                                         model_manager=model_manager)

# ------------------------------------------------------------------------------------------------------------------
# 10. SAVING ALL THE MODEL CONFIGURATIONS FOR FUTURE PREDICT USE
# ------------------------------------------------------------------------------------------------------------------
model_configurations = model_configuration_dictionary(batch_size,
                                                      buffer_size,
                                                      embedding_size,
                                                      hidden_units,
                                                      inception_model_output_size,
                                                      maximum_caption_length,
                                                      encoder_checkpoint_path,
                                                      decoder_checkpoint_path,
                                                      image_features_path,
                                                      image_directory_path)

# Configuration dictionary have been created, now save it to external json file
save_configurations(filepath=configuration_json_path, configurations=model_configurations)

# ------------------------------------------------------------------------------------------------------------------
# 9. VISUALIZING THE TRAINING (PLOT THE EPOCH VS LOSS GRAPH TO VISUALIZE MODEL TRAINING)
# ------------------------------------------------------------------------------------------------------------------
epoch_vs_loss_plot(loss_plot)
