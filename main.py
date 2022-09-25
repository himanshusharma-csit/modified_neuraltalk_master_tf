import os
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt
from Utils.CaptionLoader import load_captions, process_captions
from Utils.Dataset import split_dataset, load_images, generate_batched_dataset
from Utils.Decoder import generate_decoder
from Utils.FeatureEncoder import generate_feature_encoder
from Utils.ImageEncoder import generate_inception_image_encoder
from Utils.TextDataset import preprocess_text_dataset
from Utils.ModelDetails import ModelDetails, training_step, generate_sparse_categorical_crossentropy_loss, \
    generate_adam_optimizer

# 1. PREPROCESS THE CAPTION DATASET
from Utils.Training import initialize_pipeline_training, load_pipeline_weights

caption_file_path = os.path.abspath('D:\modified_neuraltalk_master_tf\Data\Flickr_8K\captions.txt')
image_directory_path = os.path.abspath('D:\modified_neuraltalk_master_tf\Data\Flickr_8K\Images')
image_features_path = os.path.abspath('D:\modified_neuraltalk_master_tf\Features\Flickr_8K')
model_saving_path = os.path.abspath('D:\modified_neuraltalk_master_tf')
encoder_saving_path = os.path.abspath('D:\modified_neuraltalk_master_tf\FeatureEncoderCheckpoints\Encoder.ckpt')
decoder_saving_path = os.path.abspath('D:\modified_neuraltalk_master_tf\DecoderCheckpoints\Decoder.ckpt')

# The hyperparameters for batching the dataset
# Batch size: The number of training samples in each training sample
batch_size = 64
# Buffer size: The sampling buffer size from where the batch is sampled
buffer_size = 1000
# Multimodal embedding size of model
embedding_size = 256
# Number of hidden units in the decoder
hidden_units = 256

# Log all the information about whether the operation is using CPU or GPU
# tf.debugging.set_log_device_placement(True)

# Read the captions of the flickr 8k dataset into the main memory
# The captions contains labeled dataset where each image id is associated with five captions. See example below
# (1000268201_693b08cb0e.jpg,A child in a pink dress is climbing up a set of stairs in an entry way .)
unprocessed_image_captions = load_captions(caption_file_path)

# The captions file as read above is unpacked version as every caption has its own image identifier
# Now, we associate five captions of single image with it using a dictionary where the structure will be as follows
# dict{image_id: [caption_1, caption_2, caption_3, caption_4, caption_5], image_id: [caption/s]}
processed_image_captions = process_captions(unprocessed_image_captions)

# Create training and testing image split
training_image_names, testing_image_names = split_dataset(list(processed_image_captions.keys()))

# 2. PREPROCESS THE IMAGES (FEATURE EXTRACTION)
# Generate the image encoding model
image_encoder = generate_inception_image_encoder()

# Load the training images into the main memory of the predefined input size
with tf.device('/CPU:0'):
    training_image_dataset = load_images(image_directory_path, training_image_names, image_encoder)
    # testing_image_names = load_images(image_directory_path, testing_image_names, image_encoder)

# Extract the features of the training images
# Image features have been extracted for now, so commenting this code section
# extract_features(image_features_path, image_encoder, training_image_dataset)

# 3. PREPROCESS THE CAPTIONS DATASET (CAPTION PREPROCESSING)
# Preprocess the text by converting  loit into lowercase and appending start and end tokens to it
# Then generate the labeled dataset in the form of X and Y for our problem
train_X, train_Y, text_vectorizer, maximum_caption_length = preprocess_text_dataset(image_features_path, training_image_names,
                                                            processed_image_captions)

# 4. DATASET ABSTRACTION (GENERATE A TF.DATASET VERSION OF DATA)
image_captioning_dataset = generate_batched_dataset(batch_size, buffer_size, train_X, train_Y)

# 5. GENERATE THE IMAGE ENCODER FOR ENCODING IMAGE FEATURES TO A MULTIMODAL EMBEDDING
feature_encoder = generate_feature_encoder(batch_size, embedding_size)
tf.keras.models.save_model(feature_encoder, filepath=model_saving_path)

# 6. GENERATE THE GRU BASED DECODER FOR THE CAPTIONS
decoder = generate_decoder(hidden_units, embedding_size, text_vectorizer.vocabulary_size(), maximum_caption_length)
# tf.keras.models.save_model(decoder)

# 7. GENERATE THE OPTIMIZERS AND LOSS FUNCTION
model_manager = ModelDetails(loss=generate_sparse_categorical_crossentropy_loss(),
                             optimizer=generate_adam_optimizer(),
                             batch_size=batch_size,
                             training_sample_count=len(train_X),
                             encoder_path=encoder_saving_path,
                             decoder_path=decoder_saving_path
                             )

#Initialize the model weights with the checkpoints
# feature_encoder, decoder = load_pipeline_weights(model_manager, feature_encoder, decoder)

loss_plot = initialize_pipeline_training(image_captioning_dataset, feature_encoder, decoder, text_vectorizer, model_manager)

plt.plot(loss_plot)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Plot')
plt.show()

print('Done')
