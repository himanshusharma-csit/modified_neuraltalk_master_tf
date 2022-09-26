import os.path
import re
from tqdm import tqdm
import tensorflow as tf


class TextDataset:
    def __init__(self):
        print('Successfully Initialized Text Dataset')


# This method processes the input caption dataset, removing all the unnecessary details associated with the text
def preprocess_text_dataset(image_features_path, training_image_names, processed_image_captions):
    # First, we fetch all  the captions associated with the training images
    # Then we append <startsen> and <endsen> to the start and end of each caption for making them ready for training
    # The training dataset is a dict{} structure linking image name with the list of captions
    training_dataset = fetch_training_captions(training_image_names, processed_image_captions)

    # Next, we iterate over all the training captions and convert them into lowercase and remove any special symbol contained in them
    # This way, we get a cleaned version of captions
    clean_image_captions(training_dataset)

    # We now have cleaned and processed captions of our training image samples, we now associate index with each of the word
    # That will be contained in our vocabulary using the tf.Keras text_vectorization
    text_vectorization, maximum_caption_length = create_vocabulary_vectorization(training_dataset)

    # Create labelled dataset tuple now
    train_x, train_y = prepare_labeled_dataset(image_features_path, training_dataset, text_vectorization, maximum_caption_length)
    # Dataset has been prepared, now return the training data and labels
    return train_x, train_y, text_vectorization, maximum_caption_length


# This method converts the input caption dataset into lowercase and also removes any special symbols contained in them
def clean_image_captions(processed_image_captions):
    # Iterate over all the training images and fetch the captions associated with them
    for _, captions in tqdm(processed_image_captions.items(), desc="CLEANING CAPTIONS >>> ", ascii=False, ncols=100):
        # This fetches the list of captions associated with each image, now iterate over them as well as access them individually
        for index, caption in enumerate(captions):
            # Remove special symbols from the captions using regular expression, and convert to lower as well
            clean_caption = re.sub(r"[^a-zA-Z0-9]+", ' ', caption.lower())

            # Now, create a word token list for the input caption and remove words of unit length as well
            word_tokens = get_word_token_list(clean_caption)

            # Generate a new cleaned caption from the above received word tokens
            new_caption = ' '.join(word_tokens)

            # Now, replace the old caption text with the new cleaned caption text
            captions[index] = new_caption
    print('>>> Caption cleaning completed...')


# This method generates a word token list for the input caption by removing
def get_word_token_list(clean_caption):
    # Split the clean caption into the word tokens and remove unit length words and alpha-numeric words as well
    word_tokens = list(word for word in clean_caption.split() if (len(word) > 1) and (word.isalpha()))
    # Return the word token list
    return word_tokens


# This method fetches the training captions from the overall caption dataset based on the training image names
def fetch_training_captions(training_image_names, processed_image_captions):
    training_captions = dict()
    # Iterate over the training images and fetch their processed captions
    for image_name in tqdm(training_image_names, desc="LOADING IMAGES >>> ", ascii=False, ncols=100):
        # Fetch the processed captions of the current training image
        image_captions = processed_image_captions[image_name]
        # Append the <start> and <end> token into the captions
        image_captions = append_start_end_tokens(image_captions)
        # Add this caption entry into the training caption dictionary
        training_captions[image_name] = image_captions
    # Return the generated training caption dict
    return training_captions


# This method appends the <start> and <end> token into the training captions
def append_start_end_tokens(image_captions):
    for index, caption in enumerate(image_captions):
        image_captions[index] = 'startsen ' + caption + ' endsen'
    return image_captions


# This method associates the index with each of the word token in our training library
def create_vocabulary_vectorization(training_captions):
    # Flat all the training captions
    flat_training_captions = list(caption for _, captions in training_captions.items() for caption in captions)

    # Calculate the maximum word length of the caption
    maximum_caption_length = max(len(caption.split()) for caption in flat_training_captions)

    # Initialize the TextVectorization layer to convert the words in our vocabulary into the indexes
    # We consider only the first 7000 most frequently words used in the dataset
    max_vocabulary = 7000
    text_vectorization = tf.keras.layers.experimental.preprocessing.TextVectorization(max_tokens=max_vocabulary,
                                                                                   standardize='lower_and_strip_punctuation',
                                                                                   split='whitespace',
                                                                                   ngrams=None,
                                                                                   output_mode='int',
                                                                                   output_sequence_length=maximum_caption_length,
                                                                                   pad_to_max_tokens=True)

    # Adapt the text vectorization layer on the captions
    print(">>> Adapting tokenizer on text...")
    text_vectorization.adapt(flat_training_captions)

    return text_vectorization, maximum_caption_length


# This method prepares the X and Y format training data for the given input datas
def prepare_labeled_dataset(image_features_path, training_dataset, text_vectorizer, maximum_caption_length):
    # Create two empty new lists
    x, y = list(), list()
    # Iterate over each training dataset and fetch the sequential word representations for the captions
    for image_name, captions in tqdm(training_dataset.items(), desc="PREPARING TRAINING  DATASET AND LABELS >>> ", ascii=False, ncols=100):
        absolute_image_path = os.path.join(image_features_path, image_name)
        # Now iterate over the five captions associated with the current image and fetch their encoding
        for caption in captions:
            # Convert the input captions into word indices based on the vocabulary
            caption_word_indices = text_vectorizer([caption])
            # Pad the input caption to the same length as of the maximum length to make the dataset simpler for training
            # caption_word_indices = pad_sequences([caption_word_indices], maxlen=maximum_caption_length, padding='post')[0]
            # Now append each image name and caption indices as a separate training dataset entry
            x.append(absolute_image_path)
            y.append(caption_word_indices)
        # Append complete for one image having 5 associated captions
    # Append complete for the overall training dataset, so return the labeled dataset
    return x, y
