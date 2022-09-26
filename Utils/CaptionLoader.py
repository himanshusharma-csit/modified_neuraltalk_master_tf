from tqdm import tqdm


class CaptionLoader:
    def __init__(self):
        print('Sucessfully Initialized Caption Loader')


# This method loads the caption file of the dataset into the main memory and then reads all the text contained in it
def load_captions(caption_path):
    with open(caption_path, 'r') as caption_file:
        captions = caption_file.read()
        print('>>> Dataset loading completed...')
        return captions


# This method packs the individual captions and associated them with the images they are associated with by creating a dictionary
def process_captions(unprocessed_captions):
    # Create a dictionary for mapping images with captions
    processed_captions = dict()

    # The input file has sentences separated by the newline, use this newline deliminator to split them into the individual lists
    # The first line is the (image_name, caption) text only, so we remove it from the all caption list by starting the index from 1
    image_caption_sentences = unprocessed_captions.split('\n')[1:]

    # Now, iterate over image_caption_sentences, and split the sentences into (image_name, caption) format which was initially a single image_caption_sentence
    # Separate by a comma, so we use this comma as the splitting condition
    for image_caption_sentence in tqdm(image_caption_sentences, desc="PROCESSING CAPTIONS >>>", ascii=False, ncols=100):
        # There are chances that comma(,) is included into the captions as well, so setting the split on the first comma only.
        # Rest all commas needs to be ignored as they are part of the caption
        image_name, caption = image_caption_sentence.split(sep=',', maxsplit=1)

        # Now, check if the image_name is already in the dictionary, if yes, append the new caption into the caption list
        # for the image_name into the dictionary
        if image_name in processed_captions:
            current_image_captions = processed_captions[image_name]
            current_image_captions.append(caption)

        # if not, append a new entry into the dictionary
        else:
            processed_captions[image_name] = [caption]

    print('>>> Caption processing completed...')
    return processed_captions
