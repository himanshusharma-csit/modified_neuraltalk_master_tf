class CaptionLoader:
    def __init__(self):
        print('Sucessfully Initialized Caption Loader')


# This method loads the caption file of the dataset into the main memory and then reads all the text contained in it
def load_captions(caption_path):
    with open(caption_path, 'r') as caption_file:
        captions = caption_file.read()
        return captions


# This method packs the individual captions and associated them with the images they are associated with by creating a dictionary
def process_captions(unprocessed_captions):
    # Create a dictionary for mapping images with captions
    processed_captions = dict()

    # The input file has sentences separated by the newline, use this newline deliminator to split them into the individual lists
    all_caption_sentences = unprocessed_captions.split('\n')
