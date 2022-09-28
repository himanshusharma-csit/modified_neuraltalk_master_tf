import tensorflow as tf


# This class generates the caption predictions for the incoming test image
class PredictionManager:
    configurations = None
    feature_encoder = None
    index_to_word = None
    word_to_index = None
    decoder = None

    # Initialize the prediction manager
    def __init__(self, configurations, feature_encoder, decoder):
        self.configurations = configurations
        self.feature_encoder = feature_encoder
        self.decoder = decoder
        print(">>> Predict instance initiated...")

    # Predicts the captions for the incoming image feature
    def predict(self, image_feature, vocabulary):
        # Word to index mapping for prediction
        word_to_index = tf.keras.layers.StringLookup(mask_token="", vocabulary=vocabulary)
        # Index to word mapping for prediction
        index_to_word = tf.keras.layers.StringLookup(mask_token="", vocabulary=vocabulary, invert=True)

        # Encode the image features
        image_feature = self.feature_encoder(image_feature)
        # print("Features: ", image_feature[0])
        # Predicted caption result
        result = []
        # Initialize decoder hidden states
        hidden = self.decoder.reset_states(batch_size=1)
        dec_input = tf.expand_dims([word_to_index('startsen')] * 1, 1)
        # Start generating prediction
        for i in range(int(self.configurations["maximum_caption_length"])):
            predictions, hidden = self.decoder(dec_input, image_feature, hidden)
            predicted_index = tf.random.categorical(predictions[0], 1)[0][0].numpy()
            predicted_word = tf.compat.as_text(index_to_word(predicted_index).numpy())
            result.append(predicted_word)
            dec_input = tf.expand_dims([predicted_index], 0)
            if index_to_word(predicted_index) == 'endsen':
                return result[:-1]
        return result


# This method generates a prediction instance with the required configurations and return it
def generate_prediction_manager(configurations=None, encoder=None, decoder=None):
    prediction_manager = PredictionManager(configurations, encoder, decoder)
    # Predict instance initialized successfully, now return
    return prediction_manager
