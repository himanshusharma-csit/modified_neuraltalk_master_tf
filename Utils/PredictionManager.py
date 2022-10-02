import tensorflow as tf


# This class generates the caption predictions for the incoming test image
from Utils.Attention import Attention


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
        print(">>> Prediction manager instance initiated...")

    # Predicts the captions for the incoming image feature
    def predict(self, image_feature, word_to_index, index_to_word):
        # Encoder the image features using feature encoder
        image_feature = self.feature_encoder(image_feature)
        # Reshape (5, 5, 256) to (25, 256)
        image_feature = tf.reshape(image_feature, shape=(image_feature.shape[0]*image_feature.shape[1], image_feature.shape[2]))
        # Add batch size = 1 as current we are having only one image
        image_feature = tf.expand_dims(image_feature, 0)
        # Predicted caption result
        result = []
        # Initialize decoder hidden states
        hidden = self.decoder.reset_states(batch_size=1)
        dec_input = tf.expand_dims([word_to_index('startsen')] * 1, 1)
        # Start generating prediction
        for i in range(int(self.configurations["maximum_caption_length"])):
            # Calculate the attention context for the batch of images
            attention_context, _ = self.decoder.attention(batch_features=image_feature, hidden_activations=hidden)
            # Generate the predicts for the current word
            predictions, hidden = self.decoder(dec_input, attention_context, hidden)
            # Convert prediction to vocabulary word
            predicted_index = tf.random.categorical(predictions[0], 1)[0][0].numpy()
            predicted_word = tf.compat.as_text(index_to_word(predicted_index).numpy())
            # If we have reached at the end of the prediction, return
            if predicted_word == "endsen":
                return result
            # Append it to the result now
            result.append(predicted_word)
            # Update the decoder input for the next word
            dec_input = tf.expand_dims([predicted_index], 0)
        return result


# This method generates a prediction instance with the required configurations and return it
def generate_prediction_manager(configurations=None, encoder=None, decoder=None):
    prediction_manager = PredictionManager(configurations, encoder, decoder)
    # Predict instance initialized successfully, now return
    return prediction_manager
