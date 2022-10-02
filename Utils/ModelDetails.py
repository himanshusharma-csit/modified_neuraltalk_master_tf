import tensorflow as tf
from Utils.Attention import generate_attention_instance


# This class deals with all the model details of the image captioning pipeline
class ModelDetails:
    loss = None
    optimizer = None
    batch_size = None
    training_sample_count = None
    encoder_path = None
    decoder_path = None
    attention_path = None
    embedding_path = None

    def __init__(self, loss=None, optimizer=None, batch_size=None, hidden_units=None, training_sample_count=None,
                 encoder_path=None, decoder_path=None, attention_path=None, embedding_path=None):
        self.loss = loss
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.hidden_units = hidden_units
        self.training_sample_count = training_sample_count
        self.encoder_path = encoder_path
        self.decoder_path = decoder_path
        self.attention_path = attention_path
        self.embedding_path = embedding_path


# This method generates the sparse categorical cross-entropy loss for our image captioning pipeline
def generate_sparse_categorical_cross_entropy_loss():
    return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')


# This method generates the Adam's optimizer for image captioning pipeline
def generate_adam_optimizer():
    return tf.keras.optimizers.Adam()


# This method calculates the loss between real and predicted values
def calculate_loss(loss_instance, real_value, predicted_value):
    mask = tf.math.logical_not(tf.math.equal(real_value, 0))  # Removing the padding 0s from getting modified now
    loss_ = loss_instance(real_value, predicted_value)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)


def training_step(image_tensor=None, target=None, feature_encoder=None,
                  decoder=None, tokenizer=None, model_manager=None):
    # Word to index mapping for training
    word_to_index = tf.keras.layers.StringLookup(mask_token="", vocabulary=tokenizer.get_vocabulary())

    # Initialize the loss to 0
    loss = 0

    # Initialize the hidden states of the decoder
    hidden = decoder.reset_states(target.shape[0])
    dec_input = tf.expand_dims([word_to_index('startsen')] * target.shape[0], 1)

    # Initialize the Gradient Operation Audit
    with tf.GradientTape() as tape:
        # Encode the image features into the multimodal embeddings
        batch_features = feature_encoder(image_tensor)
        batch_features = tf.reshape(batch_features, shape=(batch_features.shape[0],
                                                           batch_features.shape[1] * batch_features.shape[2],
                                                           batch_features.shape[3]))

        # Calculate the attention context for the batch of images
        attention_context, _ = decoder.attention(batch_features=batch_features, hidden_activations=hidden)

        for i in range(1, target.shape[1]):
            # Passing the attention features through the decoder
            predictions, hidden = decoder(dec_input, attention_context, hidden)
            # Now, calculate the loss by comparing current prediction with the original index
            loss += calculate_loss(model_manager.loss, target[:, i], predictions)
            # Update the new decoder input now
            dec_input = tf.expand_dims(target[:, i], 1)

        # Calculate the total batch loss
        total_loss = (loss / int(target.shape[1]))
        # Calculate total trainable parameters
        trainable_variables = feature_encoder.trainable_variables + decoder.trainable_variables
        # Calculate and apply gradient adjust over the trainable variables
        gradients = tape.gradient(loss, trainable_variables)
        model_manager.optimizer.apply_gradients(zip(gradients, trainable_variables))
        # Return the total batch loss
        return loss, total_loss
