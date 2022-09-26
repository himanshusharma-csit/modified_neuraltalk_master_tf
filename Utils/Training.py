import time
import absl.logging
from tqdm import tqdm
import tensorflow as tf
from Utils.ModelDetails import training_step

# To avoid displaying the save format warning everytime, we, verbose the logging here
absl.logging.set_verbosity(absl.logging.ERROR)

class Training:
    def __init__(self):
        print(">>> Training instance initialized...")


def initialize_pipeline_training(image_captioning_dataset=None,
                                 feature_encoder=None,
                                 decoder=None,
                                 text_vectorizer=None,
                                 model_manager=None):
    loss_plot = []
    start_epoch = 0
    EPOCHS = 5
    num_steps = model_manager.training_sample_count // model_manager.batch_size
    model_training_message()

    for epoch in range(start_epoch, EPOCHS):
        start = time.time()
        total_loss = 0

        epoch_start_statistics(epoch, start)
        for (batch, (img_tensor, target)) in tqdm(tf.data.Dataset.enumerate(image_captioning_dataset), ncols=100):
            # For now target is (64,1,max_size), so we squash the 1 dimension.
            # This is a bit of hack... not happy about it.
            batch_loss, t_loss = training_step(image_tensor=img_tensor,
                                               target=tf.squeeze(target),
                                               feature_encoder=feature_encoder,
                                               decoder=decoder,
                                               tokenizer=text_vectorizer,
                                               model_manager=model_manager)
            # Add the batch loss to the overall total loss
            total_loss += t_loss
        # Save the encoder and decoder model states after every epoch
        feature_encoder.save(filepath=model_manager.encoder_path, overwrite=True, save_format="tf")
        decoder.save(filepath=model_manager.decoder_path, overwrite=True, save_format="tf")
        # Print the end epoch statistics
        epoch_end_statistics(epoch, start, total_loss / num_steps)
    # Storing the epoch end loss value to plot later
    loss_plot.append(total_loss / num_steps)
    # Return the loss_plot values to the main
    return loss_plot


# Creates a Model training initializer header
def model_training_message():
    print(
        '\n\n********************************************************************************************************************')
    print('                                            MODEL TRAINING INITIALIZED')
    print(
        '********************************************************************************************************************')


# Creates a start epoch header
def epoch_start_statistics(epoch, epoch_start_time):
    print('\n********************************************************************************************************************')
    print('\n####################################################################################################################')
    print('>>>> EPOCH NUMBER:', epoch, '\nSUCCESSFULLY INITIALIZED AT:', time.ctime(epoch_start_time))
    print('#####################################################################################################################')
    print(
        '------------------------------------------------------------------------------------------------------------------')


# Creates an end epoch header
def epoch_end_statistics(epoch, epoch_start_time, epoch_error):
    epoch_end_time = time.time()
    print(
        '\n------------------------------------------------------------------------------------------------------------------')
    print('>>>> EPOCH NUMBER:', epoch, '\nSUCCESSFULLY COMPLETED AT:',
          time.ctime(epoch_end_time), 'IN',
          (epoch_end_time - epoch_start_time), 'seconds', '\n >>>TOTAL LOSS: ', epoch_error.numpy(), )
    print(
        '------------------------------------------------------------------------------------------------------------------')
