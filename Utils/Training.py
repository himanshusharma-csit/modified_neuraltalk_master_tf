import time
from tqdm import tqdm
import tensorflow as tf

from Utils.ModelDetails import training_step


class Training:
    def __init__(self):
        print(">>> Training instance initialized...")


def initialize_pipeline_training(image_captioning_dataset, feature_encoder, decoder, text_vectorizer, model_manager):
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
            batch_loss, t_loss = training_step(img_tensor, tf.squeeze(target), feature_encoder, decoder,
                                               text_vectorizer, model_manager)
            total_loss += t_loss

            # if (batch.numpy() % 20) == 0:
            #     average_batch_loss = batch_loss.numpy() / int(target.shape[2])
            #     print(f'\nEpoch {epoch + 1} Batch {batch} Loss {average_batch_loss:.4f}')

        epoch_end_statistics(epoch, start, total_loss / num_steps)

        # Epoch has been completed, now save the weights of the model...  to prevent any failure during training
        save_pipeline_weights(model_details=model_manager, feature_encoder=feature_encoder, decoder=decoder)

    # storing the epoch end loss value to plot later
    loss_plot.append(total_loss / num_steps)

    # print(f'Epoch {epoch + 1} Loss {total_loss / num_steps:.6f}')
    # print(f'Time taken for 1 epoch {time.time() - start:.2f} sec\n')

    return loss_plot


def model_training_message():
    print(
        '\n\n********************************************************************************************************************')
    print('                                            MODEL TRAINING INITIALIZED')
    print(
        '********************************************************************************************************************')


def epoch_start_statistics(epoch, epoch_start_time):
    print(
        '********************************************************************************************************************')
    print('\n#######################################################')
    print('>>>> EPOCH NUMBER:', epoch, '\nSUCCESSFULLY INITIALIZED AT:', time.ctime(epoch_start_time))
    print('#######################################################')
    print(
        '------------------------------------------------------------------------------------------------------------------')


def epoch_end_statistics(epoch, epoch_start_time, epoch_error):
    epoch_end_time = time.time()
    print(
        '********************************************************************************************************************')
    print(
        '------------------------------------------------------------------------------------------------------------------')
    print('>>>> EPOCH NUMBER:', epoch, '\nSUCCESSFULLY COMPLETED AT:',
          time.ctime(epoch_end_time), 'IN',
          (epoch_end_time - epoch_start_time), 'seconds', '\n >>>TOTAL LOSS: ', epoch_error.numpy(), )
    print(
        '------------------------------------------------------------------------------------------------------------------')
    print(
        '********************************************************************************************************************')


def save_pipeline_weights(model_details, feature_encoder, decoder):
    feature_encoder.save_weights(model_details.encoder_path)
    decoder.save_weights(model_details.decoder_path)


def load_pipeline_weights(model_details, feature_encoder, decoder):
    feature_encoder.load_weights('D:\modified_neuraltalk_master_tf\FeatureEncoderCheckpoints\Encoder.ckpt')
    decoder.load_weights('D:\modified_neuraltalk_master_tf\DecoderCheckpoints\Decoder.ckpt')
    return feature_encoder, decoder
