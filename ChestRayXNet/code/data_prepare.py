import os
import time

import tensorflow as tf
from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step
from tensorflow.python.platform import tf_logging as logging

from sklearn.metrics import roc_auc_score, roc_curve
import data_preproces
from dataset_utils import read_label_file

slim = tf.contrib.slim

def get_split(split_name, dataset_dir, num_classes, file_pattern, file_pattern_for_counting):
    '''
    INPUTS:
    - split_name(str): 'train' or 'validation'. Used to get the correct data split of tfrecord files
    - dataset_dir(str): the dataset directory where the tfrecord files are located
    - file_pattern(str): the file name structure of the tfrecord files in order to get the correct data
    - file_pattern_for_counting(str): the string name to identify your tfrecord files for counting

    OUTPUTS:
    - dataset (Dataset): A Dataset class object where we can read its various components for easier batch creation later.
    '''

    #check whether the split_name is train or validation or test
    if split_name not in ['train', 'validation', 'test']:
        raise ValueError('The split_name %s is not recognized. Please input either train or validation as the split_name' % (split_name))

    #Create the full path for a general file_pattern to locate the tfrecord_files
    file_pattern_path = os.path.join(dataset_dir, file_pattern % (split_name))

    #Count the total number of examples in all of these shard
    num_samples = 0
    file_pattern_for_counting = file_pattern_for_counting + '_' + split_name
    tfrecords_to_count = [os.path.join(dataset_dir, file) for file in os.listdir(dataset_dir) if file.startswith(file_pattern_for_counting)]
    for tfrecord_file in tfrecords_to_count:
        for record in tf.python_io.tf_record_iterator(tfrecord_file):
            num_samples += 1

    #Create a reader, which must be a TFRecord reader in this case
    reader = tf.TFRecordReader
    #Create the keys_to_features dictionary for the decoder
    keys_to_features = {
      'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
      'image/format': tf.FixedLenFeature((), tf.string, default_value='png'),
      'image/class/label': tf.FixedLenFeature((), tf.string, default_value=''),
    }
    #Create the items_to_handlers dictionary for the decoder.
    items_to_handlers = {
    'image': slim.tfexample_decoder.Image(),
    'label': slim.tfexample_decoder.Tensor('image/class/label'),
    }
    items_to_descriptions = {
        'image': 'A chest image that is used in binary classfication',
        'label': 'A label that is as such -- 0: no certain lesion, 1:have certain lesion'
    }

    #Start to create the decoder
    decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)
    #Create the labels_to_name file
    labels_to_name_dict = {0:'Atelectasis', 1:'Cardiomegaly', 2:'Effusion', 3:'Infiltration',
                           4:'Mass', 5:'Nodule', 6:'Pneumonia', 7:'Pneumothorax', 8:'Consolidation',
                           9:'Edema', 10:'Emphysema', 11:'Fibrosis', 12:'Pleural_Thickening', 13:'Hernia'}
    #Actually create the dataset
    dataset = slim.dataset.Dataset(
        data_sources = file_pattern_path,
        decoder = decoder,
        reader = reader,
        num_readers = 40,
        num_samples = num_samples,
        num_classes = num_classes,
        labels_to_name = labels_to_name_dict,
        items_to_descriptions = items_to_descriptions)
    return dataset

def load_batch(dataset, batch_size, num_classes, height=299, width=299, is_training=True, one_hot=False):
    '''
    Loads a batch for training or validation or test

    INPUTS:
    - dataset(Dataset): a Dataset class object that is created from the get_split function
    - batch_size(int): determines how big of a batch to train
    - height(int): the height of the image to resize to during preprocessing
    - width(int): the width of the image to resize to during preprocessing
    - is_training(bool): to determine whether to perform a training or evaluation preprocessing

    OUTPUTS:
    - images(Tensor): a Tensor of the shape (batch_size, height, width, channels) that contain one batch of images
    - labels(Tensor): the batch's labels with the shape (batch_size,) (requires one_hot_encoding).

    '''
    #First create the data_provider object
    data_provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        num_readers=40,
        shuffle=True,
        common_queue_capacity = 64 + 16 * batch_size,
        common_queue_min = 8 * batch_size)
        # common_queue_min = 64)

    #Obtain the raw image using the get method
    raw_image, label = data_provider.get(['image', 'label'])
    label = tf.string_to_number(tf.string_split([label], delimiter="").values, tf.float32)
    # logging.info('line 177: after convert to string, label shape is: %s, get_shape is : %s' % (tf.shape(label), label.get_shape()))
    label.set_shape([num_classes])
    if one_hot:
        label = tf.cast(label, tf.int32)
        label = tf.one_hot(label, depth=2)
    # logging.info('line 179: after set_shape([14]), label shape is: %s' % tf.shape(label))

    #Perform the correct preprocessing for this image depending if it is training or evaluating
    image = data_preproces.preprocess_image(raw_image, height, width, is_training)

    #As for the raw images, we just do a simple reshape to batch it up
    raw_image = tf.expand_dims(raw_image, 0)
    raw_image = tf.image.resize_nearest_neighbor(raw_image, [height, width])
    raw_image = tf.squeeze(raw_image)

    #Batch up the image by enqueing the tensors internally in a FIFO queue and dequeueing many elements with tf.train.batch.
    images, raw_images, labels = tf.train.shuffle_batch(
        [image, raw_image, label],
        batch_size = batch_size,
        capacity = 16 * batch_size,
        min_after_dequeue=8 * batch_size,
        num_threads = 16,
        allow_smaller_final_batch = True)

    return images, raw_images, labels