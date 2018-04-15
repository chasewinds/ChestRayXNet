

import tensorflow as tf
# import argparse
import sys
import os
import math
import logging
import random
import numpy as np

# from aivis.base.fileutil import read_flit


#Create an image reader object for easy reading of the images
class ImageReader(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Initializes function that decodes RGB JPEG data.
    self._decode_png_data = tf.placeholder(dtype=tf.string)
    self._decode_png= tf.image.rgb_to_grayscale(tf.image.decode_png(self._decode_png_data, channels=3)) ## modify

  def read_image_dims(self, sess, image_data):
    image = self.decode_png(sess, image_data)
    return image.shape[0], image.shape[1]

  def decode_png(self, sess, image_data):
    image = sess.run(self._decode_png,
                     feed_dict={self._decode_png_data: image_data})
    # assert len(image.shape) == 3
    # assert image.shape[2] == 3
    return image

def shuffle_and_split(image_path_pair,cut_rate):
    np.random.seed(1314)
    random.shuffle(image_path_pair)
    total_sample = len(image_path_pair)
    start_cut = int(total_sample*cut_rate)
    train_set = image_path_pair[:start_cut]
    val_set = image_path_pair[start_cut:]
    return train_set,val_set

def get_filepair(list_path,images_path,cut_rate):
    image_paths = []
    labels = []
    pair = []
    with open(list_path)as f:

        lines = f.readlines()
        for line in lines:
            line = line.strip('\n')
            split_line = line.split(' ')
            file_name = split_line[0]
            label = split_line[1]
            image_path = os.path.join(images_path,file_name)
            pair.append([image_path,label])
            #print 'before shuffle_and_split'
    train_set,val_set = shuffle_and_split(pair,cut_rate)
    print len(pair)
    print len(train_set)
    print len(val_set)
    return train_set,val_set


def int64_feature(values):
    """Returns a TF-Feature of int64s.

    Args:
    values: A scalar or list of values.

    Returns:
    a TF-Feature.
    """
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def bytes_feature(values):
    """Returns a TF-Feature of bytes.

    Args:
    values: A string.

    Returns:
    a TF-Feature.
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def image_to_tfexample(image_data, image_format, height, width, class_id):
    return tf.train.Example(features=tf.train.Features(feature={
      'image/encoded': bytes_feature(image_data),
      'image/format': bytes_feature(image_format),
      'image/class/label': bytes_feature(class_id),
      'image/height': int64_feature(height),
      'image/width': int64_feature(width),
    }))

#
# def _get_dataset_filename(dataset_dir, split_name, shard_id, tfrecord_filename, _NUM_SHARDS):
#     output_filename = '%s_%s_%05d-of-%05d.tfrecord' % (
#       tfrecord_filename, split_name, shard_id, _NUM_SHARDS)
#     # return os.path.join(dataset_dir, output_filename)
#     return output_filename
#

def convert_dataset(filenames, labels, tf_path, tfrecord_filename, _NUM_SHARDS):
    """Converts the given filenames to a TFRecord dataset.

    Args:
    split_name: The name of the dataset, either 'train' or 'validation'.
    filenames: A list of absolute paths to png or jpg images.
    class_names_to_ids: A dictionary from class names (strings) to ids
    (integers).
    dataset_dir: The directory where the converted datasets are stored.
    """
    num_per_shard = int(math.ceil(len(filenames) / float(_NUM_SHARDS)))
    print(num_per_shard, len(filenames), _NUM_SHARDS)

    with tf.Graph().as_default():
        image_reader = ImageReader()

        with tf.Session('') as sess:

            for shard_id in range(_NUM_SHARDS):
                output_filename = "%s_%05d-of-%05d.tfrecord" % (tfrecord_filename, shard_id+1, _NUM_SHARDS)
                output_filepath = os.path.join(tf_path,output_filename)
                with tf.python_io.TFRecordWriter(output_filepath) as tfrecord_writer:
                    logging.debug("A tfrecord file has been open,and is locate in: %s", output_filepath)
                    start_ndx = shard_id * num_per_shard
                    end_ndx = min((shard_id+1) * num_per_shard, len(filenames))
                    for i in range(start_ndx, end_ndx):
                        sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                            i+1, len(filenames), shard_id))
                        sys.stdout.flush()

                        # Read the filename:
                        image_data = tf.gfile.FastGFile(filenames[i], 'r').read()
                        height, width = image_reader.read_image_dims(sess, image_data)

                        class_id = labels[i]
                        logging.debug("-----The image's name be writen is :%s " % filenames[i])
                        logging.debug("-----The label be writen is :%s" % class_id)
                        example = image_to_tfexample(
                            image_data, 'png', height, width, class_id)
                        tfrecord_writer.write(example.SerializeToString())

    sys.stdout.write('\n')
    sys.stdout.flush()


# def parseargs():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("-l, --listfile", type=str, help="list label file", dest="flist", required=True)
#     parser.add_argument("-d, --data_dir", type=str, help="data dir", dest="data_dir", default="")
#     parser.add_argument("-t, --tfname", type=str, help="tfrecord filename", dest="tfname", required=True)
#     parser.add_argument("-s, --shards", type=int, help="shards", dest="shards", default=1)
#     parser.add_argument("--random_seed", type=int, help="random seed", dest="random_seed", default=1)
#
#     return parser.parse_args()
# def write_into_tfrecord():


def main():
    listfile = 'data/list/data.txt'
    tf_path = 'data/bak'
    # data_dir = os.path.expanduser('/comvol/nfs/datasets/medicine/CXR8/images')
    data_dir = os.path.expanduser('~/datasets/NIH-CXR8/images/images')
    tfname = ['train','val']
    shards = [1, 1] #
    cut_rate = 0.8
    # args = parseargs()
    # print args

    # fnames, labels = read_flist(args.flist, data_dir=args.data_dir)
    # labels_ids = dict(zip(labels, range(len(labels))))

    # args.data_dir = args.data_dir.strip()
    # if len(args.data_dir) >= 0:
    #     fnames = [os.path.join(args.data_dir, x) for x in fnames]
    train_set,val_set = get_filepair(listfile, data_dir,cut_rate)
    ## CREAT train tfrecords
    train_image = [x[0] for x in train_set]
    train_label = [x[1] for x in train_set]
    convert_dataset(train_image[:100],train_label,tf_path,'train', shards[0])
    ## create val tfrecords
    val_image = [x[0] for x in val_set]
    val_label = [x[1] for x in val_set]
    convert_dataset(val_image[:100],val_label,tf_path,'val', shards[1])

    #for iter in range(len(tfname)):
    #    convert_dataset(fnames, labels, tf_path, tfname[iter], shards[iter])
    ## fnaems is file path ,list
    ## labels is label str,list
    ## concentrate,shuffle or not?

if __name__ == "__main__":
    main()

