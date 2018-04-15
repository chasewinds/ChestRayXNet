#!/usr/bin/python

import os
from PIL import Image
import numpy as np
import tensorflow as tf
import math
import sys
import random

def get_filepair(file_path,images_path):
    file_name_pairs = []
    with open(file_path)as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip('\n')
            split_line = line.split(' ')
            file_name = split_line[0]
            labels = split_line[1]

            image_path = os.path.join(images_path,file_name)
            file_name_pairs.append([image_path,labels])
    return file_name_pairs

def int64_feature(values):
  if not isinstance(values, (tuple, list)):
    values = [values]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def bytes_feature(values):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def _get_dataset_filename(dataset_dir, split_name, shard_id,_NUM_SHARDS):
  output_filename = '%s_%05d-of-%05d.tfrecord' % (split_name, shard_id, _NUM_SHARDS)
  return os.path.join(dataset_dir, output_filename)

def image_to_tfexample(image_data, image_format, class_id,height,width):
  return tf.train.Example(features=tf.train.Features(feature={
      'image/encoded': image_data,
      'image/format': bytes_feature(image_format),
      'image/class/label': bytes_feature(class_id),
      'image/height': int64_feature(height),
      'image/width': int64_feature(width),
  }))


def write_tfrecords(dataset_dir,set_name,filename_pairs,num_shards):

    num_per_shard = int(math.ceil(len(filename_pairs) / float(num_shards)))

    for shard_id in range(num_shards):

        output_filename = _get_dataset_filename(dataset_dir, set_name, shard_id,num_shards)

        with tf.python_io.TFRecordWriter(output_filename) as writer:

            start_ndx = shard_id * num_per_shard
            end_ndx = min((shard_id + 1) * num_per_shard, len(filename_pairs))

            for i in range(start_ndx, end_ndx):
                sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                    i + 1, len(filename_pairs), shard_id))
                sys.stdout.flush()

                # Read the filename:
                image_failed = 0
                try:
                    img = np.array(Image.open(filename_pairs[i][0]))

                except:
                    image_failed += 1
                    print('A picture failed reading count: %d' % image_failed)
                    print('The file fail to write is: %s ' % filename_pairs[0])
                label = filename_pairs[i][1]
                height = img.shape[0]
                width = img.shape[1]
                img_raw = img.tobytes()
                img_raw = bytes_feature(img_raw)
                # print convert_width

                example = image_to_tfexample(img_raw, 'png', label,height,width)
                writer.write(example.SerializeToString())


def write(list_path,images_path,save_path,data_set,num_shards):

    filename_pair = get_filepair(list_path, images_path)
    np.random.seed(1413)
    random.shuffle(filename_pair) ## remember to run at the whole dataset ,or the shuffle will make write_record function error if the image doesn't in the dataset dir.
    total_img = len(filename_pair)
    trainset_size = int(total_img*0.8)
    if data_set == 'train':
        filename_pair_process = filename_pair[:trainset_size]
    else:
        filename_pair_process = filename_pair[trainset_size:]
    write_tfrecords(save_path,data_set,filename_pair_process,num_shards)
    print('File %s had been writed' % data_set)


if __name__ == "__main__":
    ## the script would be set at  '~/datasets/NIH-CXR8'
    images_path = os.path.expanduser('~/datasets/NIH-CXR8/images/images')
    list_paths = 'data/list/data.txt'
    save_path = 'data/tfrecords'
    num_shards_train = 20
    num_shards_val = 4
    write(list_paths, images_path, save_path,'train',num_shards_train)
    write(list_paths,images_path,save_path,'val',num_shards_val)


