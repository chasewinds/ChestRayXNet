from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import control_flow_ops

def preprocess_for_train(image, height, width, scope=None):
  with tf.name_scope(scope, 'distort_image', [image, height, width]):
    if image.dtype != tf.float32:
      image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    image.set_shape([None, None, 3])
    if height and width:
      # Resize the image to the specified height and width.
      image = tf.expand_dims(image, 0)
      image = tf.image.resize_images(image, [height, width],
                                     align_corners=True) ## remember in original this option is False
      image = tf.squeeze(image, [0])
    # Randomly flip the image horizontally.
    distorted_image = tf.image.random_flip_left_right(image)

    base_color_scale = tf.constant([255.0])
    image = tf.subtract(distorted_image, base_color_scale)
    # the mean and std of ImageNet is as fellow:
    total_mean = tf.constant([0.485, 0.456, 0.406])
    std = tf.constant([0.229, 0.224, 0.225])
    # in chest x ray dataset, the total mean and std is as fellow:
    # total_mean = tf.constant([126.973])
    # std = tf.constant([66.0])
    image = tf.subtract(image, total_mean)
    image = tf.div(image, std)
    return image


def preprocess_for_eval(image, height, width, central_fraction=0.875, scope=None):
  with tf.name_scope(scope, 'eval_image', [image, height, width]):
    if image.dtype != tf.float32:
      image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    image.set_shape([None, None, 3])
    if height and width:
      # Resize the image to the specified height and width.
      image = tf.expand_dims(image, 0)
      image = tf.image.resize_images(image, [height, width],
                                     align_corners=True) ## remember in original this option is False
      image = tf.squeeze(image, [0])
    ## keep the mean and std the seem as train:
    base_color_scale = tf.constant([255.0])
    image = tf.subtract(image, base_color_scale)
    # the mean and std of ImageNet is as fellow:
    total_mean = tf.constant([0.485, 0.456, 0.406])
    std = tf.constant([0.229, 0.224, 0.225])
    image = tf.subtract(image, total_mean)
    image = tf.div(image, std)
    return image


def preprocess_image(image, height, width,
                     is_training=False,
                     bbox=None,
                     fast_mode=True):

  # image = tf.image.resize_image_with_crop_or_pad(image, height, width)
  if is_training:
    return preprocess_for_train(image, height, width)
  else:
    return preprocess_for_eval(image, height, width)
