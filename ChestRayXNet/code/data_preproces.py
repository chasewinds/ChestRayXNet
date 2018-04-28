"""Provides utilities to preprocess images for networks."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.ops import control_flow_ops


def apply_with_random_selector(x, func, num_cases):
  """Computes func(x, sel), with sel sampled from [0...num_cases-1].

  Args:
    x: input Tensor.
    func: Python function to apply.
    num_cases: Python int32, number of cases to sample sel from.

  Returns:
    The result of func(x, sel), where func receives the value of the
    selector as a python integer, but sel is sampled dynamically.
  """
  sel = tf.random_uniform([], maxval=num_cases, dtype=tf.int32)
  # Pass the real x only to one of the func calls.
  return control_flow_ops.merge([
      func(control_flow_ops.switch(x, tf.equal(sel, case))[1], case)
      for case in range(num_cases)])[0]

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
    image = tf.subtract(image, base_color_scale)
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
  """Pre-process one image for training or evaluation.

  Args:
    image: 3-D Tensor [height, width, channels] with the image.
    height: integer, image expected height.
    width: integer, image expected width.
    is_training: Boolean. If true it would transform an image for train,
      otherwise it would transform it for evaluation.
    bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
      where each coordinate is [0, 1) and the coordinates are arranged as
      [ymin, xmin, ymax, xmax].
    fast_mode: Optional boolean, if True avoids slower transformations.

  Returns:
    3-D float Tensor containing an appropriately scaled image

  Raises:
    ValueError: if user does not provide bounding box
  """
  # image = tf.image.resize_image_with_crop_or_pad(image, height, width)
  if is_training:
    return preprocess_for_train(image, height, width)
  else:
    return preprocess_for_eval(image, height, width)
