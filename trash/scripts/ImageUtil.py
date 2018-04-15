
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
import numpy as np
import logging



class ImageUtil:

	@classmethod
	def crop(cls, image, offset_height, offset_width, crop_height, crop_width):

		"""Crops the given image using the provided offsets and sizes.

		Note that the method doesn't assume we know the input image size but it does
		assume we know the input image rank.

		Args:
		  image: an image of shape [height, width, channels].
		  offset_height: a scalar tensor indicating the height offset.
		  offset_width: a scalar tensor indicating the width offset.
		  crop_height: the height of the cropped image.
		  crop_width: the width of the cropped image.

		Returns:
		  the cropped (and resized) image.

		Raises:
		  InvalidArgumentError: if the rank is not 3 or if the image dimensions are
			less than the crop size.
		"""
		original_shape = tf.shape(image)

		rank_assertion = tf.Assert(
			tf.equal(tf.rank(image), 3),
			['Rank of image must be equal to 3.'])
		cropped_shape = control_flow_ops.with_dependencies(
			[rank_assertion],
			tf.stack([crop_height, crop_width, original_shape[2]]))

		size_assertion = tf.Assert(
			tf.logical_and(
				tf.greater_equal(original_shape[0], crop_height),
				tf.greater_equal(original_shape[1], crop_width)),
			['Crop size greater than the image size.'])

		offsets = tf.to_int32(tf.stack([offset_height, offset_width, 0]))

		# Use tf.slice instead of crop_to_bounding box as it accepts tensors to
		# define the crop size.
		image = control_flow_ops.with_dependencies(
			[size_assertion],
			tf.slice(image, offsets, cropped_shape))
		return tf.reshape(image, cropped_shape)


	@classmethod
	def random_crop(cls, image_list, crop_height, crop_width):
		"""Crops the given list of images.

		The function applies the same crop to each image in the list. This can be
		effectively applied when there are multiple image inputs of the same
		dimension such as:

		  image, depths, normals = _random_crop([image, depths, normals], 120, 150)

		Args:
		  image_list: a list of image tensors of the same dim
		  ension but possibly
			varying channel.
		  crop_height: the new height.
		  crop_width: the new width.

		Returns:
		  the image_list with cropped images.

		Raises:
		  ValueError: if there are multiple image inputs provided with different size
			or the images are smaller than the crop dimensions.
		"""
		if not image_list:
			raise ValueError('Empty image_list.')

		# Compute the rank assertions.
		rank_assertions = []
		for i in range(len(image_list)):
			image_rank = tf.rank(image_list[i])
			rank_assert = tf.Assert(
				tf.equal(image_rank, 3),
				['Wrong rank for tensor  %s [expected] [actual]',
				 image_list[i].name, 3, image_rank])
			rank_assertions.append(rank_assert)

		image_shape = control_flow_ops.with_dependencies(
			[rank_assertions[0]],
			tf.shape(image_list[0]))
		image_height = image_shape[0]
		image_width = image_shape[1]
		crop_size_assert = tf.Assert(
			tf.logical_and(
				tf.greater_equal(image_height, crop_height),
				tf.greater_equal(image_width, crop_width)),
			['Crop size greater than the image size.'])

		asserts = [rank_assertions[0], crop_size_assert]

		for i in range(1, len(image_list)):
			image = image_list[i]
			asserts.append(rank_assertions[i])
			shape = control_flow_ops.with_dependencies([rank_assertions[i]],
													   tf.shape(image))
			height = shape[0]
			width = shape[1]

			height_assert = tf.Assert(
				tf.equal(height, image_height),
				['Wrong height for tensor %s [expected][actual]',
				 image.name, height, image_height])
			width_assert = tf.Assert(
				tf.equal(width, image_width),
				['Wrong width for tensor %s [expected][actual]',
				 image.name, width, image_width])
			asserts.extend([height_assert, width_assert])

		# Create a random bounding box.
		#
		# Use tf.random_uniform and not numpy.random.rand as doing the former would
		# generate random numbers at graph eval time, unlike the latter which
		# generates random numbers at graph definition time.
		max_offset_height = control_flow_ops.with_dependencies(
			asserts, tf.reshape(image_height - crop_height + 1, []))
		max_offset_width = control_flow_ops.with_dependencies(
			asserts, tf.reshape(image_width - crop_width + 1, []))
		offset_height = tf.random_uniform(
			[], maxval=max_offset_height, dtype=tf.int32)
		offset_width = tf.random_uniform(
			[], maxval=max_offset_width, dtype=tf.int32)

		return [ImageUtil.crop(image, offset_height, offset_width,
								crop_height, crop_width) for image in image_list]


	@staticmethod
	def _central_crop(image_list, crop_height, crop_width):
		"""Performs central crops of the given image list.

		Args:
		  image_list: a list of image tensors of the same dimension but possibly
			varying channel.
		  crop_height: the height of the image following the crop.
		  crop_width: the width of the image following the crop.

		Returns:
		  the list of cropped images.
		"""
		outputs = []
		for image in image_list:
			image_height = tf.shape(image)[0]
			image_width = tf.shape(image)[1]

			offset_height = (image_height - crop_height) / 2
			offset_width = (image_width - crop_width) / 2

			outputs.append(ImageUtil.crop(image, offset_height, offset_width,
										   crop_height, crop_width))
		return outputs

	@staticmethod
	def mean_image_subtraction(image, means):
		"""Subtracts the given means from each image channel.

		For example:
		  means = [123.68, 116.779, 103.939]
		  image = _mean_image_subtraction(image, means)

		Note that the rank of `image` must be known.

		Args:
		  image: a tensor of size [height, width, C].
		  means: a C-vector of values to subtract from each channel.

		Returns:
		  the centered image.

		Raises:
		  ValueError: If the rank of `image` is unknown, if `image` has a rank other
			than three or if the number of channels in `image` doesn't match the
			number of values in `means`.
		"""

		logging.debug("image.get_shape(): %s", image.get_shape())
		# image = tf.Print(image, [tf.shape(image)])
		if image.get_shape().ndims != 3:
			raise ValueError('Input must be of size [height, width, C>0]')
		num_channels = image.get_shape().as_list()[-1]
		if len(means) != num_channels:
			raise ValueError('len(means) must match the number of channels')

		channels = tf.split(axis=2, num_or_size_splits=num_channels, value=image)
		for i in range(num_channels):
			# channels[i] -= means[i]
			channels[i] = tf.subtract(channels[i], means[i])

		return tf.concat(axis=2, values=channels)

	@classmethod
	def _smallest_size_at_least(cls, height, width, smallest_side):
		"""Computes new shape with the smallest side equal to `smallest_side`.

		Computes new shape with the smallest side equal to `smallest_side` while
		preserving the original aspect ratio.

		Args:
		  height: an int32 scalar tensor indicating the current height.
		  width: an int32 scalar tensor indicating the current width.
		  smallest_side: A python integer or scalar `Tensor` indicating the size of
			the smallest side after resize.

		Returns:
		  new_height: an int32 scalar tensor indicating the new height.
		  new_width: and int32 scalar tensor indicating the new width.
		"""
		smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)

		smallest_side = tf.to_float(smallest_side)
		height = tf.to_float(height)
		width = tf.to_float(width)

		scale = tf.cond(tf.greater(height, width),
						lambda: smallest_side / width,
						lambda: smallest_side / height)
		new_height = tf.to_int32(tf.maximum(tf.multiply(height, scale), smallest_side))
		new_width = tf.to_int32(tf.maximum(tf.multiply(width, scale), smallest_side))
		return new_height, new_width

	@classmethod
	def aspect_preserving_resize(cls, image, smallest_side):
		"""Resize images preserving the original aspect ratio.

		Args:
		  image: A 3-D image `Tensor`.
		  smallest_side: A python integer or scalar `Tensor` indicating the size of
			the smallest side after resize.

		Returns:
		  resized_image: A 3-D tensor containing the resized image.
		"""
		smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)

		shape = tf.shape(image)
		height = shape[0]
		width = shape[1]
		new_height, new_width = cls._smallest_size_at_least(height, width, smallest_side)
		image = tf.expand_dims(image, 0)
		resized_image = tf.image.resize_bilinear(image, [new_height, new_width],
												 align_corners=False)
		resized_image = tf.squeeze(resized_image, axis=[0])
		# resized_image.set_shape([new_height, new_width, 3])
		return resized_image

	@classmethod
	def get_mask_center(cls, size, n_channels=3):
		r = size / 2
		mask = np.array([[1 if (i - r) ** 2 + (j - r) ** 2 < r ** 2 else 0 for j in range(size)] for i in range(size)],
						dtype=np.uint8)
		if n_channels > 0:
			mask_c = np.zeros((size, size, n_channels), dtype=np.uint8)
			for i in range(n_channels):
				mask_c[:, :, i] = mask
			return mask_c
		else:
			return mask

	@classmethod
	def random_rotate(cls, img, mask_center, corner_filler):
		# rotated_img = tf.contrib.image.rotate(img, angle)
		# # rotated_img =  mask_circle * rotated_img + corners
		# if not mask_center is None:
		# 	rotated_img = tf.multiply(rotated_img, mask_center)
		#
		# if not corner_filler is None:
		# 	rotated_img = tf.add(rotated_img, corner_filler)
		angle = tf.random_uniform([]) * 2 * np.pi
		return cls.rotate(img, angle, mask_center, corner_filler)

	@classmethod
	def rotate(cls, img, angle,  mask_center, corner_filler ):
		rotated_img = tf.contrib.image.rotate(img, angle)
		# rotated_img =  mask_circle * rotated_img + corners
		if not mask_center is None:
			rotated_img = tf.multiply(rotated_img, mask_center)

		if not corner_filler is None:
			rotated_img = tf.add(rotated_img, corner_filler)

		return rotated_img
