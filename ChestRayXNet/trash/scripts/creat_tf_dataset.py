
import os, glob
import tensorflow as tf
import logging
from .imageutil import ImageUtil
import numpy as np
from ..nn.ModelFactory import ModelFactory

if float('.'.join(tf.__version__.split('.')[:2])) >= 1.4:
    import tensorflow.contrib.data as tf_data
else:
    import tensorflow.contrib.data as tf_data


class DataProvider:
    """
    Define data provider for train and validation
    """
    def __init__(self, config, m):
        self.datasets = {}
        for action in ('train', 'validate', 'test'):
            if config.get(action):
                if config[action]['data']:
                    self.datasets[action] = Dataset(action, config[action]['data'])
                else:
                    raise ValueError("Can not found [data] configuration in [%s]" % action)

        # max_batch_size = max([ds.batch_size for ds in self.datasets.values()])
        output_types = self.datasets.values()[0].dataset.output_types
        output_shapes = self.datasets.values()[0].dataset.output_shapes
        logging.debug("-----%s, %s", output_types, output_shapes)

        self.iterator = tf.contrib.data.Iterator.from_structure(output_types, output_shapes)
        self.get_next = self.iterator.get_next()

        for ds in self.datasets.values():
            ds.init_op = self.iterator.make_initializer(ds.dataset)
            ds.get_next = self.get_next

        if not m._template_on:
            for v_name, v_op in m.placeholders.items():
                data_key = v_name.split('/')[-1]
                v_op = self.get_next[data_key]
        else:
            m.set_inputs(self.get_next)



class Dataset:
    name = ''
    config = {}
    image_shape = []
    batch_size = []
    n_readers = 4           # default
    n_prefetch = 100        # default

    n_batches_per_epoch = 0
    n_epochs = 1

    tfnames = []
    n_smapels = 0

    transformer = None

    need_shuffle = False
    shuffle_buffer = 0
    shuffle_seed = 0
    ds = None
    init_ds_op = None
    get_next = None

    def __init__(self, name, config, n_epochs=1):
        self.name = name
        self.config = config
        self.image_shape = config['image_shape']

        self.batch_size = config['batch_size']
        self.n_readers = self.config.get('n_readers', self.n_readers)
        self.n_prefetch = self.config.get('prefetch', self.n_prefetch)
        self.n_epochs = n_epochs

        if config.get('transform') is not None:
            self.transformer = ImageTransformer(config.get('transform'), self.image_shape)
        else:
            self.transformer = None

        if "shuffle" in config:
            self.need_shuffle = True
            if isinstance(config['shuffle'], dict) and 'buffer' in config['shuffle']:
                self.shuffle_buffer = int(config['shuffle']['buffer'])
            else:
                self.shuffle_buffer = min(self.n_batches_per_epoch / 4, 100)

            if isinstance(config['shuffle'], dict) and 'seed' in config['shuffle']:
                self.shuffle_seed = int(config['shuffle']['seed'])
                self.shuffle_seed_placeholder = tf.placeholder_with_default(tf.to_int64(self.shuffle_seed), shape=[])
            else:
                self.shuffle_seed = None
        else:
            self.need_shuffle = False

        self.dataset = self.create_dataset()
        self.n_batches_per_epoch = (self.n_samples-1)/self.batch_size + 1

        self.init_ds_op = None
        self.get_next = None

    def create_dataset(self):
        """
            create dataset that read tfrecord files.
        :return:
        """
        if self.config['type'] == "tfrecord":
            dataset = self.dataset = self.read_tfrecord()
        elif self.config['type'] == "mnist":
            dataset = self.dataset = self.read_mnist()
        else:
            raise ValueError("Unsupported data source type: %s" % self.config['type'])

        if self.need_shuffle:
            logging.info("need shuffle, buffersize: %d, seed: %d",
                         self.shuffle_buffer*self.batch_size, self.shuffle_seed)
            dataset = dataset.shuffle(buffer_size=self.shuffle_buffer*self.batch_size, seed=self.shuffle_seed_placeholder)

        dataset = dataset.batch(self.batch_size)
        logging.info("batch, %s, %s", dataset.output_types, dataset.output_shapes)

        logging.info("set dataset %s, epochs: %d" % (self.name, self.n_epochs))
        if self.n_epochs >= 1:
            dataset = dataset.repeat(self.n_epochs)
        else:
            dataset = dataset.repeat()

        return dataset

    def read_mnist(self):

        image_file = self.config['image']
        label_file = self.config['label']
        logging.debug("image_file: %s, label_file: %s", image_file, label_file)

        with open(image_file) as f:
            buf = f.read()
        self.n_samples = (len(buf) - 16) / (28*28)
        train_image = np.frombuffer(buf, dtype=np.uint8, offset=16).  \
                          reshape([self.n_samples, 28, 28, 1]).           \
                          astype(np.float32)

        with open(label_file) as f:
            buf = f.read()
        assert(len(buf)-8 == self.n_samples)
        train_label = np.frombuffer(buf, dtype=np.uint8, offset=8).reshape(self.n_samples).astype(np.int32)

        dataset = tf.data.Dataset.zip({
            'image': tf.data.Dataset.from_tensor_slices(train_image),
            'label': tf.data.Dataset.from_tensor_slices(train_label),
        })
        logging.debug(dataset.output_types, dataset.output_shapes)

        return dataset

    def read_tfrecord(self):
        def get_num_tfrecords(tfname):
            n_samples = 0
            for _ in tf.python_io.tf_record_iterator(tfname):
                n_samples += 1
            return n_samples

        def get_file(tfname):
            logging.debug("tfname: %s", tfname)
            if isinstance(tfname, list):
                # if it is list, it should be a filename list
                tfnames = tfname
            else:
                if os.path.splitext(os.path.basename(tfname))[1] == ".tfrecord":
                    # if tfname has .tfrecord extension, it means it is a single filename and has no match pattern
                    tfnames = [tfname]
                else:
                    tfnames = glob.glob(tfname + "_*-of-*.tfrecord")

            if len(tfnames) == 0:
                raise ValueError("Can not find file pattern: %s" % tfname)

            n_samples = 0
            for fname in tfnames:
                n = get_num_tfrecords(fname)
                logging.info("file: %s, has %d records" % (fname, n))
                n_samples += n

            if n_samples == 0:
                raise ValueError("No records found in file: %s" % tfname)

            return tfnames, n_samples

        def _parse_fn(example_proto):
            keys_to_features = {
                'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
                # 'image/format': tf.FixedLenFeature((), tf.string, default_value='jpg'),
                'image/class/label': tf.FixedLenFeature(
                    [], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
            }
            parsed = tf.parse_single_example(example_proto, keys_to_features)
            raw_image = tf.image.decode_image(parsed['image/encoded'])

            image = raw_image
            if self.transformer:
                image = self.transformer(image)
            else:
                logging.info("self.image_shape: %s", self.image_shape)
                image = ImageUtil.aspect_preserving_resize(image, self.image_shape[0])
                image = ImageUtil._central_crop([image], self.image_shape[0], self.image_shape[1])[0]

            image = tf.to_float(image)
            image.set_shape(self.image_shape)

            label = tf.cast(parsed['image/class/label'], tf.int64)

            return {'raw_image': raw_image, 'image': image, 'label': label}

        self.tfnames, self.n_samples = get_file(self.config['source'])
        dataset = tf_data.TFRecordDataset(self.tfnames)

        dataset = dataset.map(_parse_fn, num_parallel_calls=self.n_readers)
        logging.info("map, %s, %s", dataset.output_types, dataset.output_shapes)

        return dataset


class ImageTransformer:
    def __init__(self, config, image_shape):
        self.config = config
        self.image_shape = image_shape
        self.mask_center = ImageUtil.get_mask_center(self.image_shape[0])
        self.corner_filler = 128 * (1 - self.mask_center)

    def __call__(self, image):
        for action in self.config:
            key, value = action.items()[0]
            if key == 'random_crop_ratio':
                resize_side_min = int(value.get('min', 1.0) * self.image_shape[0])
                resize_side_max = int(value.get('max', 1.0) * self.image_shape[0])
                logging.debug("resize_side_min: %s, resize_side_max: %s", resize_side_min, resize_side_max)
                resize_side = tf.random_uniform(
                    [], minval=resize_side_min, maxval=resize_side_max + 1, dtype=tf.int32)

                image = ImageUtil.aspect_preserving_resize(image, resize_side)
                image = ImageUtil.random_crop([image], self.image_shape[0],self.image_shape[1])[0]
            if key == 'random_crop_size':
                resize_side_min = value.get('min')
                resize_side_max = value.get('max')
                logging.debug("resize_side_min: %s, resize_side_max: %s", resize_side_min, resize_side_max)
                resize_side = tf.random_uniform(
                    [], minval=resize_side_min, maxval=resize_side_max + 1, dtype=tf.int32)

                image = ImageUtil.aspect_preserving_resize(image, resize_side)
                image = ImageUtil.random_crop([image], self.image_shape[0],self.image_shape[1])[0]
            elif key == "random_vertical" and value == True:
                image = tf.image.random_flip_up_down(image)
            elif key == "random_horizon" and value == True:
                image = tf.image.random_flip_left_right(image)
            elif key == "random_rotate" and value.get('n', 0) > 0:
                angle = tf.random_uniform([]) * value.get('n') * np.pi
                image = tf.contrib.image.rotate(image, angle)
                if self.mask_center is not None:
                    image = tf.multiply(image, self.mask_center)
                if self.corner_filler is not None:
                    image = tf.add(image, self.corner_filler)
            elif key == "scale":
                scale = eval(value)
                logging.debug("scale ratio: %f", scale)
                image = tf.multiply(image, scale)
            elif key == "channel_mean":
                means = value
                logging.info("channel_mean: %s", means)
                image = ImageUtil.mean_image_subtraction(image, means)
        return image