
import tensorflow as tf
import tensorflow.contrib.slim as slim
import os
import glob
import logging
from ImageUtil import ImageUtil

if float('.'.join(tf.__version__.split('.')[:2])) >= 1.4:
    import tensorflow.contrib.data as tf_data
else:
    import tensorflow.contrib.data as tf_data


class DataSet:
    """
    class dataset ,carry the
    image set infromation;
    """
    # self.train_tfname, self.train_batch_size, class_id = self.class_id, undersampling = True
    def __init__(self, image_shape, fname_pattern, batch_size, class_id=-1, sampling_ratio=1.0):
        self.total_mean = tf.constant([126.973], tf.float32)
        self.std = tf.constant([66.0], tf.float32)
        self.image_shape = image_shape
        self.batch_size = batch_size
        self.class_id = class_id
        self.sampling_ratio = sampling_ratio


        self.tfnames, self.n_samples, self.ds = self.get_dataset(fname_pattern)
        self.iterator = self.ds.make_one_shot_iterator()
        self.batch = self.iterator.get_next()


    def get_dataset(self, fname_pattern):

        n_readers = 10
        ## unpythonic way ,just for debug.


        def get_num_tfrecords(tfname):
            n_samples = 0
            for _ in tf.python_io.tf_record_iterator(tfname):
                n_samples += 1
            return n_samples

        def get_file(fname_pattern):
            logging.debug("fname_pattern: %s", fname_pattern)
            if isinstance(fname_pattern, list):
                # if it is list, it should be a filename list
                tfnames = fname_pattern
            else:
                if os.path.splitext(os.path.basename(fname_pattern))[1] == ".tfrecord":
                    # if tfname has .tfrecord extension, it means it is a single filename and has no match pattern
                    tfnames = [fname_pattern]
                else:
                    tfnames = glob.glob(fname_pattern + "_*-of-*.tfrecord")

            if len(tfnames) == 0:
                raise ValueError("Can not find file pattern: %s" % fname_pattern)

            n_samples = 0
            for fname in tfnames:
                n = get_num_tfrecords(fname)
                logging.info("file: %s, has %d records" % (fname, n))
                n_samples += n

            if n_samples == 0:
                raise ValueError("No records found fname_pattern: %s" % fname_pattern)

            # self.n_readers = len(tfnames) ##

            return tfnames, n_samples

        def sampling_filter(example):
            # prob_ratio = 0.2
            ##
            # label = example['image/class/label']
            # label = tf.string_to_number(tf.string_split([label], delimiter="").values, tf.float32)  ##
            # label = tf.slice(label, [self.class_id], [1])
            # label = tf.to_int64(label)
            # if label == 1:
            acceptance = tf.logical_or( ##
                            tf.equal(example['label'][0], 1),
                            tf.logical_and(
                             tf.equal(example['label'][0], 0),
                             tf.less_equal(tf.random_uniform([], dtype=tf.float32), self.sampling_ratio)))
            # acceptance = tf.equal(example['label'][0], 0)
            return acceptance

        # def upsampling(example):
        #     acc =

        def _parse_fn(example_proto):  ## the example_proto is what?

            # image_shape = [224, 224, 1]
            ## unpythonic way ,just for debug._

            keys_to_features = {
                'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
                'image/format': tf.FixedLenFeature((), tf.string, default_value='png'),
                'image/class/label': tf.FixedLenFeature((), tf.string, default_value='10'),
                ## make sence?? that is m2
            }
            parsed = tf.parse_single_example(example_proto, keys_to_features)
            raw_image = tf.image.decode_image(parsed['image/encoded'], channels=1)
            # raw_image = tf.Print(raw_image, ["raw image shape: ", tf.shape(raw_image)])
            # image = raw_image[:, :, 0]
            # image = tf.expand_dims(image, -1)
            image = raw_image



            # raw_image = tf.image.random_flip_left_right(raw_image)

            # As for the raw images, we just do a simple reshape to batch it up
            # raw_image = tf.expand_dims(raw_image, 0)
            # raw_image = tf.image.resize_nearest_neighbor(raw_image, [image_shape[0], image_shape[1]])
            # raw_image = tf.squeeze(raw_image)
            # logging.debug("raw_image.get_shape(): %s", raw_image.get_shape())
            image = ImageUtil.aspect_preserving_resize(image, self.image_shape[0])
            image = ImageUtil._central_crop([image], self.image_shape[0], self.image_shape[1])[0]
            # image = tf.Print(image, ["after center crop", tf.shape(image)])


            # logging.info("self.image_shape: %s", image_shape)
            image = tf.to_float(image)
            # tf.Print(image,['image before divide:',image], summarize=5)
            # image = tf.div(image, 255.0) ## divide by
            # tf.Print(image, ['image after divide:', image], summarize=5)

            # total_mean is:126.97351712
            # tf.Print(image,['before process', image])
            image = tf.subtract(image, self.total_mean)
            image = tf.div(image, self.std)
            image.set_shape(self.image_shape)
            # tf.Print(image,['after process', image])

            label = parsed['image/class/label']
            label = tf.string_to_number(tf.string_split([label], delimiter="").values, tf.float32)  ##
            # label = tf.Print(label, ["after string_to_number", label])
            if self.class_id != -1:
                # label = tf.reshape(label, shape=[2])
                label = tf.slice(label, [self.class_id], [1])
            else:
                logging.debug("no need to slice")

            label = tf.to_int64(label)
            logging.debug("label.shape: %s", label.get_shape())
            # label = tf.Print(label, ["after slice", label])

            return {'image': image, 'label': label}

        tfnames, n_samples = get_file(fname_pattern)
        dataset = tf_data.TFRecordDataset(tfnames)  ##tf.data.TFRecordDataset

        dataset = dataset.map(_parse_fn, num_parallel_calls=n_readers)
        logging.info("map, %s, %s", dataset.output_types, dataset.output_shapes)

        if self.sampling_ratio < 1.0:
            dataset = dataset.filter(sampling_filter)
        elif self.sampling_ratio > 1.0:
            repeat_num = tf.cast(self.sampling_ratio, tf.int64)
            base_num = tf.cast(1, tf.int64)
            ##we asume the label need to repeat is 1
            oversample_fn = lambda x: tf.cond(tf.equal(x['label'], 1), lambda: repeat_num, lambda: base_num)
            dataset = dataset.flat_map(
                lambda x: tf.data.Dataset.from_tensors(x).repeat(oversample_fn(x)))

            # dataset = dataset.filter(lambda x: tf.equal(x['label'][0], 0))

        dataset = dataset.repeat(None)
        dataset = dataset.shuffle(buffer_size=self.batch_size * 100) ##need modify
        dataset = dataset.batch(self.batch_size)

        return tfnames, n_samples, dataset

    #
    # # @classmethod
    # def train(self, tfname, batch_size, class_id, undersampling=True):
    #     self.get_dataset(tfname, batch_size, class_id, undersampling=undersampling)
    #
    # # @classmethod
    # def test(self, tfname, batch_size, class_id, undersampling=True):
    #     self.get_dataset(tfname, batch_size, class_id, undersampling=undersampling)

class DataProvider:
    """
    data_shape
    n_class
    train
    validation
    """
    def __init__(self, train_tfname, train_batch_size, validate_tfname, validate_batch_size, class_id=3):
        logging.info("train_tfname: %s, validate_tfname: %s" % (train_tfname, validate_tfname))
        self.n_classes = 14 if class_id == -1 else 1 ##there need modify m1

        self.n_readers = 10

        logging.debug("self.n_classes: %s" % self.n_classes)

        self.data_shape = [224, 224, 1]
        self.train_batch_size = train_batch_size
        self.validate_batch_size = validate_batch_size
        self.train_tfname = train_tfname
        self.validate_tfname = validate_tfname
        self.class_id = class_id
        # self.train_batch_size = train_batch_size

        # self.dataset = DataSet()
        # dataset = DataSet(self.train_tfname, self.train_batch_size, class_id=self.class_id)

        self.train = DataSet(self.data_shape, self.train_tfname, self.train_batch_size, class_id=self.class_id, sampling_ratio=0.24)
        self.validation = DataSet(self.data_shape, self.validate_tfname, self.validate_batch_size, class_id=self.class_id, sampling_ratio=1.0) ##
        self.train_batch = self.train.batch
        self.test_batch = self.validation.batch

        # ##apply undersampling
        # self.train_data = self.undersampling_dataset(dataset=self.train_data)
        # self.test_data = self.undersampling_dataset(dataset=self.test_data)
        #
        ## load batch
        # self.train_batch = self.load_batch(self.train_data)
        # self.test_batch = self.load_batch(self.test_data)




        # self.n_train_samples, self.train_dataset = self.get_dataset(train_tfname, self.n_classes)
        # self.train_batch = self.load_batch(self.train_dataset, self.data_shape[0], self.data_shape[1],
        #                              self.train_batch_size, shuffle=True, class_id=class_id)
        #
        # # self.train = DataSet(self.train_dataset, self.n_train_samples, self.train_batch, self.train_batch_size)
        # # dataset = self.train.filter(self.undersampling_filter)
        #
        # self.validate_batch_size = validate_batch_size
        # self.n_validate_samples, self.validate_dataset = self.get_dataset(validate_tfname, self.n_classes)
        # self.validate_batch = self.load_batch(self.validate_dataset, self.data_shape[0], self.data_shape[1],
        #                                    self.validate_batch_size, class_id=class_id)

        # self.validation = DataSet(self.validate_dataset, self.n_validate_samples, self.validate_batch, self.validate_batch_size)

    # @classmethod
    # def get_num_tfrecords(cls, tfname):
    #     return sum(1 for _ in tf.python_io.tf_record_iterator(tfname))



 #    def load_batch(self, dataset):
 #        dataset = dataset.shuffle(buffer_size=8 * 100)
 #        dataset = dataset.batch(self.batch_size)
 #
 #        self.iterator = dataset.make_one_shot_iterator()
 #        get_next = self.iterator.get_next()
 #        return get_next(dataset) ##
 #
 #
 #    @classmethod
 #    def load_batch(cls, dataset, height, width, batch_size, is_training=True, num_threads=1, ##note I modify cls to self to call the method undersampling_filter
 #                   capacity_of_batch=10, shuffle=False, class_id=3):
 #
 #        '''
 #        Loads a batch for training.
 #
 #        INPUTS: from the get_split function
 #        - batch_size(int): determines how big of a batch to traing or evaluation preprocessing
 #
 #        - dataset(Dataset): a Dataset class object that is created
 #        - height(int): the height of the image to resize to during preprocessing
 #        - width(int): the width of the image to resize to during preprocessing
 #        - is_training(bool): to determine whether to perform a trainin
 #        OUTPUTS:
 #        - images(Tensor): a Tensor of the shape (batch_size, height, width, channels) that contain one batch of images
 #        - labels(Tensor): the batch's labels with the shape (batch_size,) (requires one_hot_encoding).
 #
 #        '''
 #        # First create the data_provider object
 #
 #        ## dataset filter need to add there
 #        # why I can't
 #
 #        # dataset = dataset.filter(cls.undersampling_filter)
 #
 #        data_provider = slim.dataset_data_provider.DatasetDataProvider(
 #            dataset,
 #            common_queue_capacity=12 + 3 * batch_size,
 #            common_queue_min=12)
 #        ## what we need is convert dataset to queue.  remember shuffle during convert
 #        #use get_next() to do that.
 #
 #
 #        # Obtain the raw image using the get method
 #        raw_image, label = data_provider.get(['image', 'label'])
 #        # label = tf.Print(label, [label])  ## normal
 #        label = tf.string_to_number(tf.string_split([label], delimiter="").values, tf.float32)
 #
 #        # label = tf.Print(label, [label])
 #        logging.info("the shape of the tensor get from the dataset is : %s", label.get_shape())
 #        label = tf.Print(label, [label], summarize=14)
 #
 #        if class_id != -1:
 #            # label = tf.reshape(label, shape=[2])
 #            label = tf.slice(label, [class_id], [1]) ##right or not ? waiting to ensure
 #        else:
 #            logging.debug("No need to slice")
 #        logging.info("after slice, label.shape: %s", label.get_shape())
 #
 #        label = tf.Print(label, [label],summarize=10) ##to see dothe slice work well?
 #        #
 #        #     # logging.debug("the label shape after slice is %s" % label.get_shape())
 #        #     # label = tf.squeeze(label)
 #        #     logging.info("slice label.shape: %s", label.get_shape())
 #        #     # label = tf.one_hot(label,2)
 #        # else:
 #        #     label = tf.reshape(label, shape=[14])
 #
 #        # logging.debug("%s" % ('-'*20))
 #        # logging.debug("label type is :%s" % (type(label))) ## is tensor
 #        # logging.debug("label.get_shape: %s", label.get_shape())
 #        # logging.debug('label converted: %s', label)
 #
 #        # logging.debug("label converted to int %d", label)
 #
 #        # Perform the correct preprocessing for this image depending if it is training or evaluating
 #        # image = inception_preprocessing.preprocess_image(raw_image, height, width, is_training)
 #        raw_image = tf.image.random_flip_left_right(raw_image)
 #
 #        # As for the raw images, we just do a simple reshape to batch it up
 #        raw_image = tf.expand_dims(raw_image, 0)
 #        raw_image = tf.image.resize_nearest_neighbor(raw_image, [height, width])
 #        raw_image = tf.squeeze(raw_image)
 #        logging.debug("raw_image.get_shape(): %s", raw_image.get_shape())
 #
 #
 #        # Batch up the image by enqueing the tensors internally in a FIFO queue and dequeueing many elements with tf.train.batch.
 #        if shuffle:
 #            raw_images, labels = tf.train.shuffle_batch(
 #                [raw_image, label],
 #                batch_size=batch_size,
 #                num_threads=num_threads,
 #                min_after_dequeue=capacity_of_batch * batch_size/2,
 #                capacity=capacity_of_batch * batch_size,
 #                allow_smaller_final_batch=False)
 #        else:
 #            raw_images, labels = tf.train.batch(
 #                [raw_image, label],
 #                batch_size=batch_size,
 #                num_threads=num_threads,
 #                capacity=capacity_of_batch * batch_size,
 #                allow_smaller_final_batch=False)
 #
 #        return raw_images, labels
 #
 # # if __name__ == "__main__":
