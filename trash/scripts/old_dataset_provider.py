
import tensorflow as tf
import tensorflow.contrib.slim as slim
import os
import glob
import logging

class DataSet:
    """
    class dataset ,carry the
    image set infromation;
    """
    def __init__(self, dataset, n_samples, batch, batch_size):
        self.dataset = dataset
        self.batch = batch
        self.n_samples = n_samples
        self.batch_size = batch_size

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
        logging.debug("self.n_classes: %s" % self.n_classes)

        self.data_shape = [224, 224, 3]
        self.train_batch_size = train_batch_size

        self.n_train_samples, self.train_dataset = self.get_dataset(train_tfname, self.n_classes)
        self.train_batch = self.load_batch(self.train_dataset, self.data_shape[0], self.data_shape[1],
                                     self.train_batch_size, shuffle=True, class_id=class_id)
        self.train = DataSet(self.train_dataset, self.n_train_samples, self.train_batch, self.train_batch_size)

        self.validate_batch_size = validate_batch_size
        self.n_validate_samples, self.validate_dataset = self.get_dataset(validate_tfname, self.n_classes)
        self.validate_batch = self.load_batch(self.validate_dataset, self.data_shape[0], self.data_shape[1],
                                           self.validate_batch_size, class_id=class_id)

        self.validation = DataSet(self.validate_dataset, self.n_validate_samples, self.validate_batch, self.validate_batch_size)

    @classmethod
    def get_num_tfrecords(cls, tfname):
        return sum(1 for _ in tf.python_io.tf_record_iterator(tfname))

    # ============== DATASET LOADING ======================
    # We now create a function that creates a Dataset class which will give us many TFRecord files to feed in the examples into a queue in parallel.
    @classmethod
    def get_dataset(cls, tfname, n_classes):
        '''
        Obtains the split - training or validation - to create a Dataset class for feeding the examples into a queue later on. This function will
        set up the decoder and dataset information all into one Dataset class so that you can avoid the brute work later on.
        Your file_pattern is very important in locating the files later.

        INPUTS:
        - split_name(str): 'train' or 'validation'. Used to get the correct data split of tfrecord files
        - dataset_dir(str): the dataset directory where the tfrecord files are located
        - file_pattern(str): the file name structure of the tfrecord files in order to get the correct data

        OUTPUTS:
        - dataset (Dataset): A Dataset class object where we can read its various components for easier batch creation later.
        '''

        if os.path.splitext(tfname)[1] == ".tfrecord":
            tfnames = [tfname]
        else:
            tfnames = glob.glob(tfname + "_*-of-*.tfrecord")

        n_samples = 0
        for fname in tfnames:
            n = cls.get_num_tfrecords(fname)
            logging.info("file: %s, has %d records" % (fname, n))
            n_samples += n
        logging.info("n_samples: %d", n_samples) ##

        # Create a reader, which must be a TFRecord reader in this case
        reader = tf.TFRecordReader

        # Create the keys_to_features dictionary for the decoder
        keys_to_features = {
            'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
            'image/format': tf.FixedLenFeature((), tf.string, default_value='png'),
            'image/class/label': tf.FixedLenFeature((), tf.string, default_value='10'),## make sence?? that is m2
        }
        # logging.debug("label: %s" , keys_to_features['image/class/label'])


        # Create the items_to_handlers dictionary for the decoder.
        items_to_handlers = {
            'image': slim.tfexample_decoder.Image(),
            'label': slim.tfexample_decoder.Tensor('image/class/label'),
        }

        # Start to create the decoder
        decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)

        # Create the labels_to_name file
        # labels_to_name_dict = labels_to_name

        # set number of threads as num of shards
        n_readers = len(tfnames)

        # Actually create the dataset
        dataset = slim.dataset.Dataset(
            data_sources=tfnames,
            decoder=decoder,
            reader=reader,
            num_readers=n_readers,
            num_samples=n_samples,
            num_classes=n_classes,
            labels_to_name=None,
            items_to_descriptions=None)

        return n_samples, dataset

    # @classmethod
    # def get_slice(cls,label,dimention,batch_size):
    #     queue_size = 12 + 3 * batch_size
    #     label_slice = tf.slice(label,[0,dimention],[queue_size,1])
    #     return label_slice

    @classmethod
    def load_batch(cls, dataset, height, width, batch_size, is_training=True, num_threads=1,
                   capacity_of_batch=10, shuffle=False, class_id=3):
        '''
        Loads a batch for training.

        INPUTS: from the get_split function
        - batch_size(int): determines how big of a batch to traing or evaluation preprocessing

        - dataset(Dataset): a Dataset class object that is created
        - height(int): the height of the image to resize to during preprocessing
        - width(int): the width of the image to resize to during preprocessing
        - is_training(bool): to determine whether to perform a trainin
        OUTPUTS:
        - images(Tensor): a Tensor of the shape (batch_size, height, width, channels) that contain one batch of images
        - labels(Tensor): the batch's labels with the shape (batch_size,) (requires one_hot_encoding).

        '''
        # First create the data_provider object
        data_provider = slim.dataset_data_provider.DatasetDataProvider(
            dataset,
            common_queue_capacity=12 + 3 * batch_size,
            common_queue_min=12)

        # Obtain the raw image using the get method
        raw_image, label = data_provider.get(['image', 'label'])
        # label = tf.Print(label, [label])  ## normal
        label = tf.string_to_number(tf.string_split([label], delimiter="").values, tf.float32)

        # label = tf.Print(label, [label])
        logging.info("label.shape: %s", label.get_shape())
        if class_id != -1:
            # label = tf.reshape(label, shape=[2])
            label = tf.slice(label, [class_id], [1]) ##right or not ? waiting to ensure
        else:
            logging.debug("No need to slice")
        logging.info("after slice, label.shape: %s", label.get_shape())

        label = tf.Print(label, [label]) ##to see dothe slice work well?
        #
        #     # logging.debug("the label shape after slice is %s" % label.get_shape())
        #     # label = tf.squeeze(label)
        #     logging.info("slice label.shape: %s", label.get_shape())
        #     # label = tf.one_hot(label,2)
        # else:
        #     label = tf.reshape(label, shape=[14])

        # logging.debug("%s" % ('-'*20))
        # logging.debug("label type is :%s" % (type(label))) ## is tensor
        # logging.debug("label.get_shape: %s", label.get_shape())
        # logging.debug('label converted: %s', label)

        # logging.debug("label converted to int %d", label)

        # Perform the correct preprocessing for this image depending if it is training or evaluating
        # image = inception_preprocessing.preprocess_image(raw_image, height, width, is_training)
        raw_image = tf.image.random_flip_left_right(raw_image)

        # As for the raw images, we just do a simple reshape to batch it up
        raw_image = tf.expand_dims(raw_image, 0)
        raw_image = tf.image.resize_nearest_neighbor(raw_image, [height, width])
        raw_image = tf.squeeze(raw_image)
        logging.debug("raw_image.get_shape(): %s", raw_image.get_shape())


        # Batch up the image by enqueing the tensors internally in a FIFO queue and dequeueing many elements with tf.train.batch.
        if shuffle:
            raw_images, labels = tf.train.shuffle_batch(
                [raw_image, label],
                batch_size=batch_size,
                num_threads=num_threads,
                min_after_dequeue=capacity_of_batch * batch_size/2,
                capacity=capacity_of_batch * batch_size,
                allow_smaller_final_batch=False)
        else:
            raw_images, labels = tf.train.batch(
                [raw_image, label],
                batch_size=batch_size,
                num_threads=num_threads,
                capacity=capacity_of_batch * batch_size,
                allow_smaller_final_batch=False)

        return raw_images, labels
