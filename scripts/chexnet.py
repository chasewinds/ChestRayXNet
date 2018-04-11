import os, errno
import time
import shutil
from datetime import timedelta

import numpy as np
import tensorflow as tf
import logging


TF_VERSION = float('.'.join(tf.__version__.split('.')[:2]))

def mkdirs_noerror_when_exists(directory):
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise e

class DenseNet:

    def __init__(self, data_provider, growth_rate, depth,
                 total_blocks, keep_prob,
                 weight_decay, nesterov_momentum, model_type,
                 should_save_logs, should_save_model,
                 renew_logs=False,
                 reduction=1.0,
                 bc_mode=False,
                 dataset="chexnet",
                 **kwargs):
        """
        Class to implement networks from this paper
        https://arxiv.org/pdf/1611.05552.pdf

        Args:
            data_provider: Class, that have all required data sets
            growth_rate: `int`, variable from paper
            depth: `int`, variable from paper
            total_blocks: `int`, paper value == 3
            keep_prob: `float`, keep probability for dropout. If keep_prob = 1
                dropout will be disables
            weight_decay: `float`, weight decay for L2 loss, paper = 1e-4
            nesterov_momentum: `float`, momentum for Nesterov optimizer
            model_type: `str`, 'DenseNet' or 'DenseNet-BC'. Should model use
                bottle neck connections or not.
            dataset: `str`, dataset name
            should_save_logs: `bool`, should logs be saved or not
            should_save_model: `bool`, should model be saved or not
            renew_logs: `bool`, remove previous logs for current model
            reduction: `float`, reduction Theta at transition layer for
                DenseNets with bottleneck layers. See paragraph 'Compression'
                https://arxiv.org/pdf/1608.06993v3.pdf#4
            bc_mode: `bool`, should we use bottleneck layers and features
                reduction or not.
        """
        self.data_provider = data_provider
        self.data_shape = data_provider.data_shape
        self.n_classes = data_provider.n_classes
        self.depth = depth
        self.growth_rate = growth_rate
        # how many features will be received after first convolution
        # value the same as in the original Torch code
        self.first_output_features = growth_rate * 2
        self.total_blocks = total_blocks
        # self.layers_per_block = (depth - (total_blocks + 1)) // total_blocks
        self.layers_per_block = [6, 12, 24, 16]
        # self.layers_per_block = [2, 2, 2, 2]
        self.bc_mode = bc_mode
        # compression rate at the transition layers
        self.reduction = reduction
        if not bc_mode:
            print("Build %s model with %d blocks, "
                  "%d composite layers each." % (
                model_type, self.total_blocks, self.layers_per_block))
        if bc_mode:
            # self.layers_per_block = self.layers_per_block // 2
            print("Build %s model with %d blocks, "
                  "%s bottleneck layers and %s composite layers each." % (
                      model_type, self.total_blocks, self.layers_per_block,
                      self.layers_per_block))
        print("Reduction at transition layers: %.1f" % self.reduction)

        self.keep_prob = keep_prob
        self.weight_decay = weight_decay
        self.nesterov_momentum = nesterov_momentum
        self.model_type = model_type
        self.dataset_name = dataset
        self.should_save_logs = should_save_logs
        self.should_save_model = should_save_model
        self.renew_logs = renew_logs
        self.batches_step = 0

        self._define_inputs()
        self._build_graph()
        self._initialize_session()
        self._count_trainable_params()

    def _initialize_session(self):
        """Initialize session, variables, saver"""
        config = tf.ConfigProto()
        # restrict model GPU memory utilization to min required
        config.gpu_options.allow_growth = True
        # config.operation_timeout_in_ms = 30000
        self.sess = tf.Session(config=config)
        tf_ver = int(tf.__version__.split('.')[1])
        print(tf_ver)
        if TF_VERSION <= 0.10:
            self.sess.run(tf.initialize_all_variables())
            logswriter = tf.train.SummaryWriter
        else:
            self.sess.run(tf.global_variables_initializer())
            logswriter = tf.summary.FileWriter
        self.saver = tf.train.Saver()
        self.summary_writer = logswriter(self.logs_path)
        # self.summaries = tf.summary.merge_all() ## add
        self.coord = tf.train.Coordinator()
        tf.train.start_queue_runners(sess=self.sess, coord=self.coord)

    def _count_trainable_params(self):
        total_parameters = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            variable_parametes = 1
            for dim in shape:
                variable_parametes *= dim.value
            total_parameters += variable_parametes
        print("Total training params: %.1fM" % (total_parameters / 1e6))

    @property
    def save_path(self):
        try:
            save_path = self._save_path
        except AttributeError:
            save_path = 'log/saves/%s' % self.model_identifier
            mkdirs_noerror_when_exists(save_path)
            save_path = os.path.join(save_path, 'model.chkpt')
            self._save_path = save_path
        return save_path

    @property
    def logs_path(self):
        try:
            logs_path = self._logs_path
        except AttributeError:
            logs_path = 'log/%s' % self.model_identifier
            if self.renew_logs:
                shutil.rmtree(logs_path, ignore_errors=True)
            mkdirs_noerror_when_exists(logs_path)
            self._logs_path = logs_path
        return logs_path

    @property
    def model_identifier(self):
        return "{}_growth_rate={}_depth={}_dataset_{}".format(
            self.model_type, self.growth_rate, self.depth, self.dataset_name)

    def save_model(self, global_step=None):
        self.saver.save(self.sess, self.save_path, global_step=global_step)

    def load_model(self):
        try:
            self.saver.restore(self.sess, self.save_path)
        except Exception as e:
            raise IOError("Failed to to load model "
                          "from save path: %s" % self.save_path)
        self.saver.restore(self.sess, self.save_path)
        print("Successfully load model from save path: %s" % self.save_path)

    def log_loss_accuracy(self, loss, accuracy, epoch, prefix,
                          should_print=True):
        if should_print:
            print("mean cross_entropy: %f, mean accuracy: %f" % (
                loss, accuracy))
        summary = tf.Summary(value=[
            tf.Summary.Value(
                tag='loss_%s' % prefix, simple_value=float(loss)),
            tf.Summary.Value(
                # tag='accuracy_%s' % prefix, simple_value=float(accuracy))
                tag='accuracy_%s' % prefix, simple_value=accuracy)
        ])
        self.summary_writer.add_summary(summary, epoch) ##just write down to log

    def _define_inputs(self):
        shape = [None]
        shape.extend(self.data_shape)
        self.images = tf.placeholder(
            tf.float32,
            shape=shape, ## depend on image shape
            name='input_images')
        self.labels = tf.placeholder(
            tf.float32,
            # shape=[None, self.n_classes],
            shape=[None, 1], ## this is where the mistake happened!
            name='labels')
        self.learning_rate = tf.placeholder(
            tf.float32,
            shape=[],
            name='learning_rate')
        self.is_training = tf.placeholder(tf.bool, shape=[])

    def composite_function(self, _input, out_features, kernel_size=3):
        """Function from paper H_l that performs:
        - batch normalization
        - ReLU nonlinearity
        - convolution with required kernel
        - dropout, if required
        """
        with tf.variable_scope("composite_function"):
            # BN
            output = self.batch_norm(_input)
            # ReLU
            output = tf.nn.relu(output)
            # convolution
            output = self.conv2d(
                output, out_features=out_features, kernel_size=kernel_size)
            # dropout(in case of training and in case it is no 1.0)
            output = self.dropout(output)
        return output

    def bottleneck(self, _input, out_features):
        with tf.variable_scope("bottleneck"):
            output = self.batch_norm(_input)
            output = tf.nn.relu(output)
            inter_features = out_features * 4
            output = self.conv2d(
                output, out_features=inter_features, kernel_size=1,
                padding='VALID')
            output = self.dropout(output)
        return output

    def add_internal_layer(self, _input, growth_rate):
        """Perform H_l composite function for the layer and after concatenate
        input with output from composite function.
        """
        # call composite function with 3x3 kernel
        if not self.bc_mode:
            comp_out = self.composite_function(
                _input, out_features=growth_rate, kernel_size=3)
        elif self.bc_mode:
            bottleneck_out = self.bottleneck(_input, out_features=growth_rate)
            comp_out = self.composite_function(
                bottleneck_out, out_features=growth_rate, kernel_size=3)
        # concatenate _input with out from composite function
        if TF_VERSION >= 1.0:
            output = tf.concat(axis=3, values=(_input, comp_out))
        else:
            output = tf.concat(3, (_input, comp_out)) ##why concatenate the third dim?

        return output

    def add_block(self, _input, growth_rate, layers_per_block):
        """Add N H_l internal layers"""
        output = _input
        for layer in range(layers_per_block):
            with tf.variable_scope("layer_%d" % layer):
                output = self.add_internal_layer(output, growth_rate)
        return output

    def transition_layer(self, _input):
        """Call H_l composite function with 1x1 kernel and after average
        pooling
        """
        # call composite function with 1x1 kernel
        out_features = int(int(_input.get_shape()[-1]) * self.reduction)
        output = self.composite_function(
            _input, out_features=out_features, kernel_size=1)
        # run average pooling
        output = self.avg_pool(output, k=2)
        return output

    def transition_layer_to_classes(self, _input):
        """This is last transition to get probabilities by classes. It perform:
        - batch normalization
        - ReLU nonlinearity
        - wide average pooling
        - FC layer multiplication
        """
        # BN
        output = self.batch_norm(_input)
        # ReLU
        output = tf.nn.relu(output)
        # average pooling
        last_pool_kernel = int(output.get_shape()[-2])
        output = self.avg_pool(output, k=last_pool_kernel)
        # FC
        features_total = int(output.get_shape()[-1])
        output = tf.reshape(output, [-1, features_total])
        logging.debug("the shape of output tensor before matmul is :%s" % output.get_shape())
        W = self.weight_variable_xavier(
            [features_total, self.n_classes], name='W')
        logging.debug("the shape of the final layer weight is :%s" % W.get_shape())
        bias = self.bias_variable([self.n_classes])
        logits = tf.matmul(output, W) + bias
        return logits

    def conv2d(self, _input, out_features, kernel_size,
               strides=[1, 1, 1, 1], padding='SAME'):
        in_features = int(_input.get_shape()[-1])
        kernel = self.weight_variable_msra(
            [kernel_size, kernel_size, in_features, out_features],
            name='kernel')
        output = tf.nn.conv2d(_input, kernel, strides, padding)
        return output

    def avg_pool(self, _input, k):
        ksize = [1, k, k, 1]
        strides = [1, k, k, 1]
        padding = 'VALID'
        output = tf.nn.avg_pool(_input, ksize, strides, padding)
        return output

    def batch_norm(self, _input):
        output = tf.contrib.layers.batch_norm(
            _input, scale=True, is_training=self.is_training,
            updates_collections=None)
        return output

    def dropout(self, _input):
        if self.keep_prob < 1:
            output = tf.cond(
                self.is_training,
                lambda: tf.nn.dropout(_input, self.keep_prob),
                lambda: _input
            )
        else:
            output = _input
        return output

    def weight_variable_msra(self, shape, name):
        return tf.get_variable(
            name=name,
            shape=shape,
            initializer=tf.contrib.layers.variance_scaling_initializer())

    def weight_variable_xavier(self, shape, name):
        return tf.get_variable(
            name,
            shape=shape,
            initializer=tf.contrib.layers.xavier_initializer())

    def bias_variable(self, shape, name='bias'):
        initial = tf.constant(0.0, shape=shape)
        return tf.get_variable(name, initializer=initial)

    def _build_graph(self):
        tf.set_random_seed(1234)
        growth_rate = self.growth_rate
        layers_per_block = self.layers_per_block
        # first - initial 3 x 3 conv to first_output_features
        with tf.variable_scope("Initial_convolution"):
            output = self.conv2d( ## the first conv befor block
                self.images,
                out_features=self.first_output_features,
                kernel_size=3)

        # add N required blocks
        for block in range(self.total_blocks): ## bulid the four blocks
            with tf.variable_scope("Block_%d" % block):
                output = self.add_block(output, growth_rate, layers_per_block[block])
            # last block exist without transition layer
            if block != self.total_blocks - 1:
                with tf.variable_scope("Transition_after_block_%d" % block):
                    output = self.transition_layer(output)

        with tf.variable_scope("Transition_to_classes"):
            logits = self.transition_layer_to_classes(output)
        # logging.debug("logging in one epochs is : %s" % (output))
        # prediction = tf.nn.softmax(logits)
        possbility = tf.sigmoid(logits)

        logits = tf.Print(logits, ["logits: ", logits], first_n=-1)
        possbility = tf.Print(possbility,["prob: ", possbility], first_n=-1)

        prediction = tf.cast(
            tf.cast(possbility + 0.5, dtype=tf.int32), dtype=tf.float32)  ## this case can only works out when possiblity above -1,so it's no right in some case.

        # prediction = tf.Print(prediction,[prediction])
        # Losses
        # logits = tf.sigmoid(logits)
        # cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        #     logits=logits, labels=self.labels))
        cross_entropy = tf.reduce_mean(tf.losses.log_loss(
            labels=self.labels, predictions=possbility, epsilon=1e-15)) ## defult epsilon is 1e-7,try 1e-15

        self.cross_entropy = cross_entropy
        l2_loss = tf.add_n(
            [tf.nn.l2_loss(var) for var in tf.trainable_variables()])

        # optimizer and train step
        optimizer = tf.train.MomentumOptimizer(
            self.learning_rate, self.nesterov_momentum, use_nesterov=True)
        self.train_step = optimizer.minimize(
            cross_entropy + l2_loss * self.weight_decay)

        correct_prediction = tf.equal(prediction,self.labels) ##be modified
        # correct_prediction = tf.equal(
        #     tf.argmax(prediction, 1),
        #     tf.argmax(self.labels, 1))
        self.accuracy = tf.reduce_mean(
            tf.cast(correct_prediction, tf.float32))

    def train_all_epochs(self, train_params):
        n_epochs = train_params['n_epochs']
        learning_rate = train_params['initial_learning_rate']
        batch_size = train_params['batch_size']
        reduce_lr_epoch_1 = train_params['reduce_lr_epoch_1']
        reduce_lr_epoch_2 = train_params['reduce_lr_epoch_2']
        total_start_time = time.time()
        for epoch in range(1, n_epochs + 1):
            logging.info("\n%s%s%s\n", '-'*30, "Train epoch: %d" % epoch, '-'*30)
            start_time = time.time()
            if epoch == reduce_lr_epoch_1 or epoch == reduce_lr_epoch_2:
                learning_rate = learning_rate / 10.0
                print("Decrease learning rate, new lr = %f" % learning_rate)

            logging.info("Training...")
            loss, acc = self.train_one_epoch(
                self.data_provider.train, learning_rate)
            if self.should_save_logs:
                self.log_loss_accuracy(loss, acc, epoch, prefix='train')

            if train_params.get('validation_set', False):
                logging.info("Validation...")
                loss, acc = self.test(self.data_provider.validation)
                if self.should_save_logs:
                    self.log_loss_accuracy(loss, acc, epoch, prefix='valid')

            time_per_epoch = time.time() - start_time
            seconds_left = int((n_epochs - epoch) * time_per_epoch)
            print("Time per epoch: %s, Est. complete in: %s" % (
                str(timedelta(seconds=time_per_epoch)),
                str(timedelta(seconds=seconds_left))))

            if self.should_save_model:
                self.save_model()

        total_training_time = time.time() - total_start_time
        print("\nTotal training time: %s" % str(timedelta(
            seconds=total_training_time)))

    def train_one_epoch(self, data, learning_rate):
        num_examples = data.n_samples
        total_loss = []
        total_accuracy = []
        # label_all = np.zeros(num_examples) ##
        label_all_sample = []

        n_batches_per_epoch = (num_examples-1)/data.batch_size + 1
        logging.info("n_batches_per_epoch: %s", n_batches_per_epoch)
        idx = 0
        for i in range(n_batches_per_epoch):
            try:
                # batch = data.next_batch(batch_size)
                logging.info("train batch %d/%d", i+1, n_batches_per_epoch)
                batch = self.sess.run(data.batch)
                images, labels = batch['image'], batch['label']

                # labels = tf.Print(labels,[labels])
                # logging.debug("the label in one batch is %s" % labels)
                logging.debug("\nlabel shape is :%s" % (labels.shape,))
                logging.debug("\nimages shape is :%s" % (images.shape,))
                # label_all.append(labels)
                ## use flatten to convert the label([[1],[0],[1]...[1]]) to the shape [1,0,1...1].
                # label_all[idx:idx+labels.shape[0]] = labels.flatten()
                # idx += labels.shape[0] ## not workout still need modify
                label_all_sample.extend(labels.flatten())
                # logging.debug("label_all shape is :%s" % label_all.shape)
                # logging.debug("label:\n%s", labels)
                feed_dict = {
                    self.images: images,
                    self.labels: labels,
                    self.learning_rate: learning_rate,
                    self.is_training: True,
                }
                fetches = [self.train_step, self.cross_entropy, self.accuracy]
                result = self.sess.run(fetches, feed_dict=feed_dict)
                _, loss, accuracy = result
                # logging.debug("loss: %s, type of loss: %s, accuracy: %s, type of accuracy: %s", loss, type(loss), accuracy, type(accuracy))
                # the type of accuracy is numpy.float32
                class_mean_accuracy = np.mean(accuracy)
                logging.info("loss: %s, class_mean_accuracy: %s, accuracy: %s"
                              % (loss, class_mean_accuracy, accuracy)) # bug is now the accuracy is always 1.0
                total_loss.append(loss)
                total_accuracy.append(class_mean_accuracy)
                if self.should_save_logs:
                    self.batches_step += 1
                    self.log_loss_accuracy(
                        loss, class_mean_accuracy, self.batches_step, prefix='per_batch',
                        should_print=False)
            except tf.errors.OutOfRangeError:
                logging.info("no more data to read")
                break

        label_all = np.array(label_all_sample)
        positive_sample = sum(label_all == 1)
        negtive_sample = len(label_all) - positive_sample
        positive_new = 0
        negtive_new = 0
        for i in range(len(label_all)):
            if label_all[i] == 1:
                positive_new += 1
            elif label_all[i] ==0:
                negtive_new += 1
        total_num = positive_new + negtive_new
        logging.info("the total positive sample is: %s"% positive_sample)
        logging.info("the total negtive sample is: %s" % negtive_sample)
        logging.info("the total num is :%s" % total_num)

        sample_ratio = float(positive_sample)/float(total_num)
        logging.info("the sample ratio is: %s" % sample_ratio)
        logging.info("the positive sample count by another way is: %s" % positive_new)
        logging.info("the negtive sample count by another way is: %s" % negtive_new)

        # print ("the total postive sample is :%s" % positive_sample)
        # for i in range(len(label_all)):
        #     if label_all[i] == 1:

        mean_loss = np.mean(total_loss)
        mean_accuracy = np.mean(total_accuracy)
        logging.info("mean loss in total train set is: %s" % (mean_loss))
        logging.info("mean accuracy in total train set is: %s" % (mean_accuracy))
        return mean_loss, mean_accuracy

    def test(self, data):
        num_examples = data.n_samples
        total_loss = []
        total_accuracy = []
        n_batches_per_epoch = (num_examples - 1) / data.batch_size + 1
        for i in range(n_batches_per_epoch):
            # batch = data.next_batch(batch_size)
            logging.debug("test batch %d/%d", i + 1, n_batches_per_epoch)
            batch = self.sess.run(data.batch)
            images, labels = batch['image'], batch['label']
            # logging.debug("label:\n%s", labels)
            feed_dict = {
                self.images: images,
                self.labels: labels,
                self.is_training: False,
            }
            fetches = [self.cross_entropy, self.accuracy]
            loss, accuracy = self.sess.run(fetches, feed_dict=feed_dict)
            class_mean_accuracy = np.mean(accuracy)
            logging.debug("loss: %s, class_mean_accuracy: %s\naccuracy: %s"
                          % (loss, class_mean_accuracy, accuracy))
            total_loss.append(loss)
            total_accuracy.append(class_mean_accuracy)
        mean_loss = np.mean(total_loss)
        mean_accuracy = np.mean(total_accuracy)
        logging.info("mean loss in total test set is: %s" % (mean_loss))
        logging.info("mean accuracy in total test set is: %s" % (mean_accuracy))
        return mean_loss, mean_accuracy