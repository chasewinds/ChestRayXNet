import os
import time

import tensorflow as tf
from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step
from tensorflow.python.platform import tf_logging as logging

from sklearn.metrics import roc_auc_score
import inception_preprocessing
import mlog
from data_prepare import get_split, load_batch
from inception_resnet_v2 import inception_resnet_v2, inception_resnet_v2_arg_scope

slim = tf.contrib.slim

class Train():
    def __init__(self, image_set_dir, tfrecord_dir, tfrecord_prefix, log_dir, image_label_list, checkpoint_file,
                 num_classes, num_epoch, batch_size, step_size, learning_rate, lr_decay_factor, image_size, weight_decay):
        self.image_set_dir = image_set_dir
        self.tfrecord_dir = tfrecord_dir
        self.tfrecord_prefix = tfrecord_prefix
        self.image_label_list = image_label_list
        self.checkpoint_file = checkpoint_file
        self.num_classes = num_classes
        self.num_epoch = num_epoch
        self.batch_size = batch_size
        self.step_size = step_size
        self.learning_rate = learning_rate
        self.lr_decay_factor = lr_decay_factor
        self.image_size = image_size
        self.weight_decay = weight_decay
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        self.log_dir = log_dir

    def _build_graph(self):
        with tf.Graph().as_default() as graph:
            tf.logging.set_verbosity(tf.logging.INFO)  # Set the verbosity to INFO level
            # #First create the dataset and load one batch
            def load_batch_from_tfrecord(self, split_name, dataset_dir=self.tfrecord_dir, num_classes=self.num_classes,
                                         file_pattern_for_counting=self.tfrecord_prefix, batch_size=self.batch_size):
                is_training = True if split_name == 'train' else False
                file_pattern = self.tfrecord_prefix + '_%s_*.tfrecord'
                dataset = get_split(split_name, dataset_dir, num_classes, file_pattern, file_pattern_for_counting)
                images, _, labels = load_batch(dataset, batch_size, num_classes, height=self.image_size, width=self.image_size,
                                               is_training=is_training)
                return images, labels, dataset.num_samples

            ## get train data
            train_images, self.train_labels, self.num_samples = load_batch_from_tfrecord(self, 'train')
            ## get validation data
            val_images, self.val_labels, self.val_num_samples = load_batch_from_tfrecord(self, 'validation')
            # #Know the number steps to take before decaying the learning rate and batches per epoch
            self.num_batches_per_epoch = (self.num_samples - 1) / self.batch_size + 1
            self.val_num_batches_per_epoch = (self.val_num_samples - 1) / self.batch_size + 1

            with slim.arg_scope(inception_resnet_v2_arg_scope()):
                logits, end_points = inception_resnet_v2(train_images, num_classes=self.num_classes, is_training=True)
            ## convert into probabilities
            self.probabilities = tf.sigmoid(logits)

            ## new loss, just equal to the sum of 14 log loss
            loss = tf.losses.log_loss(labels=self.train_labels, predictions=self.probabilities)
            # total_loss = tf.losses.get_total_loss()  # obtain the regularization losses as well
            l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
            total_loss = loss + l2_loss * self.weight_decay

            ## convert into actual predicte
            lesion_pred = tf.cast(tf.greater_equal(self.probabilities, 0.5), tf.float32)

            # Create the global step for monitoring the learning_rate and training.
            self.global_step = get_or_create_global_step()
            decay_steps = int(self.step_size * self.num_batches_per_epoch)
            # Define your exponentially decaying learning rate
            self.lr = tf.train.exponential_decay(
                learning_rate=self.learning_rate,
                global_step=self.global_step,
                decay_steps=decay_steps,
                decay_rate=self.lr_decay_factor,
                staircase=True)
            # Now we can define the optimizer that takes on the learning rate
            optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
            # Create the train_op.
            self.train_op = slim.learning.create_train_op(total_loss, optimizer)
            # State the metrics that you want to predict. We get a predictions that is not one_hot_encoded.
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(lesion_pred, self.train_labels), tf.float32))

            # def val_graph(images, labels):
            with slim.arg_scope(inception_resnet_v2_arg_scope()):
                val_logits, val_end_points = inception_resnet_v2(val_images, num_classes=self.num_classes,
                                                                 is_training=False, reuse=True)
                self.val_probabilities = tf.sigmoid(val_logits)

            ## new loss, just equal to the sum of 14 log loss
            self.val_loss = tf.losses.log_loss(labels=self.val_labels, predictions=self.val_probabilities)
            val_lesion_pred = tf.cast(tf.greater_equal(self.val_probabilities, 0.5), tf.float32)
            self.val_accuracy = tf.reduce_mean(tf.cast(tf.equal(val_lesion_pred, self.val_labels), tf.float32))

            # Now finally create all the summaries you need to monitor and group them into one summary op.
            tf.summary.scalar('losses/Total_Loss', total_loss)
            tf.summary.scalar('accuracy', self.accuracy)
            # tf.summary.scalar('auc', auc)
            tf.summary.scalar('learning_rate', self.lr)
            tf.summary.scalar('val_losses', self.val_loss)
            tf.summary.scalar('val_accuracy', self.val_accuracy)
            self.my_summary_op = tf.summary.merge_all()

    # def init_saver(self):
    #     variables_to_restore = slim.get_variables_to_restore(exclude=self.exclude)
    #     # Now we create a saver function that actually restores the variables from a checkpoint file in a sess
    #     self.saver = tf.train.Saver(variables_to_restore)

    def restore_fn(self, sess):
        if self.checkpoint_file:
            self.saver = tf.train.Saver()
            return self.saver.restore(sess, self.checkpoint_file)
        else:
            return None ## dose None really work?

    def _init_sess(self):
        self.sv = tf.train.Supervisor(logdir=self.log_dir, summary_op=None, init_fn=self.restore_fn)
        # Run the managed session
        self.sess = self.sv.managed_session()

    # Now we need to create a training step function that runs both the train_op, metrics_op and updates the global_step concurrently.
    def train_step(self, train_op, global_step, accuracy, lr, my_summary_op, train_label, probability):
        # Check the time for each sess run
        start_time = time.time()
        total_loss, global_step_count, accuracy_value, learning_rate, auc_label, auc_prob = self.sess.run(
            [train_op, global_step, accuracy, lr, train_label, probability])
        time_elapsed = time.time() - start_time
        # Run the logging to print some results
        auc = []
        for i in range(self.num_classes):
            sub_prob = [x[i] for x in auc_prob]
            sub_label = [x[i] for x in auc_label]
            try:
                auc.append(roc_auc_score(sub_label, sub_prob))
            except:
                continue
        logging.info('global step %s: learning rate: %s, accuracy: %s , loss: %.4f, auc : %s (%.2f sec/step)',
                     global_step_count, learning_rate, accuracy_value, total_loss, auc, time_elapsed)
        return total_loss, global_step_count, accuracy_value, learning_rate, my_summary_op

    def val_step(self, validation_loss, validation_accuracy, val_label, val_probability):
        loss_value, accuracy_value, label, prob = self.sess.run(
            [validation_loss, validation_accuracy, val_label, val_probability])
        auc = []
        for i in range(self.num_classes):
            sub_prob = [x[i] for x in prob]
            sub_label = [x[i] for x in label]
            try:
                auc.append(roc_auc_score(sub_label, sub_prob))
            except:
                continue
        return loss_value, accuracy_value, auc

    def train_all_epoch(self):
        epoch_loss = []
        for step in xrange(self.num_batches_per_epoch * self.num_epoch):
            ## run a train step
            loss, global_step_count, accuracy_value, \
            learning_rate, my_summary_ops = self.train_step(self.train_op, self.global_step, self.accuracy,
                                                            self.lr, self.my_summary_op, self.train_labels, self.probabilities)
            epoch_loss.append(loss)
            # At the start of every epoch, show the vital information:
            if step % self.num_batches_per_epoch == 0:
                logging.info('Epoch %s/%s', step / self.num_batches_per_epoch + 1, self.num_epoch)
                logging.info('Current Learning Rate: %s', learning_rate)
                logging.info(
                    'Mean loss on this training epoch is: %s' % (float(sum(epoch_loss)) / max(len(epoch_loss), 1)))
                epoch_loss[:] = []
                logging.info('Accuracy in this training epoch is : %s', accuracy_value)
                val_loss_arr = []
                val_acc_arr = []
                auc_arr = [0] * self.num_classes
                for i in xrange(self.val_num_batches_per_epoch):
                    loss_values, accuracy_values, auc = self.val_step(self.val_loss, self.val_accuracy, self.val_labels,
                                                                      self.val_probabilities)
                    val_loss_arr.append(loss_values)
                    val_acc_arr.append(accuracy_values)
                    logging.info('Loss on validation batch %s is : %s' % (i, loss_values))
                    logging.info('AUC on validaton batch %s is : %s' % (i, auc))
                    # for label_idx in range(len(auc)):
                    #     auc_arr[label_idx] += auc[label_idx]
                logging.info('Mean loss on this validation epoch is: %s' % (
                float(sum(val_loss_arr)) / max(len(val_loss_arr), 1)))
                logging.info('Mean accuracy on this validation epoch is: %s' % (
                float(sum(val_acc_arr)) / max(len(val_acc_arr), 1)))
                # mean_auc = [auc / val_num_batches_per_epoch for auc in auc_arr]
                # logging.info('Mean auc on this validation epoch is: %s' % mean_auc)

            # Log the summaries every 10 step.
            if step % 10 == 0:
                summaries = self.sess.run(my_summary_ops)
                self.sv.summary_computed(self.sess, summaries)

                # Once all the training has been done, save the log files and checkpoint model
                logging.info('Finished training! Saving model to disk now.')
                self.sv.saver.save(self.sess, self.sv.save_path, global_step=self.sv.global_step)

def run():
    # ================ DATASET INFORMATION ======================

    """
    all the argment need to read from shell is as fellow:
    dataset_dir, log_dir, checkpoint_file,
    num_class,
    origin_image_path, image_label_list, file_pattern_name,
    num_epochs, batch_size, learning_rate, lr_decay_factor, num_epochs_before_decay
    """
    flags = tf.app.flags
    flags.DEFINE_string('image_set_dir', None, 'String, The dirctory of your image set')

    flags.DEFINE_string('tfrecord_dir', None, 'String, The dirctory where your tfrecord set')

    flags.DEFINE_string('tfrecord_prefix', None,
                        'String, The prefix of your tfrecord, be used to pares your tfrecord file')

    flags.DEFINE_string('log_dir', None, 'String, The dirctory where the training log be saved at')

    flags.DEFINE_string('image_label_list', None,
                        'String, the list which save all your image and lable be used in training and validation')

    flags.DEFINE_string('checkpoint_file', None, 'String, The file your model weight fine turn from')

    # hparmeters
    flags.DEFINE_integer('num_classes', 2, 'Int, Number of classes your network output')

    flags.DEFINE_integer('num_epoch', 100, 'Int, Number of epoch your network run through all the dataset')

    flags.DEFINE_integer('batch_size', 16, 'Int, Number of images in a single mini batch')

    flags.DEFINE_integer('step_size', 20, 'Int, Number of epoch your network run before apply a weight decay operator')

    flags.DEFINE_integer('image_size', 224, 'Int, Set the input image shape of your network')

    flags.DEFINE_float('weight_decay', 1e-5, 'Float, Set the weight_decay factor of your network')

    flags.DEFINE_float('learning_rate', 0.0002, 'Float, Set the learning rate of your network')

    flags.DEFINE_float('lr_decay_factor', 0.7, 'Float, Set the decay factor every time you decay your learning rate')

    FLAGS = flags.FLAGS

    trainer = Train(FLAGS.image_set_dir, FLAGS.tfrecord_dir, FLAGS.tfrecord_prefix, FLAGS.log_dir,
                    FLAGS.image_label_list, FLAGS.checkpoint_file,FLAGS.num_classes, FLAGS.num_epoch,
                    FLAGS.batch_size, FLAGS.step_size, FLAGS.learning_rate, FLAGS.lr_decay_factor,
                    FLAGS.image_size, FLAGS.weight_decay)
    trainer.train_all_epoch()

if __name__ == '__main__':
    # mlog.initlog(FLAGS.log_dir)
    run()