import os
import time
import numpy as np
from sklearn.metrics import roc_auc_score

import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.ops import math_ops
from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step
# from inception_resnet_v2 import inception_resnet_v2, inception_resnet_v2_arg_scope
from densenet import densenet121, densenet_arg_scope
from dataset_provider import load_batch, get_split
from auc import get_auc

slim = tf.contrib.slim

flags = tf.app.flags

# State your log directory where you can retrieve your model
flags.DEFINE_string('log_dir', None, 'String, log dir')

# Create a new evaluation log directory to visualize the validation process
flags.DEFINE_string('log_eval', None, 'String, dir evaluate log save')

# State the dataset directory where the validation set is found
flags.DEFINE_string('dataset_dir', None, 'String, dataset dir')
#
flags.DEFINE_string('auc_picture_path', None, 'String, the path auc picture save')

flags.DEFINE_string('tfrecord_prefix', None, 'String, the predix of tfrecord')

# State the batch_size to evaluate each time, which can be a lot more than the training batch
flags.DEFINE_integer('batch_size', 36, 'Int, batch size')

# State the number of epochs to evaluate
flags.DEFINE_integer('num_epochs', 1, 'Int, number of epochs')

flags.DEFINE_integer('num_classes', 14, 'Int, number of class need to caculate auc')

flags.DEFINE_integer('ckpt_id', 1, 'Int, last (ckpt_id)th check point file for evaluate, if ckpt_id = 1, restore last ckpt file')

FLAGS = flags.FLAGS

def parse_label(pred, label, index):
    pred = [y[index] for x in pred for y in x]
    label = [y[index] for x in label for y in x]
    return pred, label

def require_ckpt_file(log_dir=FLAGS.log_dir, ckpt_id=FLAGS.ckpt_id):
    files = os.listdir(log_dir)
    ckpt_list = [x for x in files if x.split('-')[0] == 'model.ckpt']
    return os.path.join(FLAGS.log_dir ,ckpt_list[len(ckpt_list) - ckpt_id])

def epoch_auc(label, prob, num_class):
    auc_arr = []
    for i in range(num_class):
        epoch_total_label = [y[i] for x in label for y in x]
        epoch_total_pos_prob = [y[i] for x in prob for y in x]
        try:
            auc_arr.append([i, round(roc_auc_score(epoch_total_label, epoch_total_pos_prob), 2)])
        except:
            continue
    return auc_arr

def run():
    image_size = 224
    total_pred = []
    total_label = []
    epoch_loss = []
    # get the latest checkpoint file
    # checkpoint_file = tf.train.latest_checkpoint(FLAGS.log_dir)

    # create log_dir for evaluation information
    if not os.path.exists(FLAGS.log_eval):
        os.mkdir(FLAGS.log_eval)

    # TODO: construct the graph 
    with tf.Graph().as_default() as graph:
        tf.logging.set_verbosity(tf.logging.INFO)
        # # TODO: load one batch data
        # file_pattern = FLAGS.tfrecord_prefix + '_%s_*.tfrecord'
        # dataset = get_split('validation', FLAGS.dataset_dir, FLAGS.num_classes, file_pattern, FLAGS.tfrecord_prefix)
        # images, _, labels = load_batch(dataset, batch_size=FLAGS.batch_size, num_classes=FLAGS.num_classes, is_training=False)

        # TODO: load one batch
        def load_batch_from_tfrecord(split_name, dataset_dir=FLAGS.dataset_dir, num_classes=FLAGS.num_classes,
                                     tfrecord_prefix=FLAGS.tfrecord_prefix, batch_size=FLAGS.batch_size, image_size=224):
            is_training = True if split_name == 'train' else False
            file_pattern = FLAGS.tfrecord_prefix + '_%s_*.tfrecord'
            dataset = get_split(split_name, dataset_dir, num_classes, file_pattern, tfrecord_prefix)
            images, _, labels = load_batch(dataset, batch_size, num_classes, height=image_size, width=image_size, is_training=is_training)
            return images, labels, dataset.num_samples
        # get train data
        images, labels, num_samples = load_batch_from_tfrecord('validation')

        #Create some information about the training steps
        # assert dataset.num_samples % FLAGS.batch_size == 0, 'batch size can not be div by number sampels, the total sampels is %s' % dataset.num_samples
        num_batches_per_epoch = (num_samples - 1) / FLAGS.batch_size + 1

        # Now create the inference model but set is_training=False
        with slim.arg_scope(densenet_arg_scope()):
            logits, _ = densenet121(images, fc_dropout_rate=None, num_classes=FLAGS.num_classes, is_training=True)
        
        #get all the variables to restore from the checkpoint file and create the saver function to restore
        # variables_to_restore = slim.get_variables_to_restore()
        # saver = tf.train.Saver(variables_to_restore)
        # def restore_fn(sess):
        #     # checkpoint_file = require_ckpt_file(log_dir=FLAGS.log_dir, ckpt_id=FLAGS.ckpt_id)
        #     # checkpoint_file = 'log/2muti/model.ckpt-61220'
        #     logging.info('From logging, checkpoint_file: %s' % checkpoint_file)
        #     return saver.restore(sess, checkpoint_file)

        # TODO: calculate accuaracy and loss
        sigmoid_prob = tf.sigmoid(logits)
        lesion_pred = tf.cast(tf.greater_equal(sigmoid_prob, 0.5), dtype=tf.float32)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(lesion_pred, labels), tf.float32))

        # loss = tf.losses.log_loss(labels, sigmoid_op)
        # loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits))
        # TODO: define loss, loss is the sum of 14 binary cross entropy.
        def weighted_cross_entropy(logits, labels):
            predictions = tf.sigmoid(logits)
            # weight:0: 0.012736654326434312, 1: 0.9872633456735657
            epsilon = 1e-8
            return -math_ops.multiply(labels, math_ops.log(predictions + epsilon)) - math_ops.multiply((1 - labels), math_ops.log(1 - predictions + epsilon))
        # calculate loss
        binary_crossentropy = weighted_cross_entropy(logits, labels)
        loss = tf.reduce_mean(binary_crossentropy)

        #Create the global step and an increment op for monitoring
        global_step = get_or_create_global_step()
        global_step_op = tf.assign(global_step, global_step + 1) # no apply_gradient method so manually increasing the global_step

        #Create a evaluation step function
        def eval_step(sess):
            # Simply takes in a session, runs the metrics op and some logging information
            start_time = time.time()
            global_step_count, step_logits, step_acc, step_loss, step_pred, step_label = sess.run([global_step_op, logits, accuracy, loss, sigmoid_prob, labels])
            time_elapsed = time.time() - start_time

            logging.info('The averange accuracy of this batch(total 64 samples) is: %s, run time is:%s' % (step_acc, time_elapsed))
            logging.info('The step loss is : %s' % step_loss)

            epoch_loss.append(step_loss)
            total_pred.append(step_pred)
            total_label.append(step_label)

            ## process for predict and label
            # pred_compare = [0] * len(step_pred)
            # for i in range(len(step_pred)):
            #     pred_compare[i] = 1 if step_pred[i][1] > 0.5 else 0
            # pred_compare = [1 if x[1] > 0.5 else 0 for x in step_pred]
            # logging.info("The prediction of this batch is:%s" % pred_compare)

        #Define some scalar quantities to monitor
        # tf.summary.scalar('Validation_Accuracy', accuracy)
        # my_summary_op = tf.summary.merge_all()

        #Get your supervisor
        # sv = tf.train.Supervisor(logdir=FLAGS.log_eval, summary_op=None, saver=None, init_fn=restore_fn)
        with tf.train.MonitoredTrainingSession(checkpoint_dir=FLAGS.log_dir, save_checkpoint_secs=None) as sess:
            for step in xrange(num_batches_per_epoch * FLAGS.num_epochs):
                # sess.run(sv.global_step)
                # TODO: run one evaluate step
                eval_step(sess)

            # logging.info('len pred all %s' % len(pred_all))
            # logging.info('len label all %s' % len(label_all))
            auc = epoch_auc(total_label, total_pred, 14)
            logging.info('AUC value in this dateset is : %s' % auc)
            logging.info('Mean loss one validation set is : %s' % (sum(epoch_loss) / float(len(epoch_loss))))


if __name__ == '__main__':
    run()
