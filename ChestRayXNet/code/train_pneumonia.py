import os
import time
import random

import tensorflow as tf
from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import tf_logging as logging

from sklearn.metrics import roc_auc_score
import data_preproces
from data_prepare import get_split, load_batch
from densenet_elu import densenet121, densenet161, densenet_arg_scope
from vgg import vgg_16, vgg_arg_scope
from custlearningrate import CustLearningRate
slim = tf.contrib.slim

#================ DATASET INFORMATION ======================

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

flags.DEFINE_string('tfrecord_prefix', None, 'String, The prefix of your tfrecord, be used to pares your tfrecord file')

flags.DEFINE_string('log_dir', None, 'String, The dirctory where the training log be saved at')

flags.DEFINE_string('log_txt_path', None, 'String, A txt file path which save the validation log information before every epoch')

flags.DEFINE_string('image_label_list', None, 'String, the list which save all your image and lable be used in training and validation')

flags.DEFINE_string('model_type', None, 'String, select network for training and validation')

flags.DEFINE_string('checkpoint_file', None, 'String, The file your model weight fine turn from')

# flags.DEFINE_string('checkpoint_file', None, 'String, The file your model weight fine turn from')

# hparmeters
flags.DEFINE_integer('num_classes', 2, 'Int, Number of classes your network output')

flags.DEFINE_integer('num_epoch', 100, 'Int, Number of epoch your network run through all the dataset')

flags.DEFINE_integer('batch_size', 16, 'Int, Number of images in a single mini batch')

flags.DEFINE_integer('step_size', 20, 'Int, Number of epoch your network run before apply a weight decay operator')

flags.DEFINE_float('learning_rate', 0.0003, 'Float, Set the learning rate of your network')

flags.DEFINE_float('lr_decay_factor', 0.7, 'Float, Set the decay factor every time you decay your learning rate')

flags.DEFINE_float('weight_decay', 1e-5, 'Float, the weight decay of l2 regular')

FLAGS = flags.FLAGS

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

def one_cycle_lr(step_one_epoch_n, step_two_epoch_n, min_lr, max_lr, step_two_decay):
    epochs_lr = []
    step_change = (max_lr - min_lr) / float(step_one_epoch_n / 2.0) # two step
    # 0.001 - 0.0001 = 0.0009 0.0009 / 50 = 0.00001 8
    for i in range(1, int(step_one_epoch_n / 2.0)):
        if i < step_one_epoch_n + 1:
            epochs_lr.append([i, min_lr + step_change * (i - 1)])
        else:
            epochs_lr.append([i, max_lr - step_change * (i - (step_one_epoch_n + 1))])
    for i in range(1, step_two_epoch_n):
        epochs_lr.append([i, min_lr * step_two_decay * i])
    return epochs_lr

def write_log(loss_arr, auc_arr, txt_path):
    lesion = ['Pneumonia']
    with open(txt_path, 'w') as f:
        for i in range(len(loss_arr)):
            f.write("The mean loss before Epoch %s, is %s\n" % (i + 1, loss_arr[i]))
            sample_auc = auc_arr[i]
            # lesion_auc = [[lesion[j], sample_auc[j]] for j in range(len(lesion))]
            # f.write("The AUC value of each sub class before Epoch %s, is: %s\n" % (i + 1, lesion_auc))
            f.write('%s : %s\n' % (lesion, sample_auc[1]))
            f.write('\n')

def run():
    image_size = 224
    if not os.path.exists(FLAGS.log_dir):
        os.mkdir(FLAGS.log_dir)
    #======================= TRAINING PROCESS =========================
    #Now we start to construct the graph and build our model
    with tf.Graph().as_default() as graph:
        tf.logging.set_verbosity(tf.logging.INFO) #Set the verbosity to INFO level
        # #First create the dataset and load one batch
        def load_batch_from_tfrecord(split_name, dataset_dir=FLAGS.tfrecord_dir, num_classes=FLAGS.num_classes,
                                     file_pattern_for_counting=FLAGS.tfrecord_prefix, batch_size=FLAGS.batch_size):
            is_training = True if split_name == 'train' else False
            file_pattern = FLAGS.tfrecord_prefix + '_%s_*.tfrecord'
            dataset = get_split(split_name, dataset_dir, num_classes, file_pattern, file_pattern_for_counting)
            images, _, labels = load_batch(dataset, batch_size, num_classes, height=image_size, width=image_size, is_training=is_training)
            return images, labels, dataset.num_samples

        ## get train data
        train_images, train_labels, num_samples = load_batch_from_tfrecord('train')
        ## get validation data
        val_images, val_labels, val_num_samples = load_batch_from_tfrecord('validation')
        # #Know the number steps to take before decaying the learning rate and batches per epoch
        num_batches_per_epoch = (num_samples - 1) / FLAGS.batch_size + 1
        val_num_batches_per_epoch = (val_num_samples - 1) / FLAGS.batch_size + 1

        if FLAGS.model_type == 'densenet121':
            with slim.arg_scope(densenet_arg_scope()):
                logits, _ = densenet121(train_images, fc_dropout_rate=0.5, num_classes=FLAGS.num_classes, is_training=True)

            # Define the scopes that you want to exclude for restoration
            exclude = ['densenet121/logits', 'densenet121/final_block', 'densenet121/squeeze']
            variables_to_restore = slim.get_variables_to_restore(exclude=exclude)    

        elif FLAGS.model_type == 'vgg16':
            with slim.arg_scope(vgg_arg_scope()):
                logits, _ = vgg_16(train_images, num_classes=FLAGS.num_classes, is_training=True)
            exclude = ['vgg_16/fc8'] # don't fine tuning the last layer.
            variables_to_restore = slim.get_variables_to_restore(exclude=exclude)

        ## convert into probabilities
        probabilities = tf.sigmoid(logits)
        def weighted_cross_entropy(logits, labels):
            predictions = tf.sigmoid(logits)
            # weight:0: 0.012736654326434312, 1: 0.9872633456735657
            epsilon = 1e-8
            loss = -math_ops.multiply(labels, math_ops.log(predictions + epsilon))*1 - math_ops.multiply(
            (1 - labels), math_ops.log(1 - predictions + epsilon))*1
            return loss
        binary_crossentropy = weighted_cross_entropy(logits, train_labels)
        total_loss = tf.reduce_mean(binary_crossentropy)
        # binary_crossentropy = tf.keras.backend.binary_crossentropy(target=train_labels, output=logits, from_logits=True)
        # l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables('densenet121/logits')])
        # total_loss = cross_entropy_loss + l2_loss * FLAGS.weight_decay
        # total_loss = tf.reduce_mean(total_loss)

        ## convert into actual predicte
        lesion_pred = tf.cast(tf.greater_equal(probabilities, 0.5), tf.float32)

        # Create the global step for monitoring the learning_rate and training.
        global_step = get_or_create_global_step()

        epochs_lr = [[20, 0.0003],
                     [20, 0.0001],
                     [20, 0.00001],
                     [20, 0.000001]]
        # use one cycle learning rate stratege
        # epochs_lr = [[1, 0.0001],
        #              [10, 0.0002],
        #              [20, 0.0004],
        #              [30, 0.0006],
        #              [40,0.0008],
        #              [50, 0.001],
        #              [],
        #              [],
        #              [100, 0.0001]]
        # epochs_lr = one_cycle_lr(step_one_epoch_n=60, step_two_epoch_n=10, min_lr=0.00004, max_lr=0.0004, step_two_decay=0.1)
        lr = CustLearningRate.IntervalLearningRate(epochs_lr=epochs_lr,
                                                   global_step=global_step,
                                                   steps_per_epoch=num_batches_per_epoch)

        # Now we can define the optimizer that takes on the learning rate
        optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.9, beta2=0.999, epsilon=1e-8)
        # Create the train_op.
        train_op = slim.learning.create_train_op(total_loss, optimizer)
        # State the metrics that you want to predict. We get a predictions that is not one_hot_encoded.
        accuracy = tf.reduce_mean(tf.cast(tf.equal(lesion_pred, train_labels), tf.float32))

        if FLAGS.model_type == 'densenet121':
            with slim.arg_scope(densenet_arg_scope()):
                val_logits, _ = densenet121(val_images, fc_dropout_rate=None, num_classes=FLAGS.num_classes, is_training=False, reuse=True)

        elif FLAGS.model_type == 'vgg16':
            with slim.arg_scope(vgg_arg_scope()):
                val_logits, _ = vgg_16(val_images, num_classes=FLAGS.num_classes, is_training=False, dropout_keep_prob=1, reuse=True)

        val_probabilities = tf.sigmoid(val_logits)

        ## new loss, just equal to the sum of 14 log loss
        # val_loss = tf.losses.log_loss(labels=val_labels, predictions=val_probabilities)
        # val_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=val_labels, logits=val_logits)
        # val_loss = tf.reduce_mean(val_loss)
        val_binary_crossentropy = tf.keras.backend.binary_crossentropy(target=train_labels, output=logits, from_logits=True)
        val_loss = tf.reduce_mean(val_binary_crossentropy)

        val_lesion_pred = tf.cast(tf.greater_equal(val_probabilities, 0.5), tf.float32)
        val_accuracy = tf.reduce_mean(tf.cast(tf.equal(val_lesion_pred, val_labels), tf.float32))
            # return loss, accuracy

        # Now finally create all the summaries you need to monitor and group them into one summary op.
        # tf.summary.scalar('losses/Total_Loss', total_loss)
        tf.summary.scalar('accuracy', accuracy)
        tf.summary.scalar('learning_rate', lr)
        tf.summary.scalar('val_losses', val_loss)
        tf.summary.scalar('val_accuracy', val_accuracy)
        my_summary_op = tf.summary.merge_all()


        # Now we need to create a training step function that runs both the train_op, metrics_op and updates the global_step concurrently.
        def train_step(sess, train_op, global_step, accuracy, lr, my_summary_op, train_label, probability, origin_loss):
            '''
            Simply runs a session for the three arguments provided and gives a logging on the time elapsed for each global step
            '''
            start_time = time.time()
            total_loss, global_step_count, accuracy_value, learning_rate, auc_label, auc_prob, log_loss = sess.run([train_op, global_step, accuracy, lr, train_label, probability, origin_loss])
            time_elapsed = time.time() - start_time
            #Run the logging to print some results
            #logging.info("prob output from the network is : %s, label is : %s, loss from log_loss function is : %s" % (auc_prob, auc_label, log_loss))
            # out_prob = [0 if y < 0.5 else 1 for x in auc_prob for y in x]
            # logging.info("DEBUG: sigmoid logits is : %s" % out_prob[0])
            auc = []
            for i in range(FLAGS.num_classes):
                sub_prob = [x[i] for x in auc_prob]
                sub_label = [x[i] for x in auc_label]
                try:
                    auc.append([i, round(roc_auc_score(sub_label, sub_prob), 2)])
                except:
                    continue
            epoch = global_step_count/num_batches_per_epoch + 1
            logging.info('Epoch: %s, global step %s: learning rate: %s, LOSS: %s, accuracy: %s , (%.2f sec/step)', epoch, global_step_count, learning_rate, log_loss, accuracy_value, time_elapsed)
            # logging.info("the loss in this step is : %s" % str(int(sum(sum(log_loss))) / 14.0))
            return total_loss, global_step_count, accuracy_value, learning_rate, my_summary_op, auc

        def val_step(sess, validation_loss, validation_accuracy, val_label, val_probability):
            # images, labels, _ = load_batch_from_tfrecord('val')
            # loss, accuracy = val_graph(val_images, val_labels)
            loss_value, accuracy_value, label, prob = sess.run([validation_loss, validation_accuracy, val_label, val_probability])
            auc = []
            for i in range(FLAGS.num_classes):
                sub_prob = [x[i] for x in prob]
                sub_label = [x[i] for x in label]
                try:
                    auc.append([i, round(roc_auc_score(sub_label, sub_prob), 2)])
                except:
                    continue
            logging.info('AUC on this validation batch is : %s' % auc)
            return loss_value, accuracy_value, label, prob

        # create a saver function that actually restores the variables from a checkpoint file in a sess
        saver = tf.train.Saver(variables_to_restore)
        def restore_fn(sess):
            return saver.restore(sess, FLAGS.checkpoint_file)

        # Define your supervisor for running a managed session. Do not run the summary_op automatically or else it will consume too much memory
        sv = tf.train.Supervisor(logdir=FLAGS.log_dir, summary_op=None, init_fn=restore_fn)
        # sv = tf.train.Supervisor(logdir=FLAGS.log_dir, summary_op=None)
        #Run the managed session
        with sv.managed_session() as sess:
            total_val_loss = []
            total_val_auc = []
            epoch_loss = []
            for step in xrange(num_batches_per_epoch * FLAGS.num_epoch):
                ## run a train step
                batch_loss, global_step_count, accuracy_value, learning_rate, my_summary_ops, auc = train_step(sess, train_op, global_step, accuracy, lr, my_summary_op, train_labels, probabilities, total_loss)
                epoch_loss.append(batch_loss)
                #At the start of every epoch, show some global informations and run validation set once:
                if step % num_batches_per_epoch == 0:
                    logging.info('Epoch %s/%s', step/num_batches_per_epoch + 1, FLAGS.num_epoch)
                    # logging.info('Mean loss on this training epoch is: %s' % (float(sum(epoch_loss)) / max(len(epoch_loss), 1)))
                    logging.info('Accuracy in this training epoch is : %s', accuracy_value)
                    epoch_loss[:] = []
                    val_label_arr = []
                    val_prob_arr = []
                    val_loss_arr = []
                    val_acc_arr = []
                    for i in xrange(val_num_batches_per_epoch): ## ok, I just want it run faster!
                        loss_values, accuracy_values, batch_label, batch_prob = val_step(sess, 
                        val_loss, val_accuracy, val_labels, val_probabilities)
                        # logging.info("float(sum(loss_values)) = %s" % float(sum(loss_values)))
                        batch_mean_loss = float(loss_values)
                        val_loss_arr.append(batch_mean_loss)
                        val_acc_arr.append(accuracy_values)
                        val_label_arr.append(batch_label)
                        val_prob_arr.append(batch_prob)
                        logging.info('Loss on validation batch %s is : %s' % (i, loss_values))
                        # logging.info('Accuracy on validaton batch %s is : %s' % (i, accuracy_values))
                    epoch_mean_loss = float(sum(val_loss_arr)) / max(len(val_loss_arr), 1)
                    total_val_loss.append(epoch_mean_loss)
                    logging.info('Mean loss on this validation epoch is: %s' % epoch_mean_loss)

                    mean_auc = epoch_auc(val_label_arr, val_prob_arr, FLAGS.num_classes)
                    logging.info('Mean auc on this validation epoch is: %s' % mean_auc)
                    total_val_auc.append(mean_auc)
                    # write_log(total_val_loss, total_val_auc, FLAGS.log_txt_path)

                # Log the summaries every 100 step.
                if step % 20 == 0:
                    logging.info('AUC value on the last training batch is : %s' % auc)
                    # logging.info('Loss every validation epoch collected is as fellow: %s' % total_val_loss)
                    # logging.info('AUC every validation epoch collected is as fellow : %s' % total_val_auc)
                    # logging.info('The 14 subclass loss on the last batch is : %s' % sum(batch_loss))
                    summaries = sess.run(my_summary_ops)
                    sv.summary_computed(sess, summaries)

            #Once all the training has been done, save the log files and checkpoint model
            logging.info('Finished training! Saving model to disk now.')
            sv.saver.save(sess, sv.save_path, global_step = sv.global_step)

if __name__ == '__main__':
    # mlog.initlog(FLAGS.log_dir)
    run()