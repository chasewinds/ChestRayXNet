import os
import time

import tensorflow as tf
from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step
from tensorflow.python.platform import tf_logging as logging

from sklearn.metrics import roc_auc_score
import inception_preprocessing
# import mlog
from data_prepare import get_split, load_batch
from densenet import densenet161, densenet_arg_scope
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

flags.DEFINE_string('image_label_list', None, 'String, the list which save all your image and lable be used in training and validation')

flags.DEFINE_string('checkpoint_file', None, 'String, The file your model weight fine turn from')

# flags.DEFINE_string('checkpoint_file', None, 'String, The file your model weight fine turn from')

# hparmeters
flags.DEFINE_integer('num_classes', 2, 'Int, Number of classes your network output')

flags.DEFINE_integer('num_epoch', 100, 'Int, Number of epoch your network run through all the dataset')

flags.DEFINE_integer('batch_size', 16, 'Int, Number of images in a single mini batch')

flags.DEFINE_integer('step_size', 20, 'Int, Number of epoch your network run before apply a weight decay operator')

flags.DEFINE_float('learning_rate', 0.0002, 'Float, Set the learning rate of your network')

flags.DEFINE_float('lr_decay_factor', 0.7, 'Float, Set the decay factor every time you decay your learning rate')

flags.DEFINE_float('weight_decay', 1e-4, 'Float, the weight decay of l2 regular')

FLAGS = flags.FLAGS




def run():
    image_size = 224
    #Create the log directory here. Must be done here otherwise import will activate this unneededly.
    if not os.path.exists(FLAGS.log_dir):
        os.mkdir(FLAGS.log_dir)

    # Create the file pattern of your TFRecord files so that it could be recognized later on
    # file_patterns = FLAGS.tfrecord_prefix + '_%s_*.tfrecord'

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

        with slim.arg_scope(densenet_arg_scope()):
            logits, _ = densenet161(train_images, fc_dropout_rate=0.5, num_classes=FLAGS.num_classes, is_training=True)

        # Define the scopes that you want to exclude for restoration
        exclude = ['densenet161/Logits', 'densenet161/final_block']
        variables_to_restore = slim.get_variables_to_restore(exclude=exclude)
        """
        first auc for dense161 in earlier 26 epoch:
        The auc of each class is as fellow:
        [0.63520693651405791, 0.63710852720260935, 0.65156066467040386, 0.60359253095505239, 0.51762399936986558,
        0.4967026033329211, 0.59554476355082875, 0.5342352796204638, 0.69370695676663519, 0.8078309113155997,
        0.55810731640410682, 0.60321180925870621, 0.49445176619876041, 0.46183772550485502]"""
        ## convert into probabilities
        probabilities = tf.sigmoid(logits)
       ## new loss, just equal to the sum of 14 log loss
        loss = tf.losses.log_loss(labels=train_labels, predictions=probabilities)
        #l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
        # # tf.Print(loss, ["l2_loss: ", l2_loss])
        #loss = log_loss + l2_loss * FLAGS.weight_decay
        """
        losses = -math_ops.multiply(labels, math_ops.log(predictions + epsilon)) - math_ops.multiply((1 - labels), math_ops.log(1 - predictions + epsilon))
        losses = -labels * log(pred + eps) - (1 - labels) * (1 - pred + eps)
        """
        # l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
        # loss = bais_loss + l2_loss * 1e-5
        # total_loss = tf.losses.get_total_loss()  # obtain the regularization losses as well

        ## convert into actual predicte
        lesion_pred = tf.cast(tf.greater_equal(probabilities, 0.5), tf.float32)

        # Create the global step for monitoring the learning_rate and training.
        global_step = get_or_create_global_step()
        # decay_steps = int(FLAGS.step_size * num_batches_per_epoch)
        # Define your exponentially decaying learning rate
        # lr = tf.train.exponential_decay(
        #     learning_rate=FLAGS.learning_rate,
        #     global_step=global_step,
        #     decay_steps=decay_steps,
        #     decay_rate=FLAGS.lr_decay_factor,
        #     staircase=True)

        # epochs_lr = [[5, 0.01],
        #              [40, 0.001],
        #              [60, 0.0001],
        #              [50, 0.00001]]
        epochs_lr = [[30, 0.0003],
                     [30, 0.0001],
                     [50, 0.00001],
                     [100, 0.0000006]]
        lr = CustLearningRate.IntervalLearningRate(epochs_lr=epochs_lr,
                                                   global_step=global_step,
                                                   steps_per_epoch=num_batches_per_epoch)

        # Now we can define the optimizer that takes on the learning rate
        optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.9, beta2=0.999, epsilon=1e-8)
        # Create the train_op.
        train_op = slim.learning.create_train_op(loss, optimizer)
        # State the metrics that you want to predict. We get a predictions that is not one_hot_encoded.
        accuracy = tf.reduce_mean(tf.cast(tf.equal(lesion_pred, train_labels), tf.float32))
        # metrics_op = tf.group(accuracy, lesion_pred)
        # auc, _ = tf.metrics.auc(train_labels, probabilities)

        # def val_graph(images, labels):
        with slim.arg_scope(densenet_arg_scope()):
            val_logits, _ = densenet161(val_images, fc_dropout_rate=None, num_classes=FLAGS.num_classes, is_training=False, reuse=True)
        val_probabilities = tf.sigmoid(val_logits)

        ## new loss, just equal to the sum of 14 log loss
        val_loss = tf.losses.log_loss(labels=val_labels, predictions=val_probabilities)
        val_lesion_pred = tf.cast(tf.greater_equal(val_probabilities, 0.5), tf.float32)
        val_accuracy = tf.reduce_mean(tf.cast(tf.equal(val_lesion_pred, val_labels), tf.float32))
            # return loss, accuracy

        # Now finally create all the summaries you need to monitor and group them into one summary op.
        tf.summary.scalar('losses/Total_Loss', loss)
        tf.summary.scalar('accuracy', accuracy)
        # tf.summary.scalar('auc', auc)
        tf.summary.scalar('learning_rate', lr)
        # tf.summary.scalar('epoch', )
        tf.summary.scalar('val_losses', val_loss)
        tf.summary.scalar('val_accuracy', val_accuracy)
        my_summary_op = tf.summary.merge_all()


        #Now we need to create a training step function that runs both the train_op, metrics_op and updates the global_step concurrently.
        def train_step(sess, train_op, global_step, accuracy, lr, my_summary_op, train_label, probability, origin_loss):
            '''
            Simply runs a session for the three arguments provided and gives a logging on the time elapsed for each global step
            '''
            # images, labels, num_samples = load_batch_from_tfrecord('train')
            # Know the number steps to take before decaying the learning rate and batches per epoch
            # num_batches_per_epoch = (num_samples - 1) / FLAGS.batch_size + 1
            ## get operation from train graph
            # train_op, accuracy, auc, global_step, lr, my_summary_op = train_graph(train_images, train_labels, num_batches_per_epoch)
            # Check the time for each sess run
            start_time = time.time()
            total_loss, global_step_count, accuracy_value, learning_rate, auc_label, auc_prob, log_loss = sess.run([train_op, global_step, accuracy, lr, train_label, probability, origin_loss])
            time_elapsed = time.time() - start_time
            #Run the logging to print some results
            #logging.info("prob output from the network is : %s, label is : %s, loss from log_loss function is : %s" % (auc_prob, auc_label, log_loss))
            out_prob = [0 if y < 0.5 else 1 for x in auc_prob for y in x]
            logging.info("DEBUG: sigmoid logits is : %s" % out_prob[0])
            auc = []
            for i in range(FLAGS.num_classes):
                sub_prob = [x[i] for x in auc_prob]
                sub_label = [x[i] for x in auc_label]
                try:
                    auc.append([i, roc_auc_score(sub_label, sub_prob)])
                except:
                    continue
            epoch = global_step_count/num_batches_per_epoch + 1
            logging.info('Epoch: %s, global step %s: learning rate: %s, accuracy: %s , loss: %.4f, (%.2f sec/step)', epoch, global_step_count, learning_rate, accuracy_value, total_loss, time_elapsed)
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
                    auc.append(roc_auc_score(sub_label, sub_prob))
                except:
                    continue
            return loss_value, accuracy_value, auc

        # # Define the scopes that you want to exclude for restoration
        # exclude = ['densenet121/logits', 'densenet121/final_block']
        # variables_to_restore = slim.get_variables_to_restore(exclude=exclude)
        # # Now we create a saver function that actually restores the variables from a checkpoint file in a sess
        saver = tf.train.Saver(variables_to_restore)

        def restore_fn(sess):
            return saver.restore(sess, FLAGS.checkpoint_file)
        # Define your supervisor for running a managed session. Do not run the summary_op automatically or else it will consume too much memory
        sv = tf.train.Supervisor(logdir = FLAGS.log_dir, summary_op = None, init_fn=restore_fn)
        # sv = tf.train.Supervisor(logdir=FLAGS.log_dir, summary_op=None)
        #Run the managed session
        with sv.managed_session() as sess:
            epoch_loss = []
            for step in xrange(num_batches_per_epoch * FLAGS.num_epoch):
                ## run a train step
                batch_loss, global_step_count, accuracy_value, learning_rate, my_summary_ops, auc = train_step(sess, train_op, global_step, accuracy, lr, my_summary_op, train_labels, probabilities, loss)
                epoch_loss.append(batch_loss)
                #At the start of every epoch, show the vital information:
                if step % num_batches_per_epoch == 0:
                    logging.info('Epoch %s/%s', step/num_batches_per_epoch + 1, FLAGS.num_epoch)
                    # learning_rate_value, accuracy_value, auc_value = sess.run([accuracy, auc])
                    logging.info('Current Learning Rate: %s', learning_rate)
                    logging.info('Mean loss on this training epoch is: %s' % (float(sum(epoch_loss)) / max(len(epoch_loss), 1)))
                    epoch_loss[:] = []
                    logging.info('Accuracy in this training epoch is : %s', accuracy_value)
                    val_loss_arr = []
                    val_acc_arr = []
                    auc_arr = [0] * FLAGS.num_classes
                    for i in xrange(val_num_batches_per_epoch / 10): ## ok, I just want it run faster!
                        loss_values, accuracy_values, auc = val_step(sess, val_loss, val_accuracy, val_labels, val_probabilities)
                        val_loss_arr.append(loss_values)
                        val_acc_arr.append(accuracy_values)
                        # logging.info('Loss on validation batch %s is : %s' % (i, loss_values))
                        # logging.info('Accuracy on validaton batch %s is : %s' % (i, accuracy_values))
                        # logging.info('AUC on validaton batch %s is : %s' % (i, auc))
                        for idx in range(len(auc)):
                            auc_arr[idx] += auc[idx]
                    logging.info('Mean loss on this validation epoch is: %s' % (float(sum(val_loss_arr)) / max(len(val_loss_arr), 1)))
                    logging.info('Mean accuracy on this validation epoch is: %s' % (float(sum(val_acc_arr)) / max(len(val_acc_arr), 1)))
                    mean_auc = [auc / val_num_batches_per_epoch for auc in auc_arr]
                    logging.info('Mean auc on this validation epoch is: %s' % mean_auc)

                # Log the summaries every 10 step.
                if step % 10 == 0:
                    logging.info('AUC value on the last batch is : %s' % auc)
                    summaries = sess.run(my_summary_ops)
                    sv.summary_computed(sess, summaries)

            #Once all the training has been done, save the log files and checkpoint model
            logging.info('Finished training! Saving model to disk now.')
            sv.saver.save(sess, sv.save_path, global_step = sv.global_step)

if __name__ == '__main__':
    # mlog.initlog(FLAGS.log_dir)
    run()