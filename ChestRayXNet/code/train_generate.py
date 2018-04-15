import os
import time

import tensorflow as tf
from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step
from tensorflow.python.platform import tf_logging as logging

from sklearn.metrics import roc_auc_score
import inception_preprocessing
# import mlog
from data_prepare import get_split, load_batch
from vgg import vgg_16, vgg_arg_scope
from densenet_elu import densenet121, densenet161, densenet_arg_scope
from inception_resnet_v2 import inception_resnet_v2, inception_resnet_v2_arg_scope 
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

flags.DEFINE_string('model_type', None, 'String, select network for training and validation')

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
    if model_type != 'inception_resnet_v2':
        image_size = 224
    else:
        image_size = 299
    #Create the log directory here. Must be done here otherwise import will activate this unneededly.
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

        if FLAGS.model_type == 'vgg16':
            with slim.arg_scope(vgg_arg_scope()):
                logits, _ = vgg_16(train_images, num_classes=FLAGS.num_classes, is_training=True)
            # Define the scopes that you want to exclude for restoration
            exclude = ['vgg_16/fc8']
            variables_to_restore = slim.get_variables_to_restore(exclude=exclude)
        elif FLAGS.model_type == 'densenet121':
            with slim.arg_scope(densenet_arg_scope()):
                logits, _ = densenet121(train_images, num_classes=FLAGS.num_classes, is_training=True)
            exclude = ['densenet121/Logits', 'densenet121/final_block']
            variables_to_restore = slim.get_variables_to_restore(exclude=exclude)
        elif FLAGS.model_type == 'densenet161':
            with slim.arg_scope(densenet_arg_scope()):
                logits, _ = densenet161(train_images, num_classes=FLAGS.num_classes, is_training=True)
            exclude = ['densenet161/Logits', 'densenet161/final_block']
            variables_to_restore = slim.get_variables_to_restore(exclude=exclude)  
        elif FLAGS.model_type == 'inception_resnet_v2':
            with slim.arg_scope(inception_resnet_v2_arg_scope()):
                logits, _ = inception_resnet_v2(train_images, num_classes=FLAGS.num_classes, is_training=True)
            exclude = ['InceptionResnetV2/Logits', 'InceptionResnetV2/AuxLogits']  
            variables_to_restore = slim.get_variables_to_restore(exclude=exclude)  
        
        ## convert into probabilities
        probabilities = tf.sigmoid(logits)
        ## new loss, just equal to the sum of 14 log loss
        # loss = tf.losses.log_loss(labels=train_labels, predictions=probabilities)
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=train_labels, logits=logits))
        # loss = tf.reduce_mean(loss)
        ## convert into actual predicte
        lesion_pred = tf.cast(tf.greater_equal(probabilities, 0.5), tf.float32)

        # Create the global step for monitoring the learning_rate and training.
        global_step = get_or_create_global_step()
        ## learning rate of fine tuning show be low:
        epochs_lr = [[20, 0.0002],
                     [30, 0.0001],
                     [20, 0.00001],
                     [20, 0.000001]]
        lr = CustLearningRate.IntervalLearningRate(epochs_lr=epochs_lr,
                                                   global_step=global_step,
                                                   steps_per_epoch=num_batches_per_epoch)

        # Adam will help us modify lr during training
        optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.9, beta2=0.999, epsilon=1e-8)
        # Create the train_op.
        train_op = slim.learning.create_train_op(loss, optimizer)
        # compute accuracy
        accuracy = tf.reduce_mean(tf.cast(tf.equal(lesion_pred, train_labels), tf.float32))
        # cal auc, if possible, write it to log and show on tensorbord
        # auc, _ = tf.metrics.auc(train_labels, probabilities)

       if FLAGS.model_type == 'vgg16':
            with slim.arg_scope(vgg_arg_scope()):
                val_logits, _ = vgg_16(val_images, num_classes=FLAGS.num_classes, is_training=False, dropout_keep_prob=1, reuse=True)
        elif FLAGS.model_type == 'densenet121':
            with slim.arg_scope(densenet_arg_scope()):
                val_logits, _ = densenet121(val_images, num_classes=FLAGS.num_classes, is_training=False, dropout_keep_prob=1, reuse=True)
        elif FLAGS.model_type == 'densenet161':
            with slim.arg_scope(densenet_arg_scope()):
                val_logits, _ = densenet161(val_images, num_classes=FLAGS.num_classes, is_training=False, dropout_keep_prob=1, reuse=True)
        elif FLAGS.model_type == 'inception_resnet_v2':
            with slim.arg_scope(inception_resnet_v2_arg_scope()):
                val_logits, _ = inception_resnet_v2(val_images, num_classes=FLAGS.num_classes, is_training=False, dropout_keep_prob=1, reuse=True)
        
        val_probabilities = tf.sigmoid(val_logits)
        ## new loss, just equal to the sum of 14 log loss
        # val_loss = tf.losses.log_loss(labels=val_labels, predictions=val_probabilities)
        val_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=val_labels, logits=val_logits))
        # val_loss = tf.reduce_mean(val_loss)
        val_lesion_pred = tf.cast(tf.greater_equal(val_probabilities, 0.5), tf.float32)
        val_accuracy = tf.reduce_mean(tf.cast(tf.equal(val_lesion_pred, val_labels), tf.float32))
            # return loss, accuracy

        #at the end ,create all the summaries you need to monitor and group them into one summary op, those summery can be seen on tensorbord
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
            '''run one training batch, logging basic training log and collect auc value throught this batch'''
            start_time = time.time()
            total_loss, global_step_count, accuracy_value, learning_rate, auc_label, auc_prob, log_loss = sess.run([train_op, global_step, accuracy, lr, train_label, probability, origin_loss])
            time_elapsed = time.time() - start_time
            #logging.info("prob output from the network is : %s, label is : %s, loss from log_loss function is : %s" % (auc_prob, auc_label, log_loss))
            out_prob = [0 if y < 0.5 else 1 for x in auc_prob for y in x]
            # logging.info("DEBUG: sigmoid logits is : %s" % out_prob[0])
            auc = []
            for i in range(FLAGS.num_classes):
                sub_prob = [x[i] for x in auc_prob]
                sub_label = [x[i] for x in auc_label]
                try:
                    auc.append([i, roc_auc_score(sub_label, sub_prob)])
                except:
                    continue
            epoch = global_step_count/num_batches_per_epoch + 1
            logging.info('Epoch: %s, global step %s: learning rate: %s, LOSS: %s, accuracy: %s , (%.2f sec/step)', epoch, global_step_count, learning_rate, log_loss, accuracy_value, time_elapsed)
            # logging.info("the loss in this step is : %s" % str(int(sum(sum(log_loss))) / 14.0))
            return log_loss, global_step_count, accuracy_value, learning_rate, my_summary_op, auc

        def val_step(sess, validation_loss, validation_accuracy, val_label, val_probability):
            '''run a validation batch, logging correlate auc and loss value'''
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

        # create a saver for save and restore ckpt file 
        saver = tf.train.Saver(variables_to_restore)

        def restore_fn(sess):
            return saver.restore(sess, FLAGS.checkpoint_file)
        # Define supervisor for running a managed session. If fine tuning is used, pass restore_fn to init_fn 
        sv = tf.train.Supervisor(logdir=FLAGS.log_dir, summary_op=None, init_fn=restore_fn)
        # sv = tf.train.Supervisor(logdir=FLAGS.log_dir, summary_op=None)
        #Run the managed session
        with sv.managed_session() as sess:
            epoch_loss = []
            for step in xrange(num_batches_per_epoch * FLAGS.num_epoch):
                ## run one train step
                batch_loss, global_step_count, accuracy_value, learning_rate, my_summary_ops, auc = train_step(sess, train_op, global_step, accuracy, lr, my_summary_op, train_labels, probabilities, loss)
                epoch_loss.append(batch_loss) #collect training loss per batch
                #in every new epoch:
                if step % num_batches_per_epoch == 0:
                    logging.info('Epoch %s/%s', step/num_batches_per_epoch + 1, FLAGS.num_epoch)
                    # learning_rate_value, accuracy_value, auc_value = sess.run([accuracy, auc])
                    logging.info('Current Learning Rate: %s', learning_rate)
                    # logging.info('Mean loss on this training epoch is: %s' % (float(sum(epoch_loss)) / max(len(epoch_loss), 1)))
                    epoch_loss[:] = []
                    logging.info('Accuracy in this training epoch is : %s', accuracy_value)
                    val_loss_arr = []
                    val_acc_arr = []
                    auc_arr = [0] * FLAGS.num_classes
                    #run validation set once every epoch, the origin paper say they reduce lr every time validation loss stop decrease,
                    for i in xrange(val_num_batches_per_epoch / 10): ## ok, I just want it run faster!
                        loss_values, accuracy_values, auc = val_step(sess, val_loss, val_accuracy, val_labels, val_probabilities)
                        # logging.info("float(sum(loss_values)) = %s" % float(sum(loss_values)))
                        batch_mean_loss = float(loss_values) / FLAGS.batch_size
                        val_loss_arr.append(batch_mean_loss)
                        val_acc_arr.append(accuracy_values)
                        logging.info('Loss on validation batch %s is : %s' % (i, loss_values))
                        # logging.info('Accuracy on validaton batch %s is : %s' % (i, accuracy_values))
                        logging.info('AUC on validaton batch %s is : %s' % (i, auc))
                        for idx in range(len(auc)):
                            auc_arr[idx] += auc[idx]
                    logging.info('Mean loss on this validation epoch is: %s' % (float(sum(val_loss_arr)) / max(len(val_loss_arr), 1)))
                    logging.info('Mean accuracy on this validation epoch is: %s' % (float(sum(val_acc_arr)) / max(len(val_acc_arr), 1)))
                    # logging.info('Mean loss on this validation epoch is: %s' % (float(sum(sum(val_loss_arr))) / max(len(val_loss_arr)[0], 1)))
                    # logging.info('Mean accuracy on this validation epoch is: %s' % (float(sum(sum(val_acc_arr))) / max(len(val_acc_arr)[0], 1)))
                    mean_auc = [auc / val_num_batches_per_epoch for auc in auc_arr]
                    logging.info('Mean auc on this validation epoch is: %s' % mean_auc)

                # Log the summaries every 10 step.
                if step % 10 == 0:
                    auc_train = [0] * FLAGS.num_classes
                    logging.info('AUC value on the last batch is : %s' % auc)
                    # logging.info('The 14 subclass loss on the last batch is : %s' % sum(batch_loss))
                    summaries = sess.run(my_summary_ops)
                    sv.summary_computed(sess, summaries)

            #Once all the training has been done, save the log files and checkpoint model
            logging.info('Finished training! Saving model to disk now.')
            sv.saver.save(sess, sv.save_path, global_step = sv.global_step)

if __name__ == '__main__':
    # mlog.initlog(FLAGS.log_dir)
    run()