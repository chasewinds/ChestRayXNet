import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score, roc_curve
from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step
from tensorflow.python.platform import tf_logging as logging
plt.style.use('ggplot')

from muti_label_classification.code.inception_resnet_v2 import inception_resnet_v2, inception_resnet_v2_arg_scope
from train_flowers import get_split, load_batch

slim = tf.contrib.slim

# ======================================= prepare paramters for test.===================================================
flags = tf.app.flags

#State your log directory where you can retrieve your model
flags.DEFINE_string('log_dir', None, 'String, log dir')

#Create a new evaluation log directory to visualize the validation process
flags.DEFINE_string('log_eval', None, 'String, dir evaluate log save')

#State the dataset directory where the validation set is found
flags.DEFINE_string('dataset_dir', None, 'String, dataset dir')

##
flags.DEFINE_string('auc_picture_path', None, 'String, the path auc picture save')

#State the batch_size to evaluate each time, which can be a lot more than the training batch
flags.DEFINE_integer('batch_size', 36, 'Int, batch size')

#State the number of epochs to evaluate
flags.DEFINE_integer('num_epochs', 1, 'Int, number of epochs')
FLAGS = flags.FLAGS

# ======================================= plot roc curve and caculate auc ==============================================
def calc_acc(thres, gt, preds):
    pred_label = np.zeros(preds.shape[0], dtype=np.int)
    pred_label[preds >= thres] = 1
    pred_label[preds < thres] = 0
    return np.sum(pred_label == gt) / float(len(gt))

def calc_max_acc(gt, preds, thresholds=None):
    if thresholds is not None:
        _, _, thresholds = roc_curve(gt, preds)
    acc_threds = [[t, calc_acc(t, gt, preds)] for t in thresholds]
    acc_threds = sorted(acc_threds, key=lambda x: x[1], reverse=True)
    return acc_threds[0]

def plot_roc(pred, label, auc_picture_path):
    scores_files = [(label[i], pred[i]) for i in range(len(label))]
    logging.info('len scores files is %s' % len(scores_files))
    scores = np.array([(float(x[0]), float(x[1])) for x in scores_files if x[1] != None])
    logging.info("%s image ok, %d fail" % (scores.shape[0], len(label) - scores.shape[0]))

    # logging.info("scores list shape before feed into roc_curve is :%s, %s" % (scores.shape[0], scores.shape[1]))
    # logging.info("the variable feed into roc_curve is: %s, %s" % (scores[:, 0], scores[:, 1]))
    fpr, tpr, thresholds = roc_curve(scores[:, 0], scores[:, 1], drop_intermediate=False)
    auc = roc_auc_score(scores[:, 0], scores[:, 1])
    logging.info("roc auc score is :%s" % auc)
    # logging.info('the len of auc is: %s' % len(auc))
    # logging.info("auc: %f" % auc)
    acc_thres, acc = calc_max_acc(scores[:, 0], scores[:, 1], thresholds)
    logging.info("accuracy: %f, threshold: %f" % (acc, acc_thres))

    logging.info("-----------------the false positive ratio is: %s, shape %s" % (fpr, len(fpr)))
    logging.info("-----------------the ture positive ratio is: %s, shape %s" % (tpr, len(tpr)))
    logging.info("-----------------the auc which used as label is: %s" % auc)

    plt.plot(fpr, tpr, color='red', label=("auc: %f" % auc))
    plt.plot([0, 1], [0, 1], color='blue', linewidth=2, linestyle='--')
    plt.title('Classification Test on X-Chest: %s' % 'positive VS negative')
    plt.legend(loc='lower right')
    if len(auc_picture_path) > 0:
        logging.info('be about to save')
        plt.savefig(auc_picture_path)

    plt.show()
    plt.close()

# ============================================= build graph ============================================================
def build_graph(checkpoint_file):
    with tf.Graph().as_default() as graph:
        tf.logging.set_verbosity(tf.logging.INFO)
        # Get the dataset first and load one batch of validation images and labels tensors. Set is_training as False so as to use the evaluation preprocessing
        dataset = get_split('validation', FLAGS.dataset_dir)
        images, raw_images, labels = load_batch(dataset, batch_size=FLAGS.batch_size, is_training=False)

        # Create some information about the training steps
        num_batches_per_epoch = dataset.num_samples / FLAGS.batch_size
        num_steps_per_epoch = num_batches_per_epoch

        # Now create the inference model but set is_training=False
        with slim.arg_scope(inception_resnet_v2_arg_scope()):
            logits, end_points = inception_resnet_v2(images, num_classes=dataset.num_classes, is_training=False)
            logits_op = end_points['Logits']
            pred_op = end_points['Predictions']
            # logging.info("The logits output from the model is: %s, The prediction of the model is: %s" % (end_points['Logits'], end_points['Predictions']))

        # #get all the variables to restore from the checkpoint file and create the saver function to restore
        variables_to_restore = slim.get_variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        def restore_fn(sess):
            return saver.restore(sess, checkpoint_file)

        # Just define the metrics to track without the loss or whatsoever
        predictions = tf.argmax(end_points['Predictions'], 1)
        accuracy, accuracy_update = tf.contrib.metrics.streaming_accuracy(predictions, labels)  ## decleartion?
        acc_mine = tf.reduce_mean(tf.cast(tf.equal(predictions, labels), tf.float32))
        metrics_op = tf.group(accuracy_update)

        def get_pred_and_label(sess):
            pred = sess.run([pred_op])
            label = sess.run([labels])
            label = label[0]
            # logging.info('--------visulizing the pred: %s' % pred)
            # logging.info('--------get the shape of pred: %s' % pred[0][0][1])
            pred_pos = np.empty(FLAGS.batch_size)
            for i in range(len(pred)):
                pos_list = pred[0][i]
                pred_pos[i] = pos_list[1]
                label[i] = float(label[i])
            # logging.info('--------visulizing the pred: %s' % type(pred_pos))
            logging.info('--------visulizing the label: %s' % label)
            # logging.info('--------visulizing the label: %s' % type(label))
            return pred_pos, label

        # Create the global step and an increment op for monitoring
        global_step = get_or_create_global_step()
        global_step_op = tf.assign(global_step,
                                   global_step + 1)  # no apply_gradient method so manually increasing the global_step

        # Create a evaluation step function
        def eval_step(sess, metrics_op):
            '''
            Simply takes in a session, runs the metrics op and some logging information.
            '''
            start_time = time.time()
            _, global_step_count, accuracy_value, step_logits, step_prediction, step_acc = sess.run(
                [metrics_op, global_step_op, accuracy,
                 logits_op, pred_op, acc_mine])
            time_elapsed = time.time() - start_time

            # Log some information
            # logging.info('Global Step %s: Streaming Accuracy: %.4f (%.2f sec/step)', global_step_count, accuracy_value, time_elapsed)
            logging.info('The averange accuracy of this batch(total 36 samples) is: %s' % step_acc)
            # for i in range(len(step_prediction)):
            #     # pred = 'True' if predictions[i] == labels[i] else 'False'
            #     logging.info("The prediction of %s th image is : %s" % ((i, max(step_prediction[i]))))


            return accuracy_value

        # Define some scalar quantities to monitor
        tf.summary.scalar('Validation_Accuracy', accuracy)
        my_summary_op = tf.summary.merge_all()

def main(restore_fn):
    # Get your supervisor
    sv = tf.train.Supervisor(logdir=FLAGS.log_eval, summary_op=None, saver=None, init_fn=restore_fn)

    # Now we are ready to run in one session
    total_pred = np.empty(dataset.num_samples)
    total_label = np.empty(dataset.num_samples)
    with sv.managed_session() as sess:
        for step in xrange(num_steps_per_epoch * FLAGS.num_epochs):
            sess.run(sv.global_step)
            # print vital information every start of the epoch as always
            if step % num_batches_per_epoch == 0:
                logging.info('Epoch: %s/%s', step / num_batches_per_epoch + 1, FLAGS.num_epochs)
                logging.info('Current Streaming Accuracy: %.4f', sess.run(accuracy))

            # Compute summaries every 10 steps and continue evaluating
            if step % 10 == 0:
                eval_step(sess, metrics_op=metrics_op)
                summaries = sess.run(my_summary_op)
                sv.summary_computed(sess, summaries)
                pred, label = get_pred_and_label(sess)
                np.hstack((total_pred, pred))
                np.hstack((total_label, label))
            # Otherwise just run as per normal
            else:
                eval_step(sess, metrics_op=metrics_op)
                pred, label = get_pred_and_label(sess)
                np.hstack((total_pred, pred))
                np.hstack((total_label, label))

        # At the end of all the evaluation, show the final accuracy
        logging.info('Final Streaming Accuracy: %.4f', sess.run(accuracy))