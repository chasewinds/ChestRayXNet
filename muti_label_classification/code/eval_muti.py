import matplotlib.pyplot as plt
plt.style.use('ggplot')

import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging
from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step
from inception_resnet_v2 import inception_resnet_v2, inception_resnet_v2_arg_scope
from densenet import densenet161, densenet_arg_scope
import time
import os
import mlog
from data_prepare import load_batch, get_split
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

# import matplotlib
# matplotlib.use('TkAgg')
slim = tf.contrib.slim

flags = tf.app.flags

#State your log directory where you can retrieve your model
flags.DEFINE_string('log_dir', None, 'String, log dir')

#Create a new evaluation log directory to visualize the validation process
flags.DEFINE_string('log_eval', None, 'String, dir evaluate log save')

#State the dataset directory where the validation set is found
flags.DEFINE_string('dataset_dir', None, 'String, dataset dir')

##
flags.DEFINE_string('auc_picture_path', None, 'String, the path auc picture save')

flags.DEFINE_string('tfrecord_prefix', None, 'String, the predix of tfrecord')

#State the batch_size to evaluate each time, which can be a lot more than the training batch
flags.DEFINE_integer('batch_size', 36, 'Int, batch size')

#State the number of epochs to evaluate
flags.DEFINE_integer('num_epochs', 1, 'Int, number of epochs')

flags.DEFINE_integer('num_classes', 2, 'Int, number of class need to caculate auc')

flags.DEFINE_integer('ckpt_id', 1, 'Int, last (ckpt_id)th check point file for evaluate, if ckpt_id = 1, restore last ckpt file')

FLAGS = flags.FLAGS

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

    logging.info("scores[:, 0] and scores[:, 1] is : %s, %s" % (len(scores[:, 0]), len(scores[:, 1])))
    fpr, tpr, thresholds = roc_curve(scores[:, 0], scores[:, 1], drop_intermediate=True) ## drop is true in the origin demo, if False, then the result maybe different every time.
    auc = roc_auc_score(scores[:, 0], scores[:, 1])
    logging.info("roc auc score is :%s" % auc)
    acc_thres, acc = calc_max_acc(scores[:, 0], scores[:, 1], thresholds)
    logging.info("accuracy: %f, threshold: %f" % (acc, acc_thres))

    logging.info("-----------------the false positive ratio is: , shape %s" % (len(fpr)))
    logging.info("-----------------the ture positive ratio is: , shape %s" % (len(tpr)))
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
    return auc

def parse_label(pred, label, index):
    pred = [y[index] for x in pred for y in x]
    label = [y[index] for x in label for y in x]
    return pred, label

def require_ckpt_file(log_dir=FLAGS.log_dir, ckpt_id=FLAGS.ckpt_id):
    files = os.listdir(log_dir)
    ckpt_list = [x for x in files if x.split('-')[0] == 'model.ckpt']
    return os.path.join(FLAGS.log_dir ,ckpt_list[len(ckpt_list) - ckpt_id])

def run():

    # Get the latest checkpoint file
    checkpoint_file = tf.train.latest_checkpoint(FLAGS.log_dir)

    #Create log_dir for evaluation information
    if not os.path.exists(FLAGS.log_eval):
        os.mkdir(FLAGS.log_eval)
    pred_all = []
    label_all = []
    mean_loss = []

    #Just construct the graph from scratch again
    with tf.Graph().as_default() as graph:
        tf.logging.set_verbosity(tf.logging.INFO)
        #Get the dataset first and load one batch of validation images and labels tensors. Set is_training as False so as to use the evaluation preprocessing
        file_pattern = FLAGS.tfrecord_prefix + '_%s_*.tfrecord'
        dataset = get_split('validation', FLAGS.dataset_dir, FLAGS.num_classes, file_pattern, FLAGS.tfrecord_prefix)
        images, raw_images, labels = load_batch(dataset, batch_size=FLAGS.batch_size, num_classes=FLAGS.num_classes, is_training = False)

        #Create some information about the training steps
        # assert dataset.num_samples % FLAGS.batch_size == 0, 'batch size can not be div by number sampels, the total sampels is %s' % dataset.num_samples
        num_batches_per_epoch = (dataset.num_samples - 1) / FLAGS.batch_size + 1
        num_steps_per_epoch = num_batches_per_epoch

        #Now create the inference model but set is_training=False
        # with slim.arg_scope(inception_resnet_v2_arg_scope()):
            # logits, end_points = inception_resnet_v2(images, num_classes = dataset.num_classes, is_training = False)
        with slim.arg_scope(densenet_arg_scope()):
            logits, end_points = densenet161(images, fc_dropout_rate=None, num_classes=dataset.num_classes, is_training=False)
            logits_op = logits
            sigmoid_op = tf.sigmoid(logits_op)
            # logging.info("The logits output from the model is: %s, The prediction of the model is: %s" % (end_points['Logits'], end_points['Predictions']))

        #get all the variables to restore from the checkpoint file and create the saver function to restore
        variables_to_restore = slim.get_variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
        def restore_fn(sess):
            # checkpoint_file = require_ckpt_file(log_dir=FLAGS.log_dir, ckpt_id=FLAGS.ckpt_id)
            # checkpoint_file = 'log/2muti/model.ckpt-61220'
            logging.info('From logging, checkpoint_file: %s' % checkpoint_file)
            # tf.Print(checkpoint_file, [checkpoint_file])
            return saver.restore(sess, checkpoint_file)

        #Just define the metrics to track without the loss or whatsoever
        # predictions = tf.argmax(end_points['Predictions'], 1)
        # accuracy, accuracy_update = tf.contrib.metrics.streaming_accuracy(sigmoid_op, labels) ## decleartion?
        lesion_pred = tf.cast(tf.greater_equal(sigmoid_op, 0.5), dtype=tf.float32)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(lesion_pred, labels), tf.float32))

        # one_hot_label = slim.one_hot_encoding(labels, dataset.num_classes)
        loss = tf.losses.log_loss(labels, sigmoid_op)
        # metrics_op = tf.group(accuracy_update)

        #Create the global step and an increment op for monitoring
        global_step = get_or_create_global_step()
        global_step_op = tf.assign(global_step, global_step + 1) #no apply_gradient method so manually increasing the global_step

        #Create a evaluation step function
        def eval_step(sess):
            '''
            Simply takes in a session, runs the metrics op and some logging information.
            '''
            start_time = time.time()
            global_step_count, step_logits, step_acc, step_loss, pred, label = sess.run([global_step_op, logits_op, accuracy, loss, sigmoid_op, labels])
            time_elapsed = time.time() - start_time

            #Log some information
            # logging.info('Global Step %s: Streaming Accuracy: %.4f (%.2f sec/step)', global_step_count, accuracy_value, time_elapsed)
            logging.info('The averange accuracy of this batch(total 32 samples) is: %s, run time is:%s' % (step_acc, time_elapsed))
            logging.info('The step loss is : %s' % step_loss)
            mean_loss.append(step_loss)
            ## process for predict and label
            # pred_compare = [0] * len(pred)
            # for i in range(len(pred)):
            #     pred_compare[i] = 1 if pred[i][1] > 0.5 else 0
            # pred_compare = [1 if x[1] > 0.5 else 0 for x in pred]
            # logging.info("The prediction of this batch is:%s" % pred)

            pred_pos = np.empty(FLAGS.batch_size)
            # label = [int(x) for x in label]
            # logging.info('Ground Truth of this batch is : %s' % label)

            # for i in range(len(pred)):
            #     pos_list = pred[i]
            #     pred_pos[i] = pos_list[1]
            #     label[i] = label[i]
            # pred_all.append(pred_pos)
            # label_all.append(label)

            pred_all.append(pred)
            label_all.append(label)

            # for i in range(len(step_prediction)):
            #     # pred = 'True' if predictions[i] == labels[i] else 'False'
            #     logging.info("The prediction of %s th image is : %s" % ((i, max(step_prediction[i]))))
            return step_acc

        #Define some scalar quantities to monitor
        # tf.summary.scalar('Validation_Accuracy', accuracy)
        my_summary_op = tf.summary.merge_all()

        #Get your supervisor
        sv = tf.train.Supervisor(logdir = FLAGS.log_eval, summary_op = None, saver = None, init_fn = restore_fn)

        #Now we are ready to run in one session

        with sv.managed_session() as sess:
            for step in xrange(num_steps_per_epoch * FLAGS.num_epochs):
                sess.run(sv.global_step)
                #print vital information every start of the epoch as always
                # if step % num_batches_per_epoch == 0:
                #     logging.info('Epoch: %s/%s', step / num_batches_per_epoch + 1, FLAGS.num_epochs)
                #     logging.info('Current Streaming Accuracy: %.4f', sess.run(accuracy))
                    # logging.info(loss)

                accuracy_value = eval_step(sess)
                logging.info('Step #%d, Accuracy of this step is : %.4f', step, accuracy_value)
                # #Compute summaries every 10 steps and continue evaluating
                # if step % 10 == 0:
                #     eval_step(sess, metrics_op = metrics_op)
                #     summaries = sess.run(my_summary_op)
                #     sv.summary_computed(sess, summaries)
                #
                # #Otherwise just run as per normal
                # else:

            #At the end of all the evaluation, show the final accuracy
            logging.info('Total Accuracy is : %.4f', accuracy_value)

            #Now we want to visualize the last batch's images just to see what our model has predicted
            #
            # raw_images, labels, predictions, predict_value = sess.run([raw_images, labels, predictions, pred_op])
            # pred_roc = []
            # label_roc = []
            # for batch_id in range(num_batches_per_epoch):
            #     for i in range(FLAGS.batch_size):
            #         # image, label, prediction, pred_possibility = raw_images[i], labels[i], predictions[i], round(max(predict_value[i])*100, 2)
            #         image, label, prediction, pred_possibility = raw_images[i], labels[i], predictions[i], predict_value[i][1]
            #         # pred_second = predict_value[i][1]
            #         # prediction_name, label_name = dataset.labels_to_name[prediction], dataset.labels_to_name[label]
            #         # logging.info('%s %s %s' % ('-'*20, 'next image', '-'*20))
            #         # logging.info('The prediction of this model on this image is :%s' % predict_value[i])
            #         # logging.debug('The Ground True of this image is: %s' % labels[i])
            #
            #         prediction_name = 'positive' if prediction == 1 else 'negative'
            #         label_name = 'positive' if label == 1 else 'negative'
            #         compare = 'True' if prediction_name == label_name else 'False'
            #
            #         text = 'Prediction: %s, with %.2f%% posibility predicte image is positive \n Ground Truth: %s (%s)' % (
            #         prediction_name, float(pred_possibility) * 100, label_name, compare)
            #         # text = 'Prediction:{}, with {}% belif \n Ground Truth :{}({})'.format(prediction_name, pred_possibility, label_name, compare)
            #         # logging.info(text)
            #         # img_plot = plt.imshow(image)
            #         #
            #         # # Set up the plot and hide axes
            #         # if i % 100 == 0:
            #         #     plt.title(text)
            #         #     img_plot.axes.get_yaxis().set_ticks([])
            #         #     img_plot.axes.get_xaxis().set_ticks([])
            #         #     plt.show()
            #         # plt.close()
            #
            #         pred_roc.append(predict_value[i][1])
            #         label_roc.append(labels[i])
            ## now time for roc!
            # logging.info('len pred roc %s' % len(total_pred))

            logging.info('len pred all %s' % len(pred_all))
            logging.info('len label all %s' % len(label_all))

            # logging.info('pred all %s' % pred_all)
            # logging.info('label all %s' % label_all)
            # total_pred = [item for sub in pred_all for item in sub]
            # total_label = [item for sub in label_all for item in sub]
            auc_metrics = []
            for i in range(FLAGS.num_classes):
                roc_save_path = FLAGS.auc_picture_path.split('.')[0] + str(i) + '.png'
                parsed_pred, parsed_label = parse_label(pred_all, label_all, i)
                # logging.info('the parsed predict is : %s, len is : %s' % (parsed_pred, len(parsed_pred)))
                # logging.info('the parsed lable is : %s, len is : %s' % (parsed_label, len(parsed_label)))
                auc = plot_roc(parsed_pred, parsed_label, roc_save_path)
                auc_metrics.append(auc)
            logging.info('Mean loss one validation set is : %s' % (sum(mean_loss) / float(len(mean_loss))))
            logging.info('The auc of each class is as fellow: %s' % auc_metrics)
            logging.info('Model evaluation has completed! Visit TensorBoard for more information regarding your evaluation.')

if __name__ == '__main__':
    # mlog.initlog(FLAGS.log_dir)
    run()
