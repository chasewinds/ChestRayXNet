import matplotlib.pyplot as plt
plt.style.use('ggplot')

import os
import logging
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


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

def get_auc(pred, label):
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
    return auc, fpr, tpr

def plot_roc_curve(fpr, tpr, auc, auc_picture_path):
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