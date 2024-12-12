import numpy as np
from sklearn import metrics


def compute_metrics(probas_pred, target):
    pred = (probas_pred >= 0.5).astype(np.int)
    acc = metrics.accuracy_score(target, pred)
    auroc = metrics.roc_auc_score(target, probas_pred)
    f1 = metrics.f1_score(target, pred)
    ap = metrics.average_precision_score(target, probas_pred)
    precision = metrics.precision_score(target, pred)
    recall = metrics.recall_score(target, pred)
    return acc, auroc, f1, precision, recall, ap


def count_positive(y_true):
    return np.sum(y_true == 1)


def count_negative(y_true):
    return np.sum(y_true == 0)


def count_true_positive(y_true, y_pred):
    return np.sum(np.logical_and(y_true == 1, y_pred == 1))


def count_false_positive(y_true, y_pred):
    return np.sum(np.logical_and(y_true == 0, y_pred == 1))


def count_true_negative(y_true, y_pred):
    return np.sum(np.logical_and(y_true == 0, y_pred == 0))


def count_false_negative(y_true, y_pred):
    return np.sum(np.logical_and(y_true == 1, y_pred == 0))


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def sensitivity(y_true, y_pred):
    tp = count_true_positive(y_true, y_pred)
    p = count_positive(y_true) + 1e-9
    return tp / p


def specificity(y_true, y_pred):
    tn = count_true_negative(y_true, y_pred)
    n = count_negative(y_true) + 1e-9
    return tn / n


def precision_score(y_true, y_pred):
    tp = count_true_positive(y_true, y_pred)
    fp = count_false_positive(y_true, y_pred)
    return tp / (tp + fp)


def recall_score(y_true, y_pred):
    tp = count_true_positive(y_true, y_pred)
    fn = count_false_negative(y_true, y_pred)
    return tp / (tp + fn)


def f1(y_true, y_pred):
    prec = precision_score(y_true, y_pred)
    reca = recall_score(y_true, y_pred)
    return 2 * (prec * reca) / (prec + reca)
