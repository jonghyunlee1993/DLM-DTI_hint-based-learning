import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix


def compute_sen_spec(y_test, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    sensitivity = (tp / (tp + fn)).round(4)
    specificity = (tn / (tn + fp)).round(4)

    return sensitivity, specificity


def evaluate(predictions):
    davis_pred, davis_target = [], []
    binding_pred, binding_target = [], []
    biosnap_pred, biosnap_target = [], []

    for batch in predictions:
        for i in range(batch[0].shape[0]):
            pred = batch[0][i].detach().numpy().tolist()
            target = batch[1][i].detach().numpy().tolist()
            source = batch[2][i].detach().numpy().tolist()

            if source == 1:
                davis_pred.append(pred)
                davis_target.append(target)
            elif source == 2:
                binding_pred.append(pred)
                binding_target.append(target)
            elif source == 3:
                biosnap_pred.append(pred)
                biosnap_target.append(target)

    davis_pred_label = np.where(np.array(davis_pred) >= 0.5, 1, 0)
    binding_pred_label = np.where(np.array(binding_pred) >= 0.5, 1, 0)
    biosnap_pred_label = np.where(np.array(biosnap_pred) >= 0.5, 1, 0)

    try:
        davis_auroc = roc_auc_score(davis_target, davis_pred).round(4)
        davis_auprc = average_precision_score(davis_target, davis_pred).round(4)
        davis_sen, davis_spec = compute_sen_spec(davis_target, davis_pred_label)
    except:
        davis_auroc, davis_auprc, davis_sen, davis_spec = -1, -1, -1, -1

    try:
        binding_auroc = roc_auc_score(binding_target, binding_pred).round(4)
        binding_auprc = average_precision_score(binding_target, binding_pred).round(4)
        binding_sen, binding_spec = compute_sen_spec(binding_target, binding_pred_label)
    except:
        binding_auroc, binding_auprc, binding_sen, binding_spec = -1, -1, -1, -1

    try:
        biosnap_auroc = roc_auc_score(biosnap_target, biosnap_pred).round(4)
        biosnap_auprc = average_precision_score(biosnap_target, biosnap_pred).round(4)
        biosnap_sen, biosnap_spec = compute_sen_spec(biosnap_target, biosnap_pred_label)
    except:
        biosnap_auroc, biosnap_auprc, biosnap_sen, biosnap_spec = -1, -1, -1, -1

    results = [
        davis_auroc,
        davis_auprc,
        davis_sen,
        davis_spec,
        binding_auroc,
        binding_auprc,
        binding_sen,
        binding_spec,
        biosnap_auroc,
        biosnap_auprc,
        biosnap_sen,
        biosnap_spec,
    ]

    return results
