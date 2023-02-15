import collections
import numpy as np
import pandas as pd
import copy, pdb, time, warnings, torch


from torch import nn
from torch.utils import data
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score
warnings.filterwarnings('ignore')


class EvalMetric(object):
    def __init__(self, multilabel=False):
        self.multilabel = multilabel
        self.pred_list = list()
        self.truth_list = list()
        self.top_k_list = list()
        self.loss_list = list()
        self.demo_list = list()
        self.speaker_list = list()
        
    def append_classification_results(
        self, 
        labels,
        outputs,
        loss=None,
        demographics=None,
        speaker_id=None
    ):
        predictions = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        for idx in range(len(predictions)):
            self.pred_list.append(predictions[idx])
            self.truth_list.append(labels.detach().cpu().numpy()[idx])
        if loss is not None: self.loss_list.append(loss.item())
        if demographics is not None: self.demo_list.append(loss.item())
        if speaker_id is not None: self.speaker_list.append(loss.item())
        
    def classification_summary(
        self, return_auc: bool=False
    ):
        result_dict = dict()
        result_dict['acc'] = accuracy_score(self.truth_list, self.pred_list)*100
        result_dict['uar'] = recall_score(self.truth_list, self.pred_list, average="macro")*100
        result_dict['top5_acc'] = (np.sum(self.top_k_list == np.array(self.truth_list).reshape(len(self.truth_list), 1)) / len(self.truth_list))*100
        result_dict['conf'] = np.round(confusion_matrix(self.truth_list, self.pred_list, normalize='true')*100, decimals=2)
        result_dict["loss"] = np.mean(self.loss_list)
        result_dict["sample"] = len(self.truth_list)
        return result_dict

    def demographic_parity(self, y_true, y_pred, sensitive_feature):
        """
        Calculate demographic parity metric for multi-class labels.
        Args:
            y_true (array): True labels.
            y_pred (array): Predicted labels.
            sensitive_feature (array): Sensitive attribute.
        Returns:
            demographic_parity (float): Demographic parity metric.
        """
        # Compute the proportion of each class for each group
        y_true = np.array(self.truth_list)
        y_pred = np.array(self.pred_list)
        sensitive_feature = self.demo_list

        classes = np.unique(y_true)
        prop_group1 = np.zeros(len(classes))
        prop_group2 = np.zeros(len(classes))
        for i, c in enumerate(classes):
            group1 = y_pred[(sensitive_feature == "male") & (y_true == c)]
            group2 = y_pred[(sensitive_feature == "female") & (y_true == c)]
            prop_group1[i] = len(group1) / sum(sensitive_feature == "male")
            prop_group2[i] = len(group2) / sum(sensitive_feature == "female")

        # Compute the average difference in proportions
        demographic_parity = np.mean(np.abs(prop_group1 - prop_group2))
        pdb.set_trace()
        return demographic_parity

