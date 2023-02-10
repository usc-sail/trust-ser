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
        
    def append_classification_results(
        self, 
        labels,
        outputs,
        loss
    ):
        predictions = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        for idx in range(len(predictions)):
            self.pred_list.append(predictions[idx])
            self.truth_list.append(labels.detach().cpu().numpy()[idx])
        self.loss_list.append(loss.item())
        
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
