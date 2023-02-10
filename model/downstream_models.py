
import pdb
import torch
import itertools

from torch import nn
from torch import Tensor
from collections import OrderedDict
from typing import Optional, Callable
from torch.nn import functional as F
from torch.nn.functional import normalize
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pack_padded_sequence

class SelfAttentionPooling(nn.Module):
    """
    Implementation of SelfAttentionPooling
    Original Paper: Self-Attention Encoding and Pooling for Speaker Recognition
    https://arxiv.org/pdf/2008.01077v1.pdf
    """
    def __init__(self, input_dim):
        super(SelfAttentionPooling, self).__init__()
        self.W = nn.Linear(input_dim, 1)
        self.softmax = nn.functional.softmax

    def forward(self, batch_rep, att_mask=None):
        """
            N: batch size, T: sequence length, H: Hidden dimension
            input:
                batch_rep : size (N, T, H)
            attention_weight:
                att_w : size (N, T, 1)
            return:
                utter_rep: size (N, H)
        """
        att_logits = self.W(batch_rep).squeeze(-1)
        if att_mask is not None:
            att_logits = att_mask + att_logits
        att_w = self.softmax(att_logits, dim=-1).unsqueeze(-1)
        utter_rep = torch.sum(batch_rep * att_w, dim=1)

        return utter_rep


class CNNSelfAttention(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=128,
        kernel_size=1,
        padding=0,
        pooling=5,
        dropout=0.2,
        output_class_num=4,
        conv_layer=3,
        pooling_method="att"
    ):
        super(CNNSelfAttention, self).__init__()
        
        # point-wise convolution
        if conv_layer == 3:
            self.model_seq = nn.Sequential(
                nn.Conv1d(input_dim, hidden_dim, kernel_size, padding=padding),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=padding),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=padding),
            )
        else:
            self.model_seq = nn.Sequential(
                nn.Conv1d(input_dim, hidden_dim, kernel_size, padding=padding),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=padding)
            )
        self.pooling_method = pooling_method
        self.pooling = SelfAttentionPooling(hidden_dim)
        self.weights = nn.Parameter(torch.zeros(12))
        self.out_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_class_num),
        )
        
    def forward(self, features, l=None, att_mask=None):
        # weighted
        features = features[1:]
        _, *origin_shape = features.shape
        # return transformer enc outputs [B, 12, T, D]
        features = features.view(12, -1)
        norm_weights = F.softmax(self.weights, dim=-1)
        
        # weight average
        weighted_feature = (norm_weights.unsqueeze(-1) * features).sum(dim=0)
        features = weighted_feature.view(*origin_shape)

        # weight average
        features = features.transpose(1, 2)
        features = self.model_seq(features)
        out = features.transpose(1, 2)
        if self.pooling_method == "att":
            out = self.pooling(out, att_mask).squeeze(-1)
        else:
            out = torch.mean(out, dim=1)
            pdb.set_trace()
        
        # output predictions
        predicted = self.out_layer(out)
        return predicted


class DNNClassifier(nn.Module):
    def __init__(self, num_class):
        super(DNNClassifier, self).__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Linear(128, num_class),
        )
        
    def forward(self, x):
        # the features are like a spectrogram, an image with one channel
        feat = x.mean(dim=1)
        output = self.classifier(feat)
        return output

class RNNClassifier(nn.Module):
    def __init__(
        self, 
        num_class:      int,        # Number of classes 
        input_dim:      int=768,    # feature input dim
        hidden_dim:     int=256,    # Hidden Layer size
    ):
        super(RNNClassifier, self).__init__()
        self.dropout_p = 0.4
        
        # RNN module
        self.rnn = nn.LSTM(
            input_size=256, 
            hidden_size=hidden_dim//2,
            num_layers=1, 
            batch_first=True, 
            dropout=self.dropout_p, 
            bidirectional=True
        )
        self.projector = nn.Linear(768, 256)
        self.pooling = SelfAttentionPooling(hidden_dim)
        # self.pooling = BaseSelfAttention(d_hid=hidden_dim, d_head=6)
    

        self.out_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, num_class),
        )
        
         # Projection head
        self.init_weight()

    def init_weight(self):
        for m in self._modules:
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)
            if type(m) == nn.Conv1d:
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)

    def forward(
        self, x, l, att_mask=None
    ):
        # 1. Rnn forward
        # x = F.relu(x)
        # x = pack_padded_sequence(
        #    x, l.cpu().numpy(), batch_first=True, enforce_sorted=False
        # )
        # x = self.projector(x)
        out, _ = self.rnn(x)
        # out, _ = pad_packed_sequence(   
        #    x, batch_first=True
        # )

        # if out.shape[1] != 299:
        # pdb.set_trace()
        out = self.pooling(out, att_mask).squeeze(-1)
        # pdb.set_trace()
        # out = out[:, :l, :].mean(dim=1)
        # out = self.pooling(out, l)
        # out = torch.mean(out, dim=1)
        predicted = self.out_layer(out)
    
        return predicted


    
    