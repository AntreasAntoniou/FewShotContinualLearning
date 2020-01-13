import torch
import torch.functional as F
import torch.nn as nn
import numpy as np

def int_to_one_hot(int_labels):
    num_output_units = torch.max(int_labels).long() + 1

    labels_one_hot = torch.zeros(int_labels.shape[0], num_output_units).long().to(int_labels.device)
    labels_one_hot.scatter_(1, int_labels.unsqueeze(dim=1), 1)

    labels_one_hot = labels_one_hot.view((-1, num_output_units))

    return labels_one_hot
