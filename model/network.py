import torch
from torch import nn
import torch.nn.functional as F

from .feature import Feature

from base import BaseModel


def truncated_normal_(tensor, mean=0, std=1):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        truncated_normal_(m.weight.data, mean=0, std=1e-3)
        m.bias.data.fill_(0.0)


class Network(BaseModel):
    def __init__(self, num_classes):
        super(Network, self).__init__()
        self.num_classes = num_classes

        self.feature = Feature()

        self.weight = nn.utils.weight_norm(nn.Linear(128, self.num_classes))
        self.scale = nn.parameter.Parameter(torch.tensor(0.0), requires_grad=True)

    def forward(self, x):
        feat = self.feature(x)

        logits = self.weight(feat)

        scale_softplus = F.softplus(self.scale)

        logits = scale_softplus * logits

        return logits
