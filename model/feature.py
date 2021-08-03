import torch
from torch import nn
import torch.nn.functional as F

from base import BaseModel


class Feature(BaseModel):
    def __init__(self):
        super(Feature, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.batchnorm_1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.batchnorm_2 = nn.BatchNorm2d(32)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv4 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1
        )

        self.conv5 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1
        )

        self.conv6 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1
        )
        self.upsize6 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=1, stride=2, padding=0
        )

        self.conv7 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
        )

        self.conv8 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1
        )
        self.upsize8 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=1, stride=2, padding=0
        )

        self.conv9 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1
        )

        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.6)
        self.dense10 = nn.Linear(128 * 16 * 8, 128)
        self.batchnorm_10 = nn.BatchNorm1d(128)

    def forward(self, x):
        x = F.elu(self.batchnorm_1(self.conv1(x)))

        x = F.elu(self.batchnorm_2(self.conv2(x)))

        x = F.elu(self.pool3(x))

        # Residual 4
        res4 = F.elu(self.conv4(x))
        x = x + res4
        x = F.elu(x)

        # Residual 5
        res5 = F.elu(self.conv5(x))
        x = x + res5
        x = F.elu(x)

        # Residual 6
        res6 = F.elu(self.conv6(x))
        x = F.elu(self.upsize6(x)) + res6
        x = F.elu(x)

        # Residual 7
        res7 = F.elu(self.conv7(x))
        x = x + res7
        x = F.elu(x)

        # Residual 8
        res8 = F.elu(self.conv8(x))
        x = F.elu(self.upsize8(x)) + res8
        x = F.elu(x)

        # Residual 9
        res9 = F.elu(self.conv9(x))
        x = x + res9
        x = F.elu(x)

        flatten = self.flatten(x)
        x = self.dropout(x)
        x = F.elu(self.dense10(flatten))
        x = self.batchnorm_10(x)
        l2_output = F.normalize(x, p=2, dim=1)

        return l2_output
