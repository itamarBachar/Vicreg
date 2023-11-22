import torch
import torch.nn as nn
import torch.nn.functional as F

from resnet import ResNet50


class Projector(nn.Module):
    def __init__(self):
        super(Projector, self).__init__()
        self.linear1 = nn.Linear(2048, 8192)
        self.linear2 = nn.Linear(8192, 8192)
        self.linear3 = nn.Linear(8192, 8192)
        self.batch_norm1 = nn.BatchNorm1d(8192)
        self.batch_norm2 = nn.BatchNorm1d(8192)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.batch_norm2(x)
        x = self.relu(x)
        x = self.linear3(x)
        return x


class VICReg(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.num_features = 8192
        self.embedding = 2048
        self.backbone = ResNet50()
        self.projector = Projector()

    def forward(self, x):
        # Apply batch normalization to the input only in the first forward pass
        x1, x2 = x[0]
        x1 = self.projector(self.backbone(x1))
        x2 = self.projector(self.backbone(x2))

        repr_loss = F.mse_loss(x1, x2)
        # x = torch.cat(FullGatherLayer.apply(x), dim=0)
        # y = torch.cat(FullGatherLayer.apply(y), dim=0)
        x1_mean = x1.mean(dim=0, keepdim=True)
        x2_mean = x2.mean(dim=0, keepdim=True)
        x1 = x1 - x1_mean
        x2 = x2 - x2_mean

        # Calculate the standard deviation for each input tensor along dim=0
        std_x1 = torch.sqrt(x1.var(dim=0) + 0.0001)
        std_x2 = torch.sqrt(x2.var(dim=0) + 0.0001)

        # Calculate the loss based on the standard deviations
        std_loss = (torch.mean(F.relu(1 - std_x1)) + torch.mean(F.relu(1 - std_x2))) / 2
        cov_x1 = (x1.T @ x2) / (self.args["batch_size"] - 1)
        cov_x2 = (x2.T @ x2) / (self.args["batch_size"] - 1)
        cov_loss = off_diagonal(cov_x1).pow_(2).sum().div(
            self.num_features
        ) + off_diagonal(cov_x2).pow_(2).sum().div(self.num_features)

        loss = (
                self.args["sim_coeff"] * repr_loss
                + self.args["std_coeff"] * std_loss
                + self.args["cov_coeff"] * cov_loss
        )
        return loss


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
