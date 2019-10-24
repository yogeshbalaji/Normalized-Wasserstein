import torch
import torch.nn as nn
import torch.nn.functional as F


class GradReverse(torch.autograd.Function):
    """
    Gradient reverse layer
    """
    @staticmethod
    def forward(ctx, x, constant):
        ctx.constant = constant
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.neg() * ctx.constant
        return grad_output, None

    def grad_reverse(x, constant):
        return GradReverse.apply(x, constant)


class _netF(nn.Module):
    """
    Feature extractor
    """
    def __init__(self, normalize=False):
        super(_netF, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 48, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.normalize = normalize

    def forward(self, input):
        input = input.expand(input.data.shape[0], 3, 28, 28)
        x = F.relu(F.max_pool2d(self.conv1(input), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 48 * 4 * 4)

        if self.normalize:
            # feature normalization
            x_norm = torch.norm(x, p=2, dim=1, keepdim=True).detach()
            x = x.div(x_norm.expand_as(x))

        return x


class _netC(nn.Module):
    """
    Classifier
    """
    def __init__(self, nclasses):
        super(_netC, self).__init__()
        self.fc1 = nn.Linear(48 * 4 * 4, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, nclasses)

    def forward(self, input):
        out = F.relu(self.fc1(input))
        out = self.fc2(F.dropout(out))
        out = F.relu(out)
        out = self.fc3(out)
        return out


class _netD(nn.Module):
    """
    Domain discriminator
    """
    def __init__(self):
        super(_netD, self).__init__()
        self.fc1 = nn.Linear(48 * 4 * 4, 100)
        self.fc2 = nn.Linear(100, 2)

    def forward(self, input):
        out = F.relu(self.fc1(input))
        out = self.fc2(out)
        return out


class _netD_wasserstein(nn.Module):
    """
    Domain discriminator for Wasserstein loss
    """
    def __init__(self):
        super(_netD_wasserstein, self).__init__()
        self.fc1 = nn.Linear(48 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 100)
        self.fc3 = nn.Linear(100, 1)

    def forward(self, input):
        logits = F.relu(self.fc1(input))
        logits = F.relu(self.fc2(logits))
        logits = self.fc3(logits)
        logits = logits.mean(0)
        return logits.view(1)

