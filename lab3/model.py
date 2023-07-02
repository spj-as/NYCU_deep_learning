import torch.nn as nn


class EEGNet(nn.Module):
    def __init__(self, activation=nn.ELU() ):
        super(EEGNet, self).__init__()

        # Layer 1
        self.firstconv = nn.Sequential(
            nn.Conv2d(1, 16, (1, 51), stride=(1, 1), padding=(0, 25), bias=False),
            nn.BatchNorm2d(16, eps=1e-05, momentum=0.1),
        )

        # Layer 2
        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(16, 32, (2, 1), stride=(1, 1), groups=16, bias=False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1),
            activation,
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4), padding=0),
            nn.Dropout(p=0.25),
        )

        # Layer 3
        self.separableConv = nn.Sequential(
            nn.Conv2d(32, 32, (1, 15), stride=(1, 1), padding=(0, 7), bias=False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1),
            activation,
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8), padding=0),
            nn.Dropout(p=0.25),
        )

        # FC Layer
        self.classify = nn.Sequential(nn.Flatten(), nn.Linear(736, 2, bias=True))

    def forward(self, x):

        x = self.firstconv(x)
        x = self.depthwiseConv(x)
        x = self.separableConv(x)
        x = self.classify(x)
        return x


class DeepConvNet(nn.Module):
    def __init__(self, activation=nn.ELU()):
        super(DeepConvNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 25, kernel_size=(1, 5))
        self.conv2 = nn.Sequential(
            nn.Conv2d(25, 25, kernel_size=(2, 1)),
            nn.BatchNorm2d(25, eps=1e-5, momentum=0.1),
            activation,
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=0.5),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(25, 50, kernel_size=(1, 5)),
            nn.BatchNorm2d(50, eps=1e-5, momentum=0.1),
            activation,
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=0.5),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(50, 100, kernel_size=(1, 5)),
            nn.BatchNorm2d(100, eps=1e-5, momentum=0.1),
            activation,
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=0.5),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(100, 200, kernel_size=(1, 5)),
            nn.BatchNorm2d(200, eps=1e-5, momentum=0.1),
            activation,
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=0.5),
        )
        self.out = nn.Sequential(nn.Flatten(), nn.Linear(8600, 2))

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.out(x)
        return x
