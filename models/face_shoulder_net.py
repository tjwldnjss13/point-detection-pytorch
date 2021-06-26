import torch.nn as nn

from torchvision.models import resnet34


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, use_batchnorm=True, use_pooling=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Sequential()
        self.acti = nn.LeakyReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2) if use_pooling else nn.Sequential()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.acti(x)
        x = self.pool(x)
        return x


class Linear(nn.Module):
    def __init__(self, in_channels, out_channels, use_dropout=True):
        super().__init__()
        self.lin = nn.Linear(in_channels, out_channels)
        self.dropout = nn.Dropout() if use_dropout else nn.Sequential()
        self.acti = nn.ReLU(True)

    def forward(self, x):
        x = self.lin(x)
        x = self.acti(x)
        x = self.dropout(x)

        return x


class FSNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = resnet34(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(1, 64, 7, 2, 3)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 6 * 2)
        # self.convs = nn.Sequential(
        #     Conv(1, 32),
        #     Conv(32, 64),
        #     Conv(64, 96),
        #     Conv(96, 128),
        #     Conv(128, 256),
        #     Conv(256, 512, use_pooling=False)
        # )
        # self.fcs = nn.Sequential(
        #     Linear(7 * 7 * 512, 512),
        #     nn.Linear(512, 12)
        # )

    def forward(self, x):
        x = self.resnet(x)
        # x = self.convs(x)
        # x = x.view(x.shape[0], -1)
        # x = self.fcs(x)

        return x


if __name__ == '__main__':
    from torchsummary import summary
    model = FSNet().cuda()
    summary(model, (1, 224, 224))
    print(model)