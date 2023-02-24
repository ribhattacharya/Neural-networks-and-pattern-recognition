import torch.nn as nn
import torch


class Conv(nn.Module):
    '''
    Custom class to implement convolution layer, ReLU non-linearity and Batch Normalization in a single block.
    forward method has been inherited from the pytorch nn.Module class.
    '''

    def __init__(self, in_dim=None, out_dim=None):
        super().__init__()

        self.sequence = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_dim)
        )

    def forward(self, x):
        return self.sequence(x)


class deConv(nn.Module):
    '''
    Custom class to implement deconvolution layer, ReLU non-linearity and Batch Normalization in a single block.
    forward method has been inherited from the pytorch nn.Module class.
    '''

    def __init__(self, in_dim=None, out_dim=None, kernel_size=2, stride=2):
        super().__init__()

        self.sequence = nn.Sequential(
            nn.ConvTranspose2d(in_dim, out_dim, kernel_size=kernel_size, stride=stride),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_dim)
        )

    def forward(self, x):
        return self.sequence(x)


class DoubleConv(nn.Module):
    '''
    Custom class to implement convolution block with 2 layers, each followed by ReLU non-linearity and Batch Normalization in a single block.
    forward method has been inherited from the pytorch nn.Module class.
    '''

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.bn1(self.relu(self.conv1(x)))
        x = self.bn2(self.relu(self.conv2(x)))
        return x


class FC8(nn.Module):

    def __init__(self, n_class):
        super().__init__()
        self.n_class = n_class

        # Convolution layers
        self.conv1 = DoubleConv(3, 64)  # (64,224,224)

        self.mpool = nn.MaxPool2d(2, 2, 0)  # (64,112,112)

        self.conv2 = DoubleConv(64, 128)  # (128, 112,112)

        self.conv3 = nn.Sequential(DoubleConv(128, 256), Conv(256, 256))  # (256,56,56)

        self.conv4 = nn.Sequential(DoubleConv(256, 512), Conv(512, 512))  # (512,28,28)

        self.conv5 = nn.Sequential(DoubleConv(512, 512), Conv(512, 512))  # (512,14,14)

        self.conv6 = DoubleConv(512, 4096)  # (4096,7,7)

        self.conv7 = Conv(4096, self.n_class)  # (n_class,7,7)

        self.conv8 = Conv(512, self.n_class)
        self.conv9 = Conv(256, self.n_class)
        self.conv10 = Conv(128, self.n_class)
        self.conv11 = Conv(64, self.n_class)

        # Deconvolution layers
        self.deconv_2n = deConv(2 * self.n_class, self.n_class)
        self.deconv_n = deConv(self.n_class, self.n_class)

    def forward(self, x):
        out1 = self.conv1(x)  # (64,224,224)
        out2 = self.conv2(self.mpool(out1))  # (128,112,112)
        out3 = self.conv3(self.mpool(out2))  # (256,56,56)
        out4 = self.conv4(self.mpool(out3))  # (512,28,28)
        out5 = self.conv5(self.mpool(out4))  # (512,14,14)
        out6 = self.conv6(self.mpool(out5))  # (4096,7,7)
        out7 = self.conv7(out6)  # (n_class,7,7)

        out8 = self.deconv_n(out7)  # (n_class,7,7)
        out9 = self.deconv_2n(torch.cat((out8, self.conv8(self.mpool(out4))), dim=1))  # (n_class,28,28)
        out10 = self.deconv_2n(torch.cat((out9, self.conv9(self.mpool(out3))), dim=1))  # (n_class,56,56)
        out11 = self.deconv_2n(torch.cat((out10, self.conv10(self.mpool(out2))), dim=1))  # (n_class,112,112)
        y = self.deconv_2n(torch.cat((out11, self.conv11(self.mpool(out1))), dim=1))  # (n_class,224,224)

        return y  # size=(N, n_class, H, W)