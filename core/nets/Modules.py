import torch
import torch.nn as nn
import torch.nn.functional as F

from core.layers.ConvLstmCell import ConvLstm_Cell


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False):
        super(ConvBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                      stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.conv(x)


class DeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding=0):
        super(DeconvBlock, self).__init__()

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding, output_padding=output_padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.deconv(x)


class ProbMotionEncoderCNN(nn.Module):
    def __init__(self, in_channels=3, features=(64, 128, 256, 512), kernel_size=3, stride=1, padding=1,
                 frame_size=(256, 256), bias=False):
        super(ProbMotionEncoderCNN, self).__init__()

        self.out_channels = features[-1]
        self.out_size = (frame_size[0]//(2**len(features)), frame_size[1]//(2**len(features)))

        self.downs = nn.Sequential()
        for feature in range(0, len(features)):
            if feature == 0:
                self.downs.add_module("ConvBlock{}".format(feature + 1),
                                      ConvBlock(in_channels=in_channels, out_channels=features[feature],
                                                kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
            else:
                self.downs.add_module("ConvBlock{}".format(feature + 1),
                                      ConvBlock(in_channels=features[feature-1], out_channels=features[feature],
                                                kernel_size=kernel_size, stride=stride, padding=padding))

    def forward(self, x):
        # X is a frame sequence (batch_size, num_channels, seq_len, height, width)
        batch_size, _, seq_len, height, width = x.size()
        # Initialize output
        output = torch.zeros(batch_size, self.out_channels, seq_len, self.out_size[0], self.out_size[1], device=device)

        # Unroll over time steps
        for time_step in range(seq_len):
            out = self.downs(x[:, :, time_step])
            output[:, :, time_step] = out
        return output


class ProbMotionEncoderLSTM(nn.Module):
    def __init__(self, in_channels=512, features=(512, 512, 512), kernel_size=(3, 3), padding=(1, 1), activation="relu",
                 frame_size=(32, 32)):
        super(ProbMotionEncoderLSTM, self).__init__()
        self.lstm = nn.Sequential()

        self.lstm.add_module("ConvLSTM1",
                             ConvLstm_Cell(in_channels=in_channels, out_channels=features[0], padding=padding,
                                      kernel_size=kernel_size, activation=activation, frame_size=frame_size))
        for layer in range(1, len(features)):
            self.lstm.add_module("ConvLSTM{}".format(layer+1),
                                 ConvLstm_Cell(in_channels=features[layer-1], out_channels=features[layer], padding=padding,
                                          kernel_size=kernel_size,  activation=activation, frame_size=frame_size))

    def forward(self, x):
        output = self.lstm(x)
        output = nn.Sigmoid()(output[:, :, -1])
        return output


class Generator(nn.Module):
    def __init__(self, in_channels=512, out_channels=3, features=(512, 256, 128, 64), kernel_size=3, stride=2,
                 padding=1, output_padding=1):
        super(Generator, self).__init__()

        self.gen = nn.Sequential()
        for feature in range(0, len(features)):
            if feature == 0:
                self.gen.add_module("DeconvBlock{}".format(feature + 1),
                                    DeconvBlock(in_channels=in_channels, out_channels=features[feature],
                                                kernel_size=kernel_size, stride=stride, padding=padding,
                                                output_padding=output_padding))

            else:
                self.gen.add_module("DeconvBlock{}".format(feature+1),
                                    DeconvBlock(in_channels=features[feature-1], out_channels=features[feature],
                                                kernel_size=kernel_size, stride=stride, padding=padding,
                                                output_padding=output_padding))

        self.dwise_deconv = nn.ConvTranspose2d(in_channels=features[-1], out_channels=out_channels,
                                               kernel_size=kernel_size, stride=1, padding=padding, output_padding=0)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x, activation="sigmoid"):
        x = self.gen(x)
        if activation == "sigmoid":
            x = nn.Sigmoid()(self.bn(self.dwise_deconv(x)))
        elif activation == "tanh":
            x = nn.Tanh()(self.bn(self.dwise_deconv(x)))
        return x


class FlowEstimator(nn.Module):
    def __init__(self, in_channels=6, out_channels=2, features=(64, 128, 256, 512), kernel_size=3, stride=1, padding=1):
        super(FlowEstimator, self).__init__()

        self.flow_estimator = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=features[0], kernel_size=kernel_size, stride=stride,
                      padding=padding),
            nn.BatchNorm2d(features[0]),
            nn.ReLU(inplace=True),
        )

        for feature in range(1, len(features)):
            self.flow_estimator.add_module("ConvBlock{}".format(feature),
                                           ConvBlock(in_channels=features[feature-1], out_channels=features[feature],
                                                     kernel_size=kernel_size, stride=stride, padding=padding))
        for feature in range(1, len(features)):
            self.flow_estimator.add_module("DeconvBlock{}".format(feature),
                                           DeconvBlock(in_channels=features[len(features)-feature],
                                                       out_channels=features[len(features)-feature-1],
                                                       kernel_size=2, stride=2, padding=0))

        self.dwise_conv = nn.Conv2d(in_channels=features[0], out_channels=out_channels, kernel_size=kernel_size,
                                    stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x, inp):
        x = torch.cat([x, inp], dim=1)
        x = self.flow_estimator(x)
        x = self.bn(self.dwise_conv(x))
        return nn.Tanhshrink()(x)


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=(32, 64, 64, 64, 64), kernel_size=4, stride=2, padding=0, bias=False):
        super(Discriminator, self).__init__()

        self.disc = nn.Sequential()
        for feature in range(0, len(features)):
            if feature == 0:
                self.disc.add_module("ConvBlock{}".format(feature),
                                     ConvBlock(in_channels=in_channels, out_channels=features[feature],
                                               kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
            else:
                self.disc.add_module("ConvBlock{}".format(feature),
                                     ConvBlock(in_channels=features[feature-1], out_channels=features[feature],
                                               kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=features[-1], out_channels=1, kernel_size=4, stride=1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.conv(self.disc(x))