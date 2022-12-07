import torch
import torch.nn as nn
import numpy as np

from core.nets.Support import warp_flow
import core.nets.Modules as Modules


class DualMotionGAN(nn.Module):
    def __init__(self, in_channels=3, frame_size=(256, 256), device="cpu",
                 me_cnn_features=(64, 128, 256, 512), me_lstm_features=(512, 512, 512),
                 gframe_features=(512, 256, 128, 64), gflow_features=(512, 256, 128, 64)):
        super(DualMotionGAN, self).__init__()

        self.device = device
        self.MotionEncoderCNN = Modules.ProbMotionEncoderCNN(in_channels=in_channels, features=me_cnn_features,
                                                             kernel_size=4, stride=2, padding=1,
                                                             frame_size=frame_size, bias=False).to(device)

        self.MotionEncoderLSTM = Modules.ProbMotionEncoderLSTM(in_channels=me_cnn_features[-1],
                                                               features=me_lstm_features, kernel_size=3, padding=1,
                                                               frame_size=(frame_size[0]//(2**(len(me_cnn_features))),
                                                                           frame_size[1]//(2**(len(me_cnn_features))))
                                                               ).to(device)

        self.FrameGenerator = Modules.Generator(in_channels=me_lstm_features[-1], out_channels=in_channels,
                                                features=gframe_features, kernel_size=3, stride=2, padding=1).to(device)

        self.FlowGenerator = Modules.Generator(in_channels=me_lstm_features[-1], out_channels=2,
                                               features=gflow_features, kernel_size=3, stride=2, padding=1).to(device)

        self.FusingLayer = nn.Conv2d(in_channels=2*in_channels, out_channels=in_channels, kernel_size=1, stride=1,
                                     padding=0)

    def forward(self, x):
        out_me = self.MotionEncoderLSTM(self.MotionEncoderCNN(x))
        frame_prediction = self.FrameGenerator(out_me, activation="sigmoid")
        flow_prediction = self.FlowGenerator(out_me, activation="tanh")

        flow2frame = torch.zeros((x.shape[0], x.shape[1], x.shape[3], x.shape[4]))
        complex = torch.zeros((x.shape[0], x.shape[1]*2, x.shape[3], x.shape[4])).to(self.device)
        for i in range(x.shape[0]):
            prev = torch.squeeze(x[:, :, -1][i]).cpu().detach().numpy()
            flow = np.transpose(flow_prediction[i].cpu().detach().numpy(), (1, 2, 0))
            flow2frame[i] = torch.from_numpy(warp_flow(prev, flow))
            complex[i] = torch.cat([frame_prediction[i], flow2frame[i].to(self.device)])

        prediction = nn.Sigmoid()(self.FusingLayer(complex))
        return frame_prediction, flow_prediction, prediction