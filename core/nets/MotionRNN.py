from core.layers.MotionGRU import MotionGRU
from core.layers.SpatioTemporalLSTMCell import SpatioTemporalLSTMCell
import torch
from torch import nn


# 由于 M 的存在之字形贯穿所有 cell，要保证 M 的 channel 不变，只能所有 layer 隐藏层通道数相同
class MotionRNN(nn.Module):
    def __init__(self, num_hidden,num_layers,configs,forget_bias: float = 0.01, k: int = 3, alpha=0.5):
        super(MotionRNN, self).__init__()

        self.configs=configs
        self.frame_channel = configs.patch_size * configs.patch_size * configs.img_channel
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        cell_list = []
        motion_gru=[]

        self.forget_bias = forget_bias
        self.k = k
        height = configs.img_height // configs.patch_size
        width = configs.img_width // configs.patch_size
        self.MSE_criterion = nn.MSELoss()

        for i in range(self.num_layers):
            input_channels = self.frame_channel if i == 0 else self.num_hidden[i - 1]
            cell_list.append(
                SpatioTemporalLSTMCell(in_channel=input_channels, num_hidden=self.num_hidden[i],height=height,width=width,
                                   filter_size=configs.filter_size, stride=configs.stride,layer_norm=configs.layer_norm)
            )
            if i < self.num_layers - 1:
                motion_gru.append(
                    MotionGRU(hidden_channels=self.num_hidden[i], k=self.k, alpha=alpha)
                )

        self.cell_list = nn.ModuleList(cell_list)
        self.motion_gru = nn.ModuleList(motion_gru)

        # 最后输出的通道数和输入通道数一样
        self.conv_last = nn.Conv2d(in_channels=self.num_hidden[-1], out_channels=self.frame_channel,
                                   kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=False)

    # noinspection PyUnboundLocalVariable
    def forward(self, frames_tensor, mask_true):
        # [batch, length, height, width, channel] -> [batch, length, channel, height, width]
        frames = frames_tensor.permute(0, 1, 4, 2, 3).contiguous()
        if mask_true != None:
            mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous()

        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]

        h_t = []  # 存储隐藏层
        c_t = []  # 存储cell记忆
        o_t = []  # 输出门
        d = [None]
        f = [None]
        next_frames = []  # 存储预测结果

        # 初始化最开始的隐藏状态
        for i in range(self.num_layers):
            zero_tensor = torch.zeros(batch, self.num_hidden[i], height, width).to(self.configs.device)
            h_t.append(zero_tensor)
            c_t.append(zero_tensor)
            o_t.append(zero_tensor)

        for j in range(1, self.num_layers):
            zero_tensor_d = torch.zeros(batch, 2 * self.k ** 2, height // 2, width // 2).to(self.configs.device)
            zero_tensor_f = torch.zeros(batch, 2 * self.k ** 2, height // 2, width // 2).to(self.configs.device)
            d.append(zero_tensor_d)
            f.append(zero_tensor_f)

        memory = torch.zeros(batch, self.num_hidden[0], height, width).to(self.configs.device)

        # 开始循环，模型在预测部分的输入是前一帧的预测输出
        for t in range(self.configs.total_length - 1):

            if t < self.configs.input_length:
                net = frames[:, t]
            elif mask_true!=None:
                net = mask_true[:, t - self.configs.input_length] * frames[:, t] + \
                    (1 - mask_true[:, t - self.configs.input_length]) * x_gen

            h_t[0], c_t[0], memory,o_t[0] = self.cell_list[0](net, h_t[0], c_t[0], memory)

            for i in range(1, self.num_layers):
                xx,f[i],d[i]=self.motion_gru[i - 1](h_t[i - 1], f[i], d[i])
                h_t[i], c_t[i], memory,o_t[i] = self.cell_list[i](xx, h_t[i], c_t[i], memory)
                h_t[i] = h_t[i] + (1 - o_t[i]) * h_t[i - 1]

            x_gen = self.conv_last(h_t[self.num_layers - 1])
            next_frames.append(x_gen)

        # [length, batch, channel, height, width] -> [batch, length, height, width, channel]
        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 3, 4, 2).contiguous()
        loss = self.MSE_criterion(next_frames[:, -1], frames_tensor[:, -1])
        return next_frames[:,-1].unsqueeze(1), loss

