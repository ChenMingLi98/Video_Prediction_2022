from collections import deque

from core.layers.Eidetic3DLSTMCell import Eidetic3DLSTMCell
import torch
from torch import nn


# noinspection PyTypeChecker
class Eidetic3DLSTM(nn.Module):

    def __init__(self, num_hidden,num_layers,configs, window_length: int=2, kernel_size=(2,5,5)):

        super().__init__()

        self.configs = configs
        self.frame_channel = configs.patch_size * configs.patch_size * configs.img_channel
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.window_length = window_length
        cell_list = []
        self.MSE_criterion = nn.MSELoss()

        for i in range(self.num_layers):
            input_channels = self.frame_channel if i == 0 else self.num_hidden[i - 1]
            cell_list.append(
                Eidetic3DLSTMCell(in_channels=input_channels, hidden_channels=self.num_hidden[i],
                                  depth=self.window_length, kernel_size=kernel_size)
            )
        self.cell_list = nn.ModuleList(cell_list)

        self.conv_last = nn.Conv3d(in_channels=self.num_hidden[-1], out_channels=self.frame_channel,
                                   kernel_size=(self.window_length, 1, 1), stride=1, padding=0)

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
        next_frames = []  # 存储预测结果

        # 初始化最开始的隐藏状态
        for i in range(self.num_layers):
            zero_state_h = torch.zeros(batch, self.num_hidden[i],
                                       self.window_length, height, width).to(self.configs.device)
            zero_state_c = torch.zeros(batch, self.num_hidden[i],
                                       self.window_length, height, width).to(self.configs.device)

            c_t.append([zero_state_c])
            h_t.append(zero_state_h)

        m = torch.zeros(batch, self.num_hidden[0], self.window_length, height, width).to(self.configs.device)

        input_queue = deque(maxlen=self.window_length)

        for time_step in range(self.window_length - 1):
            input_queue.append(
                torch.zeros(batch, self.frame_channel, height, width).to(self.configs.device)
            )

        for time_step in range(self.configs.total_length - 1):
            if time_step < self.configs.input_length:
                x = frames[:, time_step]
            elif mask_true != None:
                x = mask_true[:, time_step - self.configs.input_length] * frames[:, time_step] + \
                    (1 - mask_true[:, time_step - self.configs.input_length]) * x_gen

            input_queue.append(x)

            x = torch.stack(tuple(input_queue))
            x = x.permute(1, 2, 0, 3, 4)

            for i in range(self.num_layers):
                if i == 0:
                    inputs = x
                else:
                    inputs = h_t[i - 1]

                h_t[i],c,m = self.cell_list[i](inputs, h_t[i], c_t[i], m)
                c_t[i].append(c)

            x_gen = self.conv_last(h_t[-1]).squeeze(dim=2)  # [batch, channel, height, width]
            next_frames.append(x_gen)

        # [length, batch, channel, height, width] -> [batch, length, height, width, channel]
        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 3, 4, 2).contiguous()
        loss = self.MSE_criterion(next_frames[:, -1], frames_tensor[:, -1])
        return next_frames[:,-1].unsqueeze(1), loss


