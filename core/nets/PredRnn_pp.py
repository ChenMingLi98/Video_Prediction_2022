
import torch
import torch.nn as nn
from core.layers.GradientHighwayUnit import GHU as ghu
from core.layers.CausalLSTMCell import CausalLSTMCell as cslstm


class RNN(nn.Module):
    def __init__(self,num_hidden,num_layers,configs):
        super(RNN, self).__init__()

        self.configs=configs
        self.frame_channel = configs.patch_size * configs.patch_size * configs.img_channel
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.MSE_criterion = nn.MSELoss()

        cell_list = []
        ghu_list = []
        height = configs.img_height // configs.patch_size
        width = configs.img_width // configs.patch_size

        for i in range(self.num_layers):
            in_channel = self.frame_channel if i == 0 else num_hidden[i - 1]
            cell_list.append(cslstm(in_channel,num_hidden[i],height,width,configs.filter_size,
                                    configs.stride, configs.layer_norm))

        self.cell_list = nn.ModuleList(cell_list)
        self.conv_last = nn.Conv2d(self.num_hidden[-1],self.frame_channel,1, 1, 0,bias=False)
        ghu_list.append(ghu(self.num_hidden[0],height,width,configs.filter_size,configs.stride,configs.layer_norm))
        self.ghu_list = nn.ModuleList(ghu_list)


    def forward(self, frames_tensor, mask_true):
        # [batch, length, height, width, channel] -> [batch, length, channel, height, width]
        frames = frames_tensor.permute(0, 1, 4, 2, 3).contiguous()
        if mask_true != None:
            mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous()

        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]

        next_frames = []
        h_t = []
        c_t = []

        for i in range(self.num_layers):
            zeros = torch.zeros([batch, self.num_hidden[i], height, width]).to(self.configs.device)
            h_t.append(zeros)
            c_t.append(zeros)

        memory = torch.zeros([batch, self.num_hidden[0], height, width]).to(self.configs.device)
        z_t=torch.zeros([batch, self.num_hidden[0], height, width]).to(self.configs.device)

        for t in range(self.configs.total_length - 1):

            if t < self.configs.input_length:
                net = frames[:, t]
            elif mask_true != None:
                net = mask_true[:, t - self.configs.input_length] * frames[:, t] + \
                    (1 - mask_true[:, t - self.configs.input_length]) * x_gen

            h_t[0], c_t[0], memory = self.cell_list[0](net, h_t[0], c_t[0], memory)
            z_t = self.ghu_list[0](h_t[0], z_t)
            #print(z_t.shape)
            #print(h_t[1].shape)
            #print(c_t[1].shape)
            #print(memory.shape)
            h_t[1],c_t[1],memory=self.cell_list[1](z_t,h_t[1],c_t[1],memory)

            for i in range(2, self.num_layers):
                h_t[i], c_t[i], memory = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i], memory)

            x_gen = self.conv_last(h_t[self.num_layers - 1])
            next_frames.append(x_gen)

        # [length, batch, channel, height, width] -> [batch, length, height, width, channel]
        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 3, 4, 2).contiguous()
        loss = self.MSE_criterion(next_frames[:, -1], frames_tensor[:, -1])
        return next_frames[:,-1].unsqueeze(1), loss


