import torch
import torch.nn as nn
from core.layers.MIMBlock import MIMBlock
from core.layers.SpatioTemporalLSTMCell import SpatioTemporalLSTMCell

class RNN(nn.Module):
    def __init__(self,num_hidden,num_layers,configs):
        super(RNN, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=configs.img_channel, out_channels=64, kernel_size=(5, 5), stride=2, padding=2),
            nn.ELU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5, 5), stride=1, padding=2),
            nn.ELU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5, 5), stride=2, padding=2),
            nn.ELU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(5, 5), stride=1, padding=2),
            nn.ELU())
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=(5, 5), stride=1, padding=2,
                               output_padding=0),
            nn.ELU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(5, 5), stride=2, padding=2,
                               output_padding=1),
            nn.ELU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=(5, 5), stride=1, padding=2,
                               output_padding=0),
            nn.ELU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=configs.img_channel, kernel_size=(5, 5), stride=2, padding=2,
                               output_padding=1))

        self.configs=configs
        self.frame_channel=configs.patch_size * configs.patch_size * configs.img_channel
        self.num_layers=num_layers
        self.num_hidden=num_hidden

        cell_list=[]
        height=configs.img_height//configs.patch_size
        width=configs.img_width//configs.patch_size
        self.MSE_criterion=nn.MSELoss()

        for i in range(self.num_layers):
            in_channel = self.frame_channel if i == 0 else num_hidden[i - 1]
            if i==0:
                cell_list.append(
                    SpatioTemporalLSTMCell(in_channel, self.num_hidden[i], height, width, configs.filter_size,
                                           configs.stride, configs.layer_norm)
                )
            else:
                cell_list.append(
                    MIMBlock(in_channel,self.num_hidden[i],height,width,configs.filter_size,
                             configs.stride,configs.layer_norm)
                )

        self.cell_list=nn.ModuleList(cell_list)
        self.conv_last = nn.Conv2d(self.num_hidden[num_layers - 1], self.frame_channel,
                                   kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self,frames_tensor,mask_true):
        # [batch, length, height, width, channel] -> [batch, length, channel, height, width]
        frames = frames_tensor.permute(0, 1, 4, 2, 3).contiguous()
        if mask_true != None:
            mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous()

        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]

        next_frames=[]
        h_t=[]
        _h_t=[]
        c_t=[]
        n=[]
        s=[]

        for i in range(self.num_layers):
            zeros = torch.zeros([batch, self.num_hidden[i], height, width]).to(self.configs.device)
            _h_t.append(zeros)
            h_t.append(zeros)
            c_t.append(zeros)
            if i>0:
                n.append(zeros)
                s.append(zeros)
            else:
                n.append(None)
                s.append(None)

        memory=torch.zeros([batch, self.num_hidden[0], height, width]).to(self.configs.device)

        for t in range(self.configs.total_length - 1):
            if t < self.configs.input_length:
                #net = self.encoder(frames[:, t])
                net = frames[:, t]
            elif mask_true != None:
                #net = mask_true[:, t - self.configs.input_length] * self.encoder(frames[:, t]) + \
                #    (1 - mask_true[:, t - self.configs.input_length]) * self.encoder(x_gen)
                net = mask_true[:, t - self.configs.input_length] * self.encoder(frames[:, t]) + \
                     (1 - mask_true[:, t - self.configs.input_length]) * x_gen

            h_t[0], c_t[0], memory = self.cell_list[0](net, h_t[0], c_t[0], memory)
            _h_t[0]=h_t[0]

            for i in range(1, self.num_layers):
                h_t[i], c_t[i], memory,n[i],s[i] = self.cell_list[i](_h_t[i-1],h_t[i - 1], h_t[i], c_t[i], memory,n[i],s[i])
                _h_t[i]=h_t[i]

            #x_gen = self.decoder(h_t[self.num_layers - 1])
            x_gen = self.conv_last(h_t[self.num_layers - 1])
            next_frames.append(x_gen)

        # [length, batch, channel, height, width] -> [batch, length, height, width, channel]
        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 3, 4, 2).contiguous()
        loss = self.MSE_criterion(next_frames[:, -1], frames_tensor[:, -1])
        return next_frames[:,-1].unsqueeze(1), loss




