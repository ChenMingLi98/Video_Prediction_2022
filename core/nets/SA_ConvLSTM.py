#构建ConvLstm网络架构
import torch
import torch.nn as nn
from core.layers.SA_ConvLSTMCell import SA_Convlstm_cell

class SAConvLstm(nn.Module):
    def __init__(self,num_hidden,num_layers,configs):
        super(SAConvLstm, self).__init__()

        self.configs=configs
        self.frame_channel = configs.patch_size * configs.patch_size * configs.img_channel
        self.num_hidden=num_hidden
        self.num_layers=num_layers
        cell_list = []
        self.MSE_criterion=nn.MSELoss()


        for layer_idx in range(self.num_layers):
            cur_input_tensor=self.frame_channel if layer_idx==0 else self.num_hidden[layer_idx-1]
            cell_list.append(SA_Convlstm_cell(cur_input_tensor,self.num_hidden[layer_idx],
                                           configs.filter_size,configs.stride))

        self.cell_list=nn.ModuleList(cell_list)
        self.conv_last = nn.Conv2d(self.num_hidden[num_layers-1],self.frame_channel,
                                   kernel_size=1,stride=1,padding=0,bias=False)

    def forward(self,frames_tensor,mask_true):
        #input_tensor为五维参数[B,S,C,H,W]
        #mask_true为概率值大小和input_tensor一样
        frames=frames_tensor.permute(0,1,4,2,3).contiguous()
        if mask_true != None:
            mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous()

        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]

        next_frames = []
        h_t = []
        c_t = []
        m_t=[]

        for i in range(self.num_layers):
            zeros = torch.zeros([batch, self.num_hidden[i], height, width]).to(self.configs.device)
            h_t.append(zeros)
            c_t.append(zeros)
            m_t.append(zeros)

        for t in range(self.configs.total_length - 1):
            if t<self.configs.input_length:
                input=frames[:,t]
            elif mask_true != None:
                input=mask_true[:,t - self.configs.input_length]*frames[:,t]+\
                      (1-mask_true[:,t - self.configs.input_length])*x_gen

            h_t[0],c_t[0],m_t[0]=self.cell_list[0](input,h_t[0],c_t[0],m_t[0])

            for i in range(1,self.num_layers):
                h_t[i],c_t[i],m_t[i]=self.cell_list[i](h_t[i-1],h_t[i],c_t[i],m_t[i])

            x_gen=self.conv_last(h_t[self.num_layers-1])
            next_frames.append(x_gen)

        # [length, batch, channel, height, width] -> [batch, length, height, width, channel]
        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 3, 4, 2).contiguous()
        loss = self.MSE_criterion(next_frames[:,-1], frames_tensor[:, -1])
        return next_frames[:,-1].unsqueeze(1), loss






