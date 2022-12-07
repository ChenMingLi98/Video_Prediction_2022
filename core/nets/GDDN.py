import torch
import torch.nn as nn
from core.layers.SpatioTemporalLSTMCell import SpatioTemporalLSTMCell

class Predictor(nn.Module):
    def __init__(self,configs):
        super(Predictor, self).__init__()

        self.device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        #Frame Spatial Encoder
        self.encoder_conv1=nn.Sequential(
            nn.Conv2d(in_channels=configs.img_channel, out_channels=16, kernel_size=(5, 5), stride=2, padding=2),
            nn.ELU()
        )
        self.encoder_conv2=nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=(5, 5), stride=1, padding=2),
            nn.ELU()
        )
        self.encoder_conv3=nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5, 5), stride=2, padding=2),
            nn.ELU()
        )
        self.encoder_conv4=nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5, 5), stride=1, padding=2),
            nn.ELU()
        )
        self.encoder_conv5 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(5, 5), stride=2, padding=2),
            nn.ELU()
        )
        self.encoder_conv6 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(5, 5), stride=1, padding=2),
            nn.ELU()
        )
        # DAM Module
        self.DAM_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=configs.img_channel, out_channels=16, kernel_size=(5, 5), stride=2, padding=2),
            nn.Sigmoid()
        )
        self.DAM_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=(5, 5), stride=1, padding=2),
            nn.Sigmoid()
        )
        self.DAM_conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5, 5), stride=2, padding=2),
            nn.Sigmoid()
        )
        self.DAM_conv4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5, 5), stride=1, padding=2),
            nn.Sigmoid()
        )
        self.DAM_conv5 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(5, 5), stride=2, padding=2),
            nn.Sigmoid()
        )

        #Frame Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(5, 5), stride=1, padding=2,
                               output_padding=0),
            nn.ELU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=(5, 5), stride=2, padding=2,
                               output_padding=1),
            nn.ELU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(5, 5), stride=1, padding=2,
                               output_padding=0),
            nn.ELU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=(5, 5), stride=2, padding=2,
                               output_padding=1),
            nn.ELU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=16, kernel_size=(5, 5), stride=1, padding=2,
                               output_padding=0),
            nn.ELU(),
            nn.ConvTranspose2d(in_channels=16, out_channels=configs.img_channel, kernel_size=(5, 5), stride=2,
                               padding=2, output_padding=1)
        )    #[1,64,64]


        self.predrnn_layers_num = 4
        self.predrnn_hidden_num = [128, 128, 128, 128]
        self.predrnn_list = []
        for layer_idx in range(self.predrnn_layers_num):
            self.predrnn_list.append(SpatioTemporalLSTMCell(in_channel=self.predrnn_hidden_num[layer_idx],
                                                            num_hidden=self.predrnn_hidden_num[layer_idx],
                                                            filter_size=5,
                                                            height=configs.img_height // 8,
                                                            width=configs.img_width // 8,
                                                            stride=configs.stride,
                                                            layer_norm=1))
        self.predrnn_list = nn.ModuleList(self.predrnn_list)
        self.MSE_criterion = nn.MSELoss()


        self.attention_size = 128
        self.attention_func = nn.Sequential(
            nn.AdaptiveAvgPool2d([1, 1]),
            nn.Flatten(),
            nn.Linear(256, 16),
            nn.ReLU(),
            nn.Linear(16, self.attention_size),
            nn.Sigmoid())

    def forward(self,total_x,out_len,motion_x,motion_feature,motion_frame):
        #input->[N,S,C,H,W]   movingmnist->[N,10,1,64,64]
        # [batch, length, height, width, channel] -> [batch, length, channel, height, width]
        frames=total_x.permute(0, 1, 4, 2, 3).contiguous()
        motion_x=motion_x.permute(0, 1, 4, 2, 3).contiguous()
        batch_size=frames.size()[0]
        input_length=frames.size()[1]-out_len        #input_length=10
        short_x=frames[:,:input_length]              #input_size[8,4,3,128,128]
        total_length=frames.size()[1]                #total_length=20
        height=frames.size()[3]
        width=frames.size()[4]


        # motion context-aware video prediction
        h, c, out_pred = [], [], []
        for layer_i in range(self.predrnn_layers_num):
            zero_state = torch.zeros(batch_size, self.predrnn_hidden_num[layer_i], height//8, width//8).to(self.device)
            h.append(zero_state)
            c.append(zero_state)
        memory = torch.zeros([batch_size, self.predrnn_hidden_num[0], height//8, width//8]).to(self.device)

        for seq_i in range(total_length-1):
            if seq_i<input_length:
                input_x=short_x[:,seq_i,:,:,:]
                if seq_i==input_length-1:
                    difference_x=motion_frame[:,seq_i-input_length+1]
                else:
                    difference_x = motion_x[:, seq_i, :, :, :]
                out_frame_encoder_conv1=self.encoder_conv1(input_x)
                out_dam_1=self.DAM_conv1(difference_x)
                out_frame_encoder_conv2=self.encoder_conv2(torch.mul(out_frame_encoder_conv1,out_dam_1))
                out_dam_2=self.DAM_conv2(out_dam_1)
                out_frame_encoder_conv3=self.encoder_conv3(torch.mul(out_frame_encoder_conv2,out_dam_2))
                out_dam_3=self.DAM_conv3(out_dam_2)
                out_frame_encoder_conv4 = self.encoder_conv4(torch.mul(out_frame_encoder_conv3, out_dam_3))
                out_dam_4 = self.DAM_conv4(out_dam_3)
                out_frame_encoder_conv5 = self.encoder_conv5(torch.mul(out_frame_encoder_conv4, out_dam_4))
                out_dam_5 = self.DAM_conv5(out_dam_4)
                input_encoder=self.encoder_conv6(torch.mul(out_frame_encoder_conv5,out_dam_5))
            else:
                input_x=out_pred[-1]
                difference_x=motion_frame[:,seq_i-input_length+1]     #restruct origin size[1,64,64]
                out_frame_encoder_conv1 = self.encoder_conv1(input_x)
                out_dam_1 = self.DAM_conv1(difference_x)
                out_frame_encoder_conv2 = self.encoder_conv2(torch.mul(out_frame_encoder_conv1, out_dam_1))
                out_dam_2 = self.DAM_conv2(out_dam_1)
                out_frame_encoder_conv3 = self.encoder_conv3(torch.mul(out_frame_encoder_conv2, out_dam_2))
                out_dam_3 = self.DAM_conv3(out_dam_2)
                out_frame_encoder_conv4 = self.encoder_conv4(torch.mul(out_frame_encoder_conv3, out_dam_3))
                out_dam_4 = self.DAM_conv4(out_dam_3)
                out_frame_encoder_conv5 = self.encoder_conv5(torch.mul(out_frame_encoder_conv4, out_dam_4))
                out_dam_5 = self.DAM_conv5(out_dam_4)
                input_encoder = self.encoder_conv6(torch.mul(out_frame_encoder_conv5, out_dam_5))
            #print(input_encoder.size())
            h[0], c[0],memory = self.predrnn_list[0](input_encoder, h[0], c[0],memory)

            for i in range(1, self.predrnn_layers_num):
                h[i], c[i],memory = self.predrnn_list[i](h[i - 1], h[i], c[i],memory)

            if seq_i>=input_length-1:
                attention = self.attention_func(torch.cat([c[-1], motion_feature[:,seq_i-input_length+1]], dim=1))
                attention = torch.reshape(attention, (-1, self.attention_size, 1, 1))
                motion_feature_att = motion_feature[:,seq_i-input_length+1] * attention
                out_pred.append(self.decoder(torch.cat([h[-1], motion_feature_att], dim=1)))

        out_pred = torch.stack(out_pred)
        # [length, batch, channel, height, width] -> [batch, length, height, width, channel]
        next_frames = out_pred.permute(1, 0, 3, 4, 2).contiguous()
        loss=self.MSE_criterion(next_frames,total_x[:,out_len:])
        return next_frames,loss

class Motion_Context(nn.Module):
    def __init__(self,configs):
        super(Motion_Context, self).__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        #Difference Spatial Encoder
        self.difference_encoder=nn.Sequential(
            nn.Conv2d(in_channels=configs.img_channel, out_channels=16, kernel_size=(5, 5), stride=2, padding=2),
            nn.ELU(),
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=(5, 5), stride=1, padding=2),
            nn.ELU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5, 5), stride=2, padding=2),
            nn.ELU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5, 5), stride=1, padding=2),
            nn.ELU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(5, 5), stride=2, padding=2),
            nn.ELU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(5, 5), stride=1, padding=2),
            nn.ELU(),
        )
        self.difference_decoder=nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=(5, 5), stride=1, padding=2,output_padding=0),
            nn.ELU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=(5, 5), stride=2, padding=2,output_padding=1),
            nn.ELU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(5, 5), stride=1, padding=2,output_padding=0),
            nn.ELU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=(5, 5), stride=2, padding=2,output_padding=1),
            nn.ELU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=16, kernel_size=(5, 5), stride=1, padding=2,output_padding=0),
            nn.ELU(),
            nn.ConvTranspose2d(in_channels=16, out_channels=configs.img_channel, kernel_size=(5, 5), stride=2, padding=2,output_padding=1)
        )
        self.predrnn_layers_num=4
        self.predrnn_hidden_num=[128,128,128,128]
        self.predrnn_list=[]
        for layer_idx in range(self.predrnn_layers_num):
            self.predrnn_list.append(SpatioTemporalLSTMCell(in_channel=self.predrnn_hidden_num[layer_idx],
                                                            num_hidden=self.predrnn_hidden_num[layer_idx],
                                                            filter_size=5,
                                                            height=configs.img_height//8,
                                                            width=configs.img_width//8,
                                                            stride=configs.stride,
                                                            layer_norm=1))
        self.predrnn_list=nn.ModuleList(self.predrnn_list)
        self.MSE_criterion = nn.MSELoss()


    def forward(self,memory_x,out_length,phase,mask_true):
        memory_x=memory_x.permute(0, 1, 4, 2, 3).contiguous()
        total_length=memory_x.size()[1]      #total_frame=20 total_difference_frame=19
        input_length=memory_x.size()[1]-out_length
        batch=memory_x.shape[0]
        height=memory_x.shape[3]
        width=memory_x.shape[4]
        next_difference_frames = []
        next_difference_features=[]
        h_t = []
        c_t = []

        for i in range(self.predrnn_layers_num):
            zeros = torch.zeros([batch, self.predrnn_hidden_num[i], height//8, width//8]).to(self.device)
            h_t.append(zeros)
            c_t.append(zeros)
        memory = torch.zeros([batch, self.predrnn_hidden_num[0], height//8, width//8]).to(self.device)

        for t in range(total_length-1):
            if t<input_length:
                input_x=memory_x[:,t,:,:,:]
                input_x=self.difference_encoder(input_x)
            else:
                if phase=='train':
                    input_x = mask_true[:, t - input_length] * self.difference_encoder(memory_x[:, t]) + \
                              (1 - mask_true[:, t - input_length]) * self.difference_encoder(d_gen)
                else:
                    input_x=self.difference_encoder(d_gen)


            h_t[0], c_t[0],memory = self.predrnn_list[0](input_x, h_t[0], c_t[0],memory)

            for i in range(1, self.predrnn_layers_num):
                h_t[i], c_t[i],memory = self.predrnn_list[i](h_t[i - 1], h_t[i], c_t[i],memory)

            d_gen = self.difference_decoder(h_t[self.predrnn_layers_num - 1])           #[16,1,64,64]

            if t>=input_length-1:
                next_difference_features.append(h_t[self.predrnn_layers_num - 1])       #[16,128,16,16]
                next_difference_frames.append(d_gen)


        # [length, batch, channel, height, width] -> [batch, length, channel, height, width]
        next_difference_frames = torch.stack(next_difference_frames, dim=0).permute(1, 0, 2, 3, 4).contiguous()          #[16,10,1,64,64]
        next_difference_features = torch.stack(next_difference_features, dim=0).permute(1, 0, 2, 3, 4).contiguous()      #[16,10,128,16,16]
        loss = self.MSE_criterion(next_difference_frames, memory_x[:, out_length-1:])
        return (next_difference_features,next_difference_frames),loss





