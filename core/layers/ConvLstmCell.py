#ConvLstm组成单元，公式更新单元（核心）
import torch
import torch.nn as nn

class ConvLstm_Cell(nn.Module):
    def __init__(self,in_channel, num_hidden, height,width, filter_size, stride):
        super(ConvLstm_Cell, self).__init__()

        self.num_hidden=num_hidden
        self.padding=filter_size//2
        self._forget_bias=1.0

        self.conv_x=nn.Sequential(
            nn.Conv2d(in_channels=in_channel,
                      out_channels=4 * self.num_hidden,
                      kernel_size=filter_size,
                      padding=self.padding,
                      stride=stride,
                      bias=False),
            nn.BatchNorm2d(4*self.num_hidden,affine=True, track_running_stats=True)
        )

        self.conv_h=nn.Sequential(
            nn.Conv2d(in_channels=self.num_hidden,
                      out_channels=4*self.num_hidden,
                      kernel_size=filter_size,
                      padding=self.padding,
                      stride=stride,
                      bias=False),
            nn.BatchNorm2d(4*self.num_hidden,affine=True, track_running_stats=True)
        )

        #self.w_ci=nn.Parameter(torch.zeros(1,self.num_hidden,height,width),requires_grad=True)
        #self.w_cf=nn.Parameter(torch.zeros(1,self.num_hidden,height,width),requires_grad=True)
        #self.w_co=nn.Parameter(torch.zeros(1,self.num_hidden,height,width),requires_grad=True)

    def forward(self,input_tensor,h,c):

        x=input_tensor
        h_cur,c_cur=h,c
        x_concat=self.conv_x(x)
        h_concat=self.conv_h(h_cur)

        i_x,f_x,g_x,o_x=torch.split(x_concat,self.num_hidden,dim=1)
        i_h,f_h,g_h,o_h=torch.split(h_concat,self.num_hidden,dim=1)


        i=torch.sigmoid(i_x+i_h)
        f=torch.sigmoid(f_x+f_h+self._forget_bias)
        g=torch.tanh(g_x+g_h)
        c_next=f*c_cur+i*g
        o=torch.sigmoid(o_x+o_h)
        h_next=o*torch.tanh(c_next)

        return h_next,c_next


