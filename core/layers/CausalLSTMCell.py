import torch
import torch.nn as nn
from core.layers.TensorLayerNorm import tensor_layer_norm

class CausalLSTMCell(nn.Module):
    def __init__(self,in_channel,num_hidden,height,width,filter_size,stride,layer_norm):
        super(CausalLSTMCell, self).__init__()

        self.num_hidden=num_hidden
        self.padding=filter_size//2
        self.layer_norm = layer_norm
        self._forget_bias =1.0
        self.filter_size=filter_size
        self.stride=stride

        self.bn_h_cc = tensor_layer_norm(self.num_hidden * 4,height,width)
        self.bn_c_cc = tensor_layer_norm(self.num_hidden * 3,height,width)
        self.bn_m_cc = tensor_layer_norm(self.num_hidden * 3,height,width)
        self.bn_x_cc = tensor_layer_norm(self.num_hidden * 7,height,width)
        self.bn_c2m = tensor_layer_norm(self.num_hidden * 4,height,width)
        self.bn_o_m = tensor_layer_norm(self.num_hidden,height,width)

        self.h_cc_conv = nn.Conv2d(self.num_hidden,self.num_hidden*4,filter_size,stride,self.padding)
        self.c_cc_conv = nn.Conv2d(self.num_hidden,self.num_hidden*3,filter_size,stride,self.padding)

        self.x_cc_conv = nn.Conv2d(in_channel,self.num_hidden*7,filter_size,stride,self.padding)
        self.c2m_conv  = nn.Conv2d(self.num_hidden,self.num_hidden*4,filter_size,stride,self.padding)
        self.o_m_conv = nn.Conv2d(self.num_hidden,self.num_hidden,filter_size,stride,self.padding)
        self.o_conv = nn.Conv2d(self.num_hidden, self.num_hidden,filter_size,stride,self.padding)
        self.cell_conv = nn.Conv2d(self.num_hidden*2,self.num_hidden,1,1,0)


    def forward(self,x,h,c,m):
        self.m_cc_conv = nn.Conv2d(m.shape[1], self.num_hidden * 3, self.filter_size,self.stride, self.padding).cuda()
        h_cc = self.h_cc_conv(h)
        c_cc = self.c_cc_conv(c)
        m_cc = self.m_cc_conv(m)
        x_cc = self.x_cc_conv(x)
        if self.layer_norm:
            h_cc = self.bn_h_cc(h_cc)
            c_cc = self.bn_c_cc(c_cc)
            m_cc = self.bn_m_cc(m_cc)
            x_cc = self.bn_x_cc(x_cc)

        i_h, g_h, f_h, o_h = torch.split(h_cc,self.num_hidden, 1)
        i_c, g_c, f_c = torch.split(c_cc,self.num_hidden, 1)
        i_m, f_m, m_m = torch.split(m_cc,self.num_hidden, 1)
        i_x, g_x, f_x, o_x, i_x_, g_x_, f_x_ = torch.split(x_cc,self.num_hidden, 1)

        i = torch.sigmoid(i_x + i_h+ i_c)
        f = torch.sigmoid(f_x + f_h + f_c + self._forget_bias)
        g = torch.tanh(g_x + g_h + g_c)
        c_new = f * c + i * g
        c2m = self.c2m_conv(c_new)
        if self.layer_norm:
            c2m = self.bn_c2m(c2m)

        i_c, g_c, f_c, o_c = torch.split(c2m,self.num_hidden, 1)


        ii = torch.sigmoid(i_c + i_x_ + i_m)
        ff = torch.sigmoid(f_c + f_x_ + f_m + self._forget_bias)
        gg = torch.tanh(g_c + g_x_)
        m_new = ff * torch.tanh(m_m) + ii * gg
        o_m = self.o_m_conv(m_new)
        if self.layer_norm:
             o_m = self.bn_o_m(o_m)

        o = torch.tanh(o_x + o_c + o_m)
        o = self.o_conv(o)
        #此时c_new以及m_new的格式均为[b,c,h,w]
        cell = torch.cat([c_new, m_new],1)
        cell = self.cell_conv(cell)
        h_new = o * torch.tanh(cell)
        return h_new, c_new, m_new

