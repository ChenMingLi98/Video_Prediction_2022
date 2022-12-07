import torch
import torch.nn as nn
from core.layers.TensorLayerNorm import tensor_layer_norm

class MIMBlock(nn.Module):
    def __init__(self,in_channel,num_hidden,height,width,filter_size,stride,layer_norm):
        super(MIMBlock, self).__init__()
        self.num_hidden=num_hidden
        self.layer_norm=layer_norm
        self._forget_bias = 1.0
        self.padding=filter_size//2

        self.mim_n=MIMN(in_channel,self.num_hidden,height,width,filter_size,stride,layer_norm)
        self.mim_s=MIMS(num_hidden,height,width,filter_size,stride,layer_norm)

        self.ln_x=tensor_layer_norm(self.num_hidden*6,height,width)
        self.ln_h=tensor_layer_norm(self.num_hidden*3,height,width)
        self.ln_m=tensor_layer_norm(self.num_hidden*3,height,width)

        self.conv_x=nn.Conv2d(in_channel,self.num_hidden*6,filter_size,stride,self.padding)
        self.conv_h=nn.Conv2d(self.num_hidden,self.num_hidden*3,filter_size,stride,self.padding)
        self.conv_m=nn.Conv2d(self.num_hidden,self.num_hidden*3,filter_size,stride,self.padding)
        self.conv_o_c=nn.Conv2d(self.num_hidden,self.num_hidden,filter_size,stride,self.padding)
        self.conv_o_m=nn.Conv2d(self.num_hidden,self.num_hidden,filter_size,stride,self.padding)
        self.conv_last=nn.Conv2d(self.num_hidden*2,self.num_hidden,1,1,0)

    def forward(self,x_,x,h,c,m,n,s):
        x_concat=self.conv_x(x)
        h_concat=self.conv_h(h)
        m_concat=self.conv_m(m)
        if self.layer_norm:
            x_concat=self.ln_x(x_concat)
            h_concat=self.ln_h(h_concat)
            m_concat=self.ln_m(m_concat)

        g_x,i_x,gg_x,ii_x,ff_x,o_x=torch.split(x_concat,self.num_hidden,dim=1)
        g_h,i_h,o_h=torch.split(h_concat,self.num_hidden,dim=1)
        gg_m,ii_m,ff_m=torch.split(m_concat,self.num_hidden,dim=1)

        g=torch.tanh(g_x+g_h)
        i=torch.sigmoid(i_x+i_h)

        h_diff=x-x_
        n,d=self.mim_n(h_diff,n)
        s,t=self.mim_s(d,c,s)

        c=t+i*g

        gg=torch.tanh(gg_x+gg_m)
        ii=torch.sigmoid(ii_x+ii_m)
        ff=torch.sigmoid(ff_x+ff_m+self._forget_bias)

        m=ff*m+ii*gg
        o=torch.sigmoid(o_x+o_h+self.conv_o_c(c)+self.conv_o_m(m))

        states=torch.cat([c,m],dim=1)
        h=o*torch.tanh(self.conv_last(states))

        return h,c,m,n,s

class MIMN(nn.Module):
    def __init__(self,in_channel,num_hidden,height,width,filter_size,stride,layer_norm):
        super(MIMN, self).__init__()

        self.num_hidden=num_hidden
        self.padding=filter_size//2
        self._forget_bias=1.0
        self.layer_norm=layer_norm

        self.ln_h_diff=tensor_layer_norm(self.num_hidden*4,height,width)
        self.ln_n=tensor_layer_norm(self.num_hidden*3,height,width)
        self.ln_w_no=tensor_layer_norm(self.num_hidden,height,width)

        self.conv_h_diff=nn.Conv2d(in_channel,self.num_hidden*4,filter_size,stride,self.padding)
        self.conv_n=nn.Conv2d(self.num_hidden,self.num_hidden*3,filter_size,stride,self.padding)
        self.conv_w_no=nn.Conv2d(self.num_hidden,self.num_hidden,filter_size,stride,self.padding)

    def forward(self,h_diff,n):
        h_diff_concat=self.conv_h_diff(h_diff)
        n_concat=self.conv_n(n)
        if self.layer_norm:
            h_diff_concat=self.ln_h_diff(h_diff_concat)
            n_concat=self.ln_n(n_concat)

        g_h,i_h,f_h,o_h=torch.split(h_diff_concat,self.num_hidden,dim=1)
        g_n,i_n,f_n=torch.split(n_concat,self.num_hidden,dim=1)

        g=torch.tanh(g_h+g_n)
        i=torch.sigmoid(i_h+i_n)
        f=torch.sigmoid(f_h+f_n+self._forget_bias)

        n=f*n+i*g

        o_n=self.conv_w_no(n)
        o_n=self.ln_w_no(o_n)
        o=torch.sigmoid(o_h+o_n)
        d=o*torch.tanh(n)

        return n,d

class MIMS(nn.Module):
    def __init__(self,num_hidden,height,width,filter_size,stride,layer_norm):
        super(MIMS, self).__init__()

        self.num_hidden=num_hidden
        self.padding=filter_size//2
        self._forget_bias=1.0
        self.layer_norm=layer_norm

        self.ln_d=tensor_layer_norm(self.num_hidden*4,height,width)
        self.ln_c=tensor_layer_norm(self.num_hidden*4,height,width)
        self.ln_w_so=tensor_layer_norm(self.num_hidden,height,width)

        self.conv_d=nn.Conv2d(self.num_hidden,self.num_hidden*4,filter_size,stride,self.padding)
        self.conv_c=nn.Conv2d(self.num_hidden,self.num_hidden*4,filter_size,stride,self.padding)
        self.conv_w_so=nn.Conv2d(self.num_hidden,self.num_hidden,filter_size,stride,self.padding)

    def forward(self,d,c,s):
        d_concat=self.conv_d(d)
        c_concat=self.conv_c(c)
        if self.layer_norm:
            d_concat=self.ln_d(d_concat)
            c_concat=self.ln_c(c_concat)

        g_d,i_d,f_d,o_d=torch.split(d_concat,self.num_hidden,dim=1)
        g_c,i_c,f_c,o_c=torch.split(c_concat,self.num_hidden,dim=1)

        g=torch.tanh(g_d+g_c)
        i=torch.sigmoid(i_d+i_c)
        f=torch.sigmoid(f_d+f_c+self._forget_bias)

        s=f*s+i*g

        o_s=self.conv_w_so(s)
        o_s=self.ln_w_so(o_s)
        o=torch.sigmoid(o_d+o_c+o_s)
        t=o*torch.tanh(s)

        return s,t




