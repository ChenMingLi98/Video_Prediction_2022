from core.layers.TensorLayerNorm import tensor_layer_norm
import torch
import torch.nn as nn

class GHU(nn.Module):
    def __init__(self,num_hidden,height,width,filter_size,stride,layer_norm):
        super(GHU,self).__init__()
        """Initialize the Gradient Highway Unit.
        """

        self.padding=filter_size//2
        self.num_features = num_hidden
        self.layer_norm = layer_norm

        self.bn_z_concat = tensor_layer_norm(self.num_features*2,height,width)
        self.bn_x_concat = tensor_layer_norm(self.num_features*2,height,width)

        self.z_concat_conv = nn.Conv2d(self.num_features,self.num_features*2,filter_size,stride,self.padding)
        self.x_concat_conv = nn.Conv2d(self.num_features,self.num_features*2,filter_size,stride,self.padding)


    def forward(self,x,z):
        z_concat = self.z_concat_conv(z)
        x_concat = self.x_concat_conv(x)
        if self.layer_norm:
            z_concat = self.bn_z_concat(z_concat)
            x_concat = self.bn_x_concat(x_concat)

        gates = torch.add(x_concat, z_concat)
        p, u = torch.split(gates, self.num_features, 1)
        p = torch.tanh(p)
        u = torch.sigmoid(u)
        z_new = u * p + (1 - u) * z
        return z_new


