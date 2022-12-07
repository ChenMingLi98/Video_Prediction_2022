#coding=utf-8
import torch.nn as nn

def tensor_layer_norm(num_features,height,width):
	return nn.LayerNorm([num_features,height,width])
