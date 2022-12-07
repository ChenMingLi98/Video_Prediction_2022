import torch
import torch.nn as nn

device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
class self_attention_memory_module(nn.Module):  # SAM
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        # h(hidden): layer q, k, v
        # m(memory): layer k2, v2
        # layer z, m are for layer after concat(attention_h, attention_m)

        # layer_q, k, v are for h (hidden) layer
        # Layer_ k2, v2 are for m (memory) layer
        # Layer_z, m are using after concatinating attention_h and attention_m layer

        self.layer_q = nn.Conv2d(input_dim, hidden_dim, 1)
        self.layer_k = nn.Conv2d(input_dim, hidden_dim, 1)
        self.layer_k2 = nn.Conv2d(input_dim, hidden_dim, 1)
        self.layer_v = nn.Conv2d(input_dim, input_dim, 1)
        self.layer_v2 = nn.Conv2d(input_dim, input_dim, 1)
        self.layer_z = nn.Conv2d(input_dim * 2, input_dim * 2, 1)
        self.layer_m = nn.Conv2d(input_dim * 3, input_dim * 3, 1)
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim

    def forward(self, h, m):
        batch_size, channel, H, W = h.shape
        # feature aggregation
        ##### hidden h attention #####
        K_h = self.layer_k(h)
        Q_h = self.layer_q(h)
        K_h = K_h.view(batch_size, self.hidden_dim, H * W)
        Q_h = Q_h.view(batch_size, self.hidden_dim, H * W)
        Q_h = Q_h.transpose(1, 2)

        A_h = torch.softmax(torch.bmm(Q_h, K_h), dim=-1)  # batch_size, H*W, H*W

        V_h = self.layer_v(h)
        V_h = V_h.view(batch_size, self.input_dim, H * W)
        Z_h = torch.matmul(A_h, V_h.permute(0, 2, 1))

        ###### memory m attention #####
        K_m = self.layer_k2(m)
        V_m = self.layer_v2(m)
        K_m = K_m.view(batch_size, self.hidden_dim, H * W)
        V_m = V_m.view(batch_size, self.input_dim, H * W)
        A_m = torch.softmax(torch.bmm(Q_h, K_m), dim=-1)
        V_m = self.layer_v2(m)
        V_m = V_m.view(batch_size, self.input_dim, H * W)
        Z_m = torch.matmul(A_m, V_m.permute(0, 2, 1))
        Z_h = Z_h.transpose(1, 2).view(batch_size, self.input_dim, H, W)
        Z_m = Z_m.transpose(1, 2).view(batch_size, self.input_dim, H, W)

        ### Z_h & Z_m (from attention) then, concat then computation ####
        W_z = torch.cat([Z_h, Z_m], dim=1)
        Z = self.layer_z(W_z)
        ## Memory Updating (Ref: SA-ConvLSTM)
        combined = self.layer_m(torch.cat([Z, h], dim=1))  # 3 * input_dim
        mo, mg, mi = torch.split(combined, self.input_dim, dim=1)
        ### (Ref: SA-ConvLSTM)
        mi = torch.sigmoid(mi)
        new_m = (1 - mi) * m + mi * torch.tanh(mg)
        new_h = torch.sigmoid(mo) * new_m

        return new_h, new_m


class SA_Convlstm_cell(nn.Module):
    def __init__(self, in_channel, num_hidden, filter_size, stride):
        super().__init__()
        self.input_channels = in_channel
        self.num_hidden = num_hidden
        self.kernel_size = filter_size
        self.padding = filter_size//2
        self.attention_layer = self_attention_memory_module(self.num_hidden, self.num_hidden).to(device)
        self.conv2d = nn.Sequential(
            nn.Conv2d(in_channels=self.input_channels + self.num_hidden, out_channels=4 * self.num_hidden,
                      kernel_size=self.kernel_size,stride=stride, padding=self.padding)
            , nn.GroupNorm(4 * self.num_hidden, 4 * self.num_hidden))  # (num_groups, num_channels)

    def forward(self, x, h,c,m):
        h=h
        c=c
        m=m

        combined = torch.cat([x, h], dim=1)  # (batch_size, input_dim + hidden_dim, img_size[0], img_size[1])

        combined_conv = self.conv2d(combined)  # (batch_size, 4 * hidden_dim, img_size[0], img_size[1])
        i, f, o, g = torch.split(combined_conv, self.num_hidden, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        # Finish typical Convlstm above in the forward()
        # Attention below
        h_next, m_next = self.attention_layer(h_next, m)

        return h_next, c_next, m_next

    def init_hidden(self, batch_size, img_size):  # h, c, m initalize
        h, w = img_size

        return (torch.zeros(batch_size, self.hidden_dim, h, w).to(self.device),
                torch.zeros(batch_size, self.hidden_dim, h, w).to(self.device),
                torch.zeros(batch_size, self.hidden_dim, h, w).to(self.device))

