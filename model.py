from time import sleep
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import pywt
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.normalized_shape = tuple(normalized_shape)
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.normalized_shape))
            self.bias = nn.Parameter(torch.zeros(self.normalized_shape))

    def forward(self, input):
        mean = input.mean(dim=(1, 2), keepdim=True)
        variance = input.var(dim=(1, 2), unbiased=False, keepdim=True)
        input = (input - mean) / torch.sqrt(variance + self.eps)
        if self.elementwise_affine:
            input = input * self.weight + self.bias
        return input


class GLU(nn.Module):
    def __init__(self, features, dropout=0.1):
        super(GLU, self).__init__()
        self.conv1 = nn.Conv2d(features, features*2, (1, 1))
        self.conv2 = nn.Conv2d(features, features*2, (1, 1))
        self.conv3 = nn.Conv2d(features*2, features, (1, 1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x,weight):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        out = x1 * F.gelu(x2)
        out = self.dropout(out)
        out = self.conv3(out)
        return out*weight+out


class Conv(nn.Module):
    def __init__(self, features, dropout=0.1):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(features, features, (1, 1))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class TemporalEmbedding(nn.Module):
    def __init__(self, time, features):
        super(TemporalEmbedding, self).__init__()

        self.time = time
        # temporal embeddings
        self.time_day = nn.Parameter(torch.empty(time, features))
        nn.init.xavier_uniform_(self.time_day)

        self.time_week = nn.Parameter(torch.empty(7, features))
        nn.init.xavier_uniform_(self.time_week)

    def forward(self, x):
        day_emb = x[..., 1]
        time_day = self.time_day[
            (day_emb[:, -1, :] * self.time).type(torch.LongTensor)
        ]
        time_day = time_day.transpose(1, 2).unsqueeze(-1)

        week_emb = x[..., 2]
        time_week = self.time_week[
            (week_emb[:, -1, :]).type(torch.LongTensor)
        ]
        time_week = time_week.transpose(1, 2).unsqueeze(-1)

        tem_emb = time_day + time_week
        return tem_emb


class SpatialAttention(nn.Module):
    def __init__(self, device, d_model, head, num_nodes, seq_length=1, dropout=0.1):
        super(SpatialAttention, self).__init__()
        assert d_model % head == 0
        self.d_k = d_model // head
        self.head = head
        self.num_nodes = num_nodes
        self.seq_length = seq_length
        self.d_model = d_model

        self.v = Conv(d_model)
        self.concat = Conv(d_model)



    def forward(self, input, adj_list=None):
        value =  self.v(input)

        value = value.view(
            value.shape[0], -1, self.d_k, value.shape[2], self.seq_length
        ).permute(
            0, 1, 4, 3, 2
        )

        attn_dyn = torch.einsum("bnm,bhlnc->bhlnc",adj_list , value)

        x =  attn_dyn
        x = (
            x.permute(0, 1, 4, 3, 2)
            .contiguous()
            .view(x.shape[0], self.d_model, self.num_nodes, self.seq_length)
        )
        x = self.concat(x)

        return x


class Encoder(nn.Module):
    def __init__(self, device, d_model, head, num_nodes, seq_length=1, dropout=0.1):
        "Take in model size and number of heads."
        super(Encoder, self).__init__()
        assert d_model % head == 0
        self.d_k = d_model // head  # We assume d_v always equals d_k
        self.head = head
        self.num_nodes = num_nodes
        self.seq_length = seq_length
        self.d_model = d_model
        self.attention = SpatialAttention(
            device, d_model, head, num_nodes, seq_length=seq_length
        )
        self.LayerNorm = LayerNorm(
            [d_model, num_nodes, seq_length], elementwise_affine=False
        )
        self.dropout1 = nn.Dropout(p=dropout)
        self.glu = GLU(d_model)
        self.dropout2 = nn.Dropout(p=dropout)

        self.adaptive_embedding = nn.init.xavier_uniform_(
            nn.Parameter(torch.empty(d_model, num_nodes, 1))
        )
    def forward(self, input, adj_list=None):
        # 64 64 170 12
        x= self.attention(input,adj_list)
        x = x + input
        x = self.LayerNorm(x)
        x = self.dropout1(x)
        x = self.glu(x,self.adaptive_embedding) + x

        x = self.LayerNorm(x)
        x = self.dropout2(x)
        return x


class DualChannelLearner(nn.Module):
    def __init__(self, features=128, layers=4, length=12, num_nodes=170, dropout=0.1):
        super(DualChannelLearner, self).__init__()



        kernel_size = int(length / 3 + 1)
        self.high_freq_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(features, features, (1, kernel_size)),
                nn.ReLU(),
                nn.Dropout(dropout)) for _ in range(3)
        ])
        self.low_freq_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(features, features, (1, kernel_size)),
                nn.ReLU(),

                nn.Dropout(dropout)) for _ in range(3)
        ])
        self.a=nn.Conv2d(features, features, (1, 1))
        self.b = nn.Conv2d(features, features, (1, 1))
    def forward(self, XL, XH):
        XH = nn.functional.pad(XH, (1, 0, 0, 0))
        XL = nn.functional.pad(XL, (1, 0, 0, 0))
        output=XL[..., -1:]+XH[..., -1:]
        for layer in self.low_freq_layers:
            XL = layer(XL)

        for layer in self.high_freq_layers:
            XH = layer(XH)
        f = F.sigmoid(self.a(XH) + self.b(XL))
        output =  (1 - f) * XH + f * XL+output
        # output = XL + XH   #高频和低频特征相加，得到频域特征
        return output
class TATT_1(nn.Module):
    def __init__(self, c_in, num_nodes, tem_size):
        super(TATT_1, self).__init__()

        self.conv1 = nn.Conv2d(c_in, 1, kernel_size=(1, 1),
                            stride=(1, 1), bias=False)
        self.conv2 = nn.Conv2d(num_nodes, 1, kernel_size=(1, 1),
                            stride=(1, 1), bias=False)
        self.w = nn.Parameter(torch.rand(num_nodes, c_in), requires_grad=True)
        nn.init.xavier_uniform_(self.w)

        self.b = nn.Parameter(torch.zeros(tem_size, tem_size), requires_grad=True)
        self.v = nn.Parameter(torch.rand(tem_size, tem_size), requires_grad=True)
        nn.init.xavier_uniform_(self.v)
        # nn.init.xavier_uniform_(self.b)
        self.bn = nn.BatchNorm1d(tem_size)

    def forward(self, seq):

        seq = seq.transpose(3, 2)

        seq = seq.permute(0, 1, 3, 2).contiguous()
        c1 = seq.permute(0, 1, 3, 2)  # b,c,n,l->b,c,l,n
        f1 = self.conv1(c1).squeeze()  # b,l,n

        c2 = seq.permute(0, 2, 1, 3)  # b,c,n,l->b,n,c,l
        f2 = self.conv2(c2).squeeze(axis=1)  # b,c,n  [50, 1, 12]

        logits = torch.sigmoid(torch.matmul(torch.matmul(f1, self.w), f2) + self.b)
        logits = torch.matmul(self.v, logits)
        logits = logits.permute(0, 2, 1).contiguous()

        logits = self.bn(logits).permute(0, 2, 1).contiguous()

        coefs = torch.softmax(logits, -1)
        T_coef = coefs.transpose(-1, -2)

        x_1 = torch.einsum('bcnl,blq->bcnq', seq, T_coef)

        return x_1

class DC_STGNet(nn.Module):
    def __init__(
        self,
        device,
        input_dim=3,
        channels=64,
        num_nodes=170,
        input_len=12,
        output_len=12,
        dropout=0.1,
    ):
        super().__init__()

        # attributes
        self.device = device
        self.num_nodes = num_nodes
        self.node_dim = channels
        self.input_len = input_len
        self.input_dim = input_dim
        self.output_len = output_len
        self.head = 1

        if num_nodes == 170 or num_nodes == 307 or num_nodes == 358  or num_nodes == 883:
            time = 288
        elif num_nodes == 250 or num_nodes == 266:
            time = 48
        elif num_nodes>200:
            time = 96

        self.Temb = TemporalEmbedding(time, channels)


        self.network_channel = channels * 2

        self.SpatialBlock = Encoder(
            device,
            d_model=self.network_channel,
            head=self.head,
            num_nodes=num_nodes,
            seq_length=1,
            dropout=dropout,
        )

        self.fc_st = nn.Conv2d(
            self.network_channel, self.network_channel, kernel_size=(1, 1)
        )
        self.fc_st2 = nn.Conv2d(
            self.network_channel, self.network_channel, kernel_size=(1, 1)
        )

        self.regression_layer = nn.Conv2d(
            self.network_channel, self.output_len, kernel_size=(1, 1)
        )
        self.start_conv_1 = nn.Conv2d(self.input_dim, channels, kernel_size=(1, 1))
        self.start_conv_2 = nn.Conv2d(self.input_dim, channels, kernel_size=(1, 1))
        self.DCL = DualChannelLearner(
                    features = 128,
                    layers = 4,
                    length = 12,
                    num_nodes = self.num_nodes,
                    dropout=0.1
                )
        self.fc_d = nn.Conv2d(channels, 10, kernel_size=(1, 1))
        self.fc_w = nn.Conv2d(channels, 10, kernel_size=(1, 1))
        self.fc = nn.Linear(3, 1)
        self.nodevec_p1 = nn.Parameter(torch.randn(288, 40).to(device), requires_grad=True).to(device)
        self.nodevec_p2 = nn.Parameter(torch.randn(7, 40).to(device), requires_grad=True).to(device)
        self.node_embeddings = nn.Parameter(torch.randn(num_nodes, 40), requires_grad=True).to(device)
        self.nodevec_pk = nn.Parameter(torch.randn(128, 40, 40), requires_grad=True).to(device)

    def param_num(self):
        return sum([param.nelement() for param in self.parameters()])

    def forward(self, history_data):


        # 原始张量转 NumPy
        residual_numpy = history_data.cpu().detach().numpy()

        # 一层小波分解（db1）
        # 返回 [cA₁, cD₁]，分别是第一层近似系数（低频）和细节系数（高频）
        coef = pywt.wavedec(residual_numpy, 'db1', level=1)

        # 构造只保留低频或高频的系数列表
        coefl = [coef[0], None]  # 仅保留 cA₁，其它置 None，用于重构低频
        coefh = [None, coef[1]]  # 仅保留 cD₁，其它置 None，用于重构高频

        # 逆小波重构（wavelet reconstruction）
        xl = pywt.waverec(coefl, 'db1')  # 低频特征
        xh = pywt.waverec(coefh, 'db1')  # 高频特征

        xl = torch.from_numpy(xl).to(self.device)
        xh = torch.from_numpy(xh).to(self.device)

        input_data_1 = self.start_conv_1(xl)
        input_data_2 = self.start_conv_2(xh)

        input_data2 = self.DCL(input_data_1, input_data_2)


        day=history_data[:,1,0,-1]*288
        week=history_data[:,2,0,-1]
        days=self.nodevec_p1[day.cpu().numpy()]
        weeks = self.nodevec_p2[week.cpu().numpy()]
        adp = torch.einsum('ai, jik->ajk', days+weeks, self.nodevec_pk)

        adp = torch.einsum('ck, ajk->ajc', self.node_embeddings, adp)
        input_data=input_data2.squeeze()
        adj_f = torch.einsum('abc, abd->acd', input_data, adp)
        adj_f = F.relu(adj_f)
        adj_f = F.softmax(adj_f, dim=2)





        tem_emb = self.Temb(history_data.permute(0, 3, 2, 1))

        data_st = torch.cat([input_data2] + [tem_emb], dim=1)

        data_st = self.SpatialBlock(data_st,adj_f) + self.fc_st2(data_st)* torch.sigmoid(self.fc_st(data_st))

        prediction = self.regression_layer(data_st)

        return prediction
