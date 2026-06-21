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


class TConv(nn.Module):
    def __init__(self, features=128, layer=4, length=12, dropout=0.1):
        super(TConv, self).__init__()
        layers = []
        kernel_size = int(length / layer + 1)
        # for i in range(3):
        #     self.conv1 = nn.Conv2d(features, features, (1, 2))
        #     self.relu1 = nn.ReLU()
        #     self.dropout1 = nn.Dropout(dropout)
        #     self.conv = nn.Conv2d(features, features, (1, kernel_size))
        #     self.relu = nn.ReLU()
        #     self.dropout = nn.Dropout(dropout)
        #     layers += [nn.Sequential(self.conv1, self.relu1, self.dropout1,self.conv, self.relu, self.dropout)]
        # self.tcn = nn.Sequential(*layers)

        # self.conv = nn.Conv2d(features, features, (1, 2))
        # self.relu = nn.ReLU()

        self.relu2 = nn.ReLU()
        #构建空间部分的两个记忆点，分别是第1步memory和第12步memory2
        self.memory = nn.Parameter(torch.randn(features, 170))
        nn.init.xavier_uniform_(self.memory)
        self.memory2 = nn.Parameter(torch.randn(features, 170))
        nn.init.xavier_uniform_(self.memory)
    def forward(self, x):
        adj_dyn_1 =torch.einsum("bcnt, cm->bnm", x[..., :1], self.memory).contiguous()


        adj_dyn_2 =  torch.einsum("bcnt, cm->bnm", x[..., -1:], self.memory2).contiguous()



        # adj = torch.cat((x[..., :1], x[..., -1:]),dim=-1).permute(0,3,2,1)
        # adj=adj@adj.transpose(3,2)
        adj=adj_dyn_2-adj_dyn_1/ math.sqrt(x.shape[1])  #计算两个记忆点之前的差值，从而得到流量趋势
        # adj=torch.softmax(self.relu2(adj),dim=-1)
        # x = nn.functional.pad(x, (1, 0, 0, 0))


        # x = self.tcn(x)+self.relu(self.conv(x[...,-2:]))+x[...,-1:]

        # adj1=x.squeeze()
        # adj1 = adj1.transpose(2, 1) @ adj1
        adj=torch.softmax(self.relu2(adj),dim=-1)

        return x[...,-1:],adj    #返回时序特征的第12步和12步趋势的softmax值


class SpatialAttention(nn.Module):
    def __init__(self, device, d_model, head, num_nodes, seq_length=1, dropout=0.1):
        super(SpatialAttention, self).__init__()
        assert d_model % head == 0
        self.d_k = d_model // head
        self.head = head
        self.num_nodes = num_nodes
        self.seq_length = seq_length
        self.d_model = d_model
        # self.q = Conv(d_model)
        self.v = Conv(d_model)
        self.concat = Conv(d_model)


        # nn.init.xavier_uniform_(self.memory)

        # self.weight = nn.Parameter(torch.ones(d_model, num_nodes, seq_length))


        # self.nodevec1 = nn.Parameter(torch.randn( self.num_nodes,10 ), requires_grad=True)
        #


    def forward(self, input, adj_list=None):
        value =  self.v(input)
        # query = query.view(
        #     query.shape[0], -1, self.d_k, query.shape[2], self.seq_length
        # ).permute(0, 1, 4, 3, 2)
        # adj_dyn_1 = torch.softmax(
        #     F.relu(
        #         torch.einsum("bcnt, cm->bnm", value, self.memory).contiguous()
        #         / math.sqrt(value.shape[1])
        #     ),
        #     -1,
        # )
        value = value.view(
            value.shape[0], -1, self.d_k, value.shape[2], self.seq_length
        ).permute(
            0, 1, 4, 3, 2
        )

        # key = torch.softmax(self.memory / math.sqrt(self.d_k), dim=-1)
        # query = torch.softmax(query / math.sqrt(self.d_k), dim=-1)



        # kv = torch.einsum("hlnx, bhlny->bhlxy", key, value)
        # attn_qkv = torch.einsum("bhlnx, bhlxy->bhlny", query, kv)
        adj_f= adj_list


        attn_dyn = torch.einsum("bnm,bhlnc->bhlnc",adj_f , value)



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

class STAMT(nn.Module):
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

        # self.tconv = TConv(channels, layer=4, length=self.input_len)

        # self.start_conv = nn.Conv2d(self.input_dim, channels, kernel_size=(1, 1))

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
        input_data = history_data
        residual_cpu = input_data.cpu()
        import pywt

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


        # history_data = history_data.permute(0, 3, 2, 1)
        # input_data = self.start_conv(input_data)

        # input,adj = self.tconv(input_data)    #构建邻接矩阵
        # three = F.softmax(F.relu(input_data2.squeeze().permute(0, 2, 1) @ input_data2.squeeze()), dim=-1).unsqueeze(-1)
        #
        # one=self.fc_d(self.Temb(history_data.permute(0, 3, 2, 1))[0]).squeeze()
        # two=self.fc_w(self.Temb(history_data.permute(0, 3, 2, 1))[1]).squeeze()
        # one=F.softmax(F.relu(one.permute(1, 0) @ one), dim=-1).unsqueeze(-1).expand_as(three)
        # two = F.softmax(F.relu(two.permute(1, 0) @ two), dim=-1).unsqueeze(-1).expand_as(three)
        # fusion = torch.cat((three,one, two), dim=-1)
        # adj_f = torch.softmax(self.fc(fusion).squeeze(), -1)

        day=history_data[:,1,0,-1]*288
        week=history_data[:,2,0,-1]
        days=self.nodevec_p1[day.cpu().numpy()]
        weeks = self.nodevec_p2[week.cpu().numpy()]
        adp = torch.einsum('ai, jik->ajk', days+weeks, self.nodevec_pk)

        adp = torch.einsum('ck, ajk->ajc', self.node_embeddings, adp)
        input_data=input_data2.squeeze()
        adj_f = torch.einsum('abc, abd->acd', input_data, adp)
        adp = F.relu(adj_f)
        adj_f = F.softmax(adp, dim=2)


        # topk_values, topk_indices = torch.topk(adj_f, k=int(adj_f.shape[1] * 0.3), dim=-1)
        # mask = torch.zeros_like(adj_f)
        # mask.scatter_(-1, topk_indices, 1)
        # adj_f = adj_f * mask




        tem_emb = self.Temb(history_data.permute(0, 3, 2, 1))

        data_st = torch.cat([input_data2] + [tem_emb], dim=1)   #数据由频域数据、时序数据最后一步和时间嵌入共同组成
        #空间记忆点
        data_st = self.SpatialBlock(data_st,adj_f) + self.fc_st2(data_st)* torch.sigmoid(self.fc_st(data_st))

        prediction = self.regression_layer(data_st)

        return prediction
