import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 选择cpu或者GPU
seed_gpu = 0
np.random.seed(seed_gpu)
torch.manual_seed(seed_gpu)
torch.cuda.manual_seed(seed_gpu)
cudnn.deterministic = True
cudnn.benchmark = False

import argparse
from parse import parse_method, parser_add_main_args
from graphformer import *


class GCNLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, A: torch.Tensor):
        super(GCNLayer, self).__init__()
        self.A = A
        self.BN = nn.BatchNorm1d(input_dim)  ##########批归一化（Batch Normalization），使每一层的特征在训练时更稳定和更容易收敛。###########
        self.Activition = nn.LeakyReLU()  ###########激活函数，用来引入非线性，从而让模型有更强的表达能力。##########
        self.sigma1 = torch.nn.Parameter(torch.tensor([0.1], requires_grad=True))
        # 第一层GCN
        self.GCN_liner_theta_1 = nn.Sequential(nn.Linear(input_dim, 256))
        self.GCN_liner_out_1 = nn.Sequential(nn.Linear(input_dim, output_dim))
        nodes_count = self.A.shape[0]
        self.I = torch.eye(nodes_count, nodes_count, requires_grad=False).to(
            device)  # 这个函数为单位矩阵，在图卷积中经常用它来让每个节点在聚合邻居时也考虑到自己
        self.mask = torch.ceil(self.A * 0.00001)  # 返回具有 input 元素的ceil的新张量，该整数是大于或等于每个元素的最小整数。

    def A_to_D_inv(self, A: torch.Tensor):
        D = A.sum(1)
        D_hat = torch.diag(torch.pow(D, -0.5))
        return D_hat

    def forward(self, H, model='normal'):
        # # 方案一：minmax归一化
        # H = self.BN(H)
        # H_xx1= self.GCN_liner_theta_1(H)
        # A = torch.clamp(torch.sigmoid(torch.matmul(H_xx1, H_xx1.t())), min=0.1) * self.mask + self.I
        # if model != 'normal': A=torch.clamp(A,0.1) #This is a trick.
        # D_hat = self.A_to_D_inv(A)
        # A_hat = torch.matmul(D_hat, torch.matmul(A,D_hat))
        # output = torch.mm(A_hat, self.GCN_liner_out_1(H))
        # output = self.Activition(output)

        # # 方案二：softmax归一化 (加速运算)
        H = self.BN(H)
        H_xx1 = self.GCN_liner_theta_1(H)
        e = torch.sigmoid(torch.matmul(H_xx1, H_xx1.t()))
        zero_vec = -9e15 * torch.ones_like(e)
        A = torch.where(self.mask > 0, e, zero_vec) + self.I
        if model != 'normal': A = torch.clamp(A,
                                              0.1)  # This is a trick for the Indian Pines.torch.clamp(input, min, max, out=None) 将输入input张量每个元素的范围限制到区间 [min,max]，返回结果到一个新张量
        A = F.softmax(A, dim=1)
        output = self.Activition(
            torch.mm(A, self.GCN_liner_out_1(H)))  ##### 最终输出：将特征H通过线性层映射，再和A这个加权矩阵相乘，融合邻居信息，然后通过激活函数得到新的特征######

        return output, A


class MSSSConv(nn.Module):
    def __init__(self, in_ch, out_ch, ks_list=(5, 7, 9)):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_ch)
        self.point = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.branches = nn.ModuleList([
            nn.Conv2d(out_ch, out_ch, kernel_size=k, padding=k//2, groups=out_ch, bias=False)
            for k in ks_list
        ])
        self.act = nn.LeakyReLU(inplace=True)
        self.fuse = nn.Conv2d(out_ch * len(ks_list), out_ch, 1, bias=False)

    def forward(self, x):
        x = self.bn(x)
        x = self.point(x)               # 通道映射
        x = self.act(x)
        feats = [branch(x) for branch in self.branches]   # 多尺度深度卷积
        x = torch.cat(feats, dim=1)     # 拼接所有尺度
        x = self.fuse(x)                # 通道融合
        x = self.act(x)
        return x

class GTCFN(nn.Module):
    def __init__(self,
                 height: int,
                 width: int,
                 changel: int,
                 class_count: int,
                 Q: torch.Tensor,
                 A: torch.Tensor,
                 adj: torch.Tensor,
                 denoise_layers_count: int = 3,
                 pixel_layers_count: int = 2,
                 gcn_layers_count: int = 2,
                 model='normal'):
        super(GTCFN, self).__init__()
        # 类别数,即网络最终输出通道数
        self.class_count = class_count  # 类别数
        # 网络输入数据大小
        self.channel = changel  # 200
        self.height = height  # 145
        self.width = width  # 145
        self.Q = Q  # 20125*196
        self.A = A  # 196*196
        self.adj = adj
        self.model = model  # smoothed
        self.norm_col_Q = Q / (torch.sum(Q, 0, keepdim=True))  # 列归一化Q20125*196

        layers_count = 2

        # Spectra Transformation Sub-Network
        self.CNN_denoise = nn.Sequential()
        for i in range(denoise_layers_count):  #####原本channel=200个波段（通道）的高光谱数据会被转换成128个特征通道#####
            if i == 0:
                self.CNN_denoise.add_module('CNN_denoise_BN' + str(i), nn.BatchNorm2d(self.channel))
                self.CNN_denoise.add_module('CNN_denoise_Conv' + str(i),
                                            nn.Conv2d(self.channel, 128, kernel_size=(1, 1)))
                self.CNN_denoise.add_module('CNN_denoise_Act' + str(i), nn.LeakyReLU())
            else:
                self.CNN_denoise.add_module('CNN_denoise_BN' + str(i), nn.BatchNorm2d(128), )
                self.CNN_denoise.add_module('CNN_denoise_Conv' + str(i), nn.Conv2d(128, 128, kernel_size=(1, 1)))
                self.CNN_denoise.add_module('CNN_denoise_Act' + str(i), nn.LeakyReLU())

        # Pixel-level Convolutional Sub-Network——利用SSConv层对已经变换后的特征进行空间和光谱的双重提取
        self.CNN_Branch = nn.Sequential(
            MSSSConv(128, 128, ks_list=(5, 7, 9)),
            MSSSConv(128, 64, ks_list=(5, 7, 9))
        )

        # Superpixel-level Graph Sub-Network——利用GCNLayer根据图结构（节点为超像素）整合信息，从超像素层面进一步提升特征表示能力。
        self.GCN_Branch = nn.Sequential()
        for i in range(gcn_layers_count):
            if i < gcn_layers_count - 1:
                self.GCN_Branch.add_module('GCN_Branch' + str(i),
                                           GCNLayer(128, 128, self.A))  ###将利用超像素的邻接关系A来融合来自相邻超像素的特征###
            else:
                self.GCN_Branch.add_module('GCN_Branch' + str(i), GCNLayer(128, 64, self.A))

        # Softmax layer——用于最终分类输出
        self.Softmax_linear = nn.Sequential(nn.Linear(128, self.class_count))  ##这里的128维度是将GCN的64维和CNN的64维结合在一起##

        parser = argparse.ArgumentParser(description='General Training Pipeline')
        parser_add_main_args(parser)
        args = parser.parse_args()
        print(args)


        self.graphformer = GraphFormer(in_channels=128, hidden_channels=args.hidden_channels, out_channels=64,
                                     num_layers=args.num_layers,
                                     dropout=args.dropout,
                                     num_heads=args.num_heads, use_bn=args.use_bn, nb_random_features=args.M,
                                     use_gumbel=args.use_gumbel, use_residual=args.use_residual, use_act=args.use_act,
                                     use_jk=args.use_jk,
                                     nb_gumbel_sample=args.K, rb_order=args.rb_order, rb_trans=args.rb_trans).to(device)

    def forward(self, x: torch.Tensor):  #####前向传播的过程#####
        '''
        :param x: H*W*C
        :return: probability_map
        '''
        (h, w, c) = x.shape

        # 先去除噪声
        noise = self.CNN_denoise(
            torch.unsqueeze(x.permute([2, 0, 1]), 0))  # permute是维度交换https://zhuanlan.zhihu.com/p/76583143
        noise = torch.squeeze(noise, 0).permute([1, 2, 0])
        clean_x = noise  # clean_x是一个(H, W, 128)的张量
        ##########STsN层，得到145*145*128的张量 clean_x #########

        clean_x_flatten = clean_x.reshape([h * w, -1])
        superpixels_flatten = torch.mm(self.norm_col_Q.t(),
                                       clean_x_flatten)  # 将像素特征聚合到超像素层面上，得到每个超像素的平均特征表示(低频部分) 196*21025 * 21025*128
        hx = clean_x

        # CNN与GCN分两条支路
        CNN_result = self.CNN_Branch(torch.unsqueeze(hx.permute([2, 0, 1]), 0))  # spectral-spatial convolution
        CNN_result = torch.squeeze(CNN_result, 0).permute([1, 2, 0]).reshape([h * w, -1])
        ##########PCsN层，CNN输出结果，得到21025*64的张量 CNN_result #########

        # GCN层 1 转化为超像素 x_flat 乘以 列归一化Q
        H = superpixels_flatten
        if self.model == 'normal':
            for i in range(len(self.GCN_Branch)): H, _ = self.GCN_Branch[i](H)
        else:
            for i in range(len(self.GCN_Branch)): H, _ = self.GCN_Branch[i](H, model='smoothed')
        ##########SGsN层:GCN层，得到196*64的张量 H #########

        GCN_result = torch.matmul(self.Q, H)  # 这里self.norm_row_Q == self.Q
        ##########GCN输出结果，21025*196*196*64得到21025*64的张量 H #########

        graphformer_result, _ = self.graphformer(superpixels_flatten, self.adj,
                                               tau=0.25)  #######tau=0.25是该模块的超参数，影响注意力的权重分布
        NF_result = torch.matmul(self.Q, graphformer_result)

        # 两组特征融合(两种融合方式)
        Y = torch.cat([NF_result, CNN_result], dim=-1)
        Y = self.Softmax_linear(Y)  # 128*16层输出21025*16
        Y = F.softmax(Y, -1)
        return Y

