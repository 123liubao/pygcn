import torch.nn as nn
import torch.nn.functional as F
from pygcn.layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        # nfeat 初始特征数；nhid 隐藏层特征数
        self.gc1 = GraphConvolution(nfeat, nhid)  # 构造第一层 GCN
        # nhid 隐藏层特征数；nclass 最终类别数
        self.gc2 = GraphConvolution(nhid, nclass)  # 构造第二层 GCN
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)
