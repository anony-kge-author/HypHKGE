"""Hyperbolic Knowledge Graph embedding models where all parameters are defined in tangent spaces."""
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from models.base import KGModel
from utils.euclidean import givens_rotations, givens_reflection
from utils.hyperbolic import mobius_add, expmap0, project, hyp_distance_multi_c
#import AttH_TC
HYP_MODELS = ["HypHKGE" ]


class BaseH(KGModel):
    """Trainable curvature for each relationship."""

    def __init__(self, args):
        super(BaseH, self).__init__(args.sizes, args.rank, args.dropout, args.gamma, args.dtype, args.bias,
                                    args.init_size)
        self.entity.weight.data = self.init_size * torch.randn((self.sizes[0], self.rank), dtype=self.data_type)
        self.rel.weight.data = self.init_size * torch.randn((self.sizes[1], 2 * self.rank), dtype=self.data_type)
        self.rel_diag = nn.Embedding(self.sizes[1], self.rank)
        self.rel_diag.weight.data = 2 * torch.rand((self.sizes[1], self.rank), dtype=self.data_type) - 1.0
        self.multi_c = args.multi_c
        if self.multi_c:
            c_init = torch.ones((self.sizes[1], 1), dtype=self.data_type)
        else:
            c_init = torch.ones((1, 1), dtype=self.data_type)
        self.c = nn.Parameter(c_init, requires_grad=True)

    def get_rhs(self, queries, eval_mode):
        """Get embeddings and biases of target entities."""
        if eval_mode:
            return self.entity.weight, self.bt.weight
        else:
            return self.entity(queries[:, 2]), self.bt(queries[:, 2])

    def similarity_score(self, lhs_e, rhs_e, eval_mode):
        """Compute similarity scores or queries against targets in embedding space."""
        lhs_e, c = lhs_e
        return - hyp_distance_multi_c(lhs_e, rhs_e, c, eval_mode) ** 2





class HypHKGE(BaseH):
    def __init__(self,args):
        super(HypHKGE, self).__init__(args)
        self.T = 0                              #双曲层次间变换
        self.R = 1                              #双曲层次内变换
        self.C = 1                              #基于注意力的可学习曲率

        self.rel.weight.data = self.init_size * torch.randn((self.sizes[1], 2 * self.rank), dtype=self.data_type)

        self.act = nn.Softmax(dim=1)


        c_w_init = torch.rand((self.sizes[1], self.rank), dtype=self.data_type)
        self.c_w = nn.Parameter(c_w_init, requires_grad=True)

        c_init = torch.ones((self.sizes[1], 1), dtype=self.data_type)
        self.c = nn.Parameter(c_init, requires_grad=True)

        c_h_init = torch.ones((self.sizes[0], 1), dtype=self.data_type)
        self.c_h = nn.Parameter(c_h_init, requires_grad=True)

        self.rel_c = nn.Embedding(self.sizes[1], self.rank)
        self.rel_c.weight.data = self.init_size * torch.randn((self.sizes[1], self.rank), dtype=self.data_type)

        self.total_a = nn.Embedding(1, self.rank)
        self.total_a.weight.data = torch.randn((1, self.rank), dtype=self.data_type)

    def get_queries(self, queries):
        head_c = self.entity(queries[:, 0])
        rel_c = self.rel_c(queries[:, 1])
        if self.C == 1:
            m = torch.mean(head_c * rel_c, dim=1, keepdim=True)
            n = torch.mean(rel_c * rel_c, dim=1, keepdim=True)
            att_h, att_r = torch.chunk(self.act(torch.cat((m, n), dim=1)),2,dim=1)
            att_embedding = head_c*att_h + rel_c*att_r
            c_b = torch.mean(att_embedding*self.total_a.weight,dim=1,keepdim=True)
            c = F.softplus(c_b)
        else:
            c = 1

        rel1, rel2 = torch.chunk(self.rel(queries[:, 1]), 2, dim=1)
        if self.T == 1:
            point = rel2.reshape(-1,1)
            rel3 = point.repeat(1,2).reshape(-1,2*self.rank)
            rel3_w, rel3_b = torch.chunk(rel3,2,dim=1)
            head_e = head_c * rel3_w
        else:
            head_e = head_c

        head = expmap0(head_e, c)
        rel1 = expmap0(rel1, c)

        if self.R ==1:
            res1 = givens_rotations(self.rel_diag(queries[:, 1]), head)
        else:
            res1 = head
        res2 = project(mobius_add(res1, rel1, c),c)

        return (res2, c), self.bh(queries[:, 0])

