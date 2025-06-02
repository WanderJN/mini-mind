import math
from typing import List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
import os
import pandas as pd

from torch.utils.data import IterableDataset, Dataset
import json
import numpy as np
from transformers import  PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import PretrainedConfig
from transformers.activations import ACT2FN


class MiniMindConfig(PretrainedConfig):
    model_type = "minimind"

    def __init__(
            self,
            dropout: float = 0.0,
            bos_token_id: int = 1,
            eos_token_id: int = 2,
            hidden_act: str = 'silu',
            hidden_size: int = 512,
            intermediate_size: int = None,
            max_position_embeddings: int = 32768,
            num_attention_heads: int = 8,
            num_hidden_layers: int = 8,
            num_key_value_heads: int = 2,
            vocab_size: int = 6400,
            rms_norm_eps: float = 1e-05,
            rope_theta: int = 1000000.0,
            flash_attn: bool = True,
            ####################################################
            # Here are the specific configurations of MOE
            ####################################################
            num_experts_per_tok: int = 2,
            n_routed_experts: int = 4,
            n_shared_experts: int = 1,
            scoring_func: str = 'softmax',
            aux_loss_alpha: float = 0.1,
            seq_aux: bool = True,
            norm_topk_prob: bool = True,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.flash_attn = flash_attn
        ####################################################
        # Here are the specific configurations of MOE
        ####################################################
        self.num_experts_per_tok = num_experts_per_tok  # 每个token选择的专家数量
        self.n_routed_experts = n_routed_experts  # 总的专家数量
        self.n_shared_experts = n_shared_experts  # 共享专家
        self.scoring_func = scoring_func  # 评分函数，默认为'softmax'
        self.aux_loss_alpha = aux_loss_alpha  # 辅助损失的alpha参数
        self.seq_aux = seq_aux  # 是否在序列级别上计算辅助损失
        self.norm_topk_prob = norm_topk_prob  # 是否标准化top-k概率


class FeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config

        if config.intermediate_size is None:
            intermediate_size = int(config.hidden_size * 8 / 3)
            config.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        # 经过一次增维，路由，再降维
        return self.dropout(self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)))


class MoeGate(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts

        self.scoring_func = config.scoring_func
        self.alpha = config.aux_loss_alpha
        self.seq_aux = config.seq_aux

        self.norm_topk_prob = config.norm_topk_prob
        self.gating_dim = config.hidden_size
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        # self.weight参数初始化
        self.reset_parameters()

    def reset_parameters(self) -> None:
        import torch.nn.init as init
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    
    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        # 将hidden_states展平为 (bsz*seq_len, hidden_dim)，相当于对每一个token都进行专家路由操作
        hidden_states = hidden_states.view(-1, h)
        # 执行路由操作，(bsz*seq_len, hidden_dim) * (hidden_dim, n_routed_experts)，每个token输出在各路由下的logits值
        logits = F.linear(hidden_states, self.weight, None)

        if self.scoring_func == 'softmax':
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(f'insupportable scoring function for MoE gating: {self.scoring_func}')
        
        # 取最高的k个专家权重和他们的下标，(bsz*seq_len, top_k)
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        # 再重新对最高的k个专家的权重进行归一化，让他们权重总和为1，(bsz*seq_len, top_k)
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator
        
        # 在训练时需要平衡专家使用频率，防止某些专家被过度使用或完全忽略
        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)  # 形状 (bsz, seq_len * aux_topk)

            # 序列级辅助损失 
            if self.seq_aux:
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)

                # ce记录每个batchsize中所有token 各个专家被选择的次数，最后归一化表示
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                ce.scatter_add(
                    dim=1,       # 按专家维度添加
                    index=topk_idx_for_aux_loss,
                    scr=torch.ones(bsz, seq_len*aux_topk, device=hidden_states.device)
                ).div_(     # 归一化：每个专家能被选中的概率是 aux_topk / self.n_routed_experts次
                    seq_len * aux_topk / self.n_routed_experts
                )

                # 计算负载均衡损失
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
            else:
                # 将每个batchsize每个seq每个选取的专家得分编号，编码成one-hot
                # [[1, 0, 0, 0],  # 第1个token，选择专家0
                #  [0, 0, 1, 0],  # 第1个token，选择专家2
                #  [1, 0, 0, 0],  # 第2个token，选择专家0
                #  [0, 0, 0, 1],  # 第2个token，选择专家3
                #  [...]]         # 第n个token，选择专家x
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts) # 形状 (bsz * seq_len * aux_topk, n_routed_experts)

                ce = mask_ce.float().mean(0)      # 计算所有 token 下，每个专家的 平均激活概率，形状 (n_routed_experts)
                pi = scores_for_aux.mean(0)       # 所有 token 下，每个专家的 路由分数的平均值，形状 (n_routed_experts)
                fi = ce * self.n_routed_experts   # 所有 token 下，每个专家的 使用频率
                aux_loss = (pi * fi).sum() * self.alpha 
        else:
            aux_loss = 0
    
        return topk_idx, topk_weight, aux_loss
            

class MoeFeedForward(nn.Module):
    def __init__(self, config:MiniMindConfig):
        super().__init__()
    
        self.config = config
        self.experts = nn.ModuleList([
            FeedForward(config)
            for _ in range(config.n_routed_experts)
        ])
        self.gate = MoeGate(config)
        # 创建共享专家网络
        if config.n_shared_experts > 0:
            self.shared_experts = nn.ModuleList([
                FeedForward(config)
                for _ in range(config.shared_experts)
            ])

    def forward(self, x):
        identity = x
        orig_shape = x.shape
        bsz, seq_len, _ = x.shape

        # 使用门控机制选择专家
        topk_idx, topk_weight, aux_loss = self.gate(x)
        x = x.view(-1, x.shape[-1])
        flat_topk_idx = topk_idx.view(-1)  # (bsz * seq * topk)

        if self.training:
            # 将输入的x矩阵重复topk次
            x = x.repeat_interleave(self.config.num_experts_per_tok, dim=0) # (topk * bsz * seq, hid_dim)
            y = torch.empty_like(x, dtype=torch.float16)

            # 对展平的所有token计算，遍历每个专家，只计算index相符的token
            for i, expert in enumerate(self.experts):
                y[flat_topk_idx == i] = expert(x[flat_topk_idx == i]).to(y.dtype)


