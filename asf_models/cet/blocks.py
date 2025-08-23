# -*- coding: utf-8 -*-

# Copyright (c) Microsoft Corporation.

# Licensed under the MIT License.



import math

from typing import Callable, Dict, Optional, Tuple



import torch

import torch.nn as nn

import torch.nn.functional as F

from torch import Tensor

from torch.nn import Parameter



torch._C._jit_set_profiling_mode(False)

torch._C._jit_set_profiling_executor(False)

torch._C._jit_override_can_fuse_on_cpu(True)

torch._C._jit_override_can_fuse_on_gpu(True)



@torch.jit.script

def softmax_dropout(input, dropout_prob: float, is_training: bool):

    return F.dropout(F.softmax(input, -1), dropout_prob, is_training)



# @torch.jit.script

def SmoothSoftmax(

    input: Tensor,

    local_attn_weight: Tensor,

    return_dtype: Optional[torch.dtype] = None,

    eps: float = 1e-16,

) -> Tensor:

    # input = input.to(torch.float64)

    # Compute e_ij

    max_value = input.max(dim=-1, keepdim=True).values

    local_attn_weight = local_attn_weight.to(input.dtype)



    max_value = torch.where(max_value == float("-inf"), torch.zeros_like(max_value), max_value)



    input = input - max_value

    e_ij = torch.exp(input) * (local_attn_weight.unsqueeze(1))



    # Compute softmax along the last dimension

    softmax = e_ij / (torch.sum(e_ij, dim=-1, keepdim=True) + eps)



    if return_dtype is not None:

        softmax = softmax.to(return_dtype)



    return softmax



@torch.jit.script

def gaussian(x, mean, std):

    pi = 3.14159

    a = (2 * pi) ** 0.5

    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)



class GaussianLayer(nn.Module):

    def __init__(self, K=128, edge_types=1024):

        super().__init__()

        self.K = K

        self.means = nn.Embedding(1, K)

        self.stds = nn.Embedding(1, K)

        self.mul = nn.Embedding(edge_types, 1, padding_idx=0)

        self.bias = nn.Embedding(edge_types, 1, padding_idx=0)

        nn.init.uniform_(self.means.weight, 0, 3)

        nn.init.uniform_(self.stds.weight, 0, 3)

        nn.init.constant_(self.bias.weight, 0)

        nn.init.constant_(self.mul.weight, 1)



    def forward(self, x, edge_types):

        mul = self.mul(edge_types).sum(dim=-2)

        bias = self.bias(edge_types).sum(dim=-2)

        x = mul * x.unsqueeze(-1) + bias

        x = x.expand(-1, -1, -1, self.K)

        mean = self.means.weight.float().view(-1)

        std = self.stds.weight.float().view(-1).abs() + 1e-5

        return gaussian(x.float(), mean, std).type_as(self.means.weight)



class NodeGaussianLayer(nn.Module):

    def __init__(self, K=128, edge_types=1024):

        super().__init__()

        self.K = K

        self.means = nn.Embedding(1, K)

        self.stds = nn.Embedding(1, K)

        self.mul = nn.Embedding(edge_types, 1, padding_idx=0)

        self.bias = nn.Embedding(edge_types, 1, padding_idx=0)

        nn.init.uniform_(self.means.weight, 0, 3)

        nn.init.uniform_(self.stds.weight, 0, 3)

        nn.init.constant_(self.bias.weight, 0)

        nn.init.constant_(self.mul.weight, 1)



    def forward(self, x, edge_types):

        mul = self.mul(edge_types).sum(dim=-2)

        bias = self.bias(edge_types).sum(dim=-2)

        x = mul * x.unsqueeze(-1) + bias

        x = x.expand(-1, -1, self.K)

        mean = self.means.weight.float().view(-1)

        std = self.stds.weight.float().view(-1).abs() + 1e-5

        return gaussian(x.float(), mean, std).type_as(self.means.weight)



class NonLinear(nn.Module):

    def __init__(self, input, output_size, hidden=None):

        super(NonLinear, self).__init__()

        if hidden is None:

            hidden = input

        self.layer1 = nn.Linear(input, hidden)

        self.layer2 = nn.Linear(hidden, output_size)



    def forward(self, x):

        x = F.gelu(self.layer1(x))

        x = self.layer2(x)

        return x



class NodeTaskHead(nn.Module):

    def __init__(

        self,

        embed_dim: int,

        num_heads: int,

    ):

        super().__init__()

        self.embed_dim = embed_dim

        self.q_proj: Callable[[Tensor], Tensor] = nn.Linear(embed_dim, embed_dim)

        self.k_proj: Callable[[Tensor], Tensor] = nn.Linear(embed_dim, embed_dim)

        self.v_proj: Callable[[Tensor], Tensor] = nn.Linear(embed_dim, embed_dim)

        self.num_heads = num_heads

        self.scaling = (embed_dim // num_heads) ** -0.5

        self.force_proj1: Callable[[Tensor], Tensor] = nn.Linear(embed_dim, 1)

        self.force_proj2: Callable[[Tensor], Tensor] = nn.Linear(embed_dim, 1)

        self.force_proj3: Callable[[Tensor], Tensor] = nn.Linear(embed_dim, 1)



    def forward(

        self,

        query: Tensor,

        attn_bias: Tensor,

        delta_pos: Tensor,

    ) -> Tensor:

        bsz, n_node, _ = query.size()

        q = (

            self.q_proj(query).view(bsz, n_node, self.num_heads, -1).transpose(1, 2)

            * self.scaling

        )

        k = self.k_proj(query).view(bsz, n_node, self.num_heads, -1).transpose(1, 2)

        v = self.v_proj(query).view(bsz, n_node, self.num_heads, -1).transpose(1, 2)

        attn = q @ k.transpose(-1, -2)  # [bsz, head, n, n]

        attn_probs = softmax_dropout(

            attn.view(-1, n_node, n_node) + attn_bias, 0.1, self.training

        ).view(bsz, self.num_heads, n_node, n_node)

        rot_attn_probs = attn_probs.unsqueeze(-1) * delta_pos.unsqueeze(1).type_as(

            attn_probs

        )  # [bsz, head, n, n, 3]

        rot_attn_probs = rot_attn_probs.permute(0, 1, 4, 2, 3)

        x = rot_attn_probs @ v.unsqueeze(2)  # [bsz, head , 3, n, d]

        x = x.permute(0, 3, 2, 1, 4).contiguous().view(bsz, n_node, 3, -1)

        f1 = self.force_proj1(x[:, :, 0, :]).view(bsz, n_node, 1)

        f2 = self.force_proj2(x[:, :, 1, :]).view(bsz, n_node, 1)

        f3 = self.force_proj3(x[:, :, 2, :]).view(bsz, n_node, 1)

        cur_force = torch.cat([f1, f2, f3], dim=-1).float()

        return cur_force



class GatedEquivariantBlock(nn.Module):

    """Gated Equivariant Block as defined in Schütt et al. (2021):

    Equivariant message passing for the prediction of tensorial properties and molecular spectra

    """



    def __init__(

        self,

        hidden_channels,

        out_channels,

        intermediate_channels=None,

        activation="silu",

        scalar_activation=False,

    ):

        super(GatedEquivariantBlock, self).__init__()

        self.out_channels = out_channels



        if intermediate_channels is None:

            intermediate_channels = hidden_channels



        self.vec1_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)

        self.vec2_proj = nn.Linear(hidden_channels, out_channels, bias=False)



        act_class_mapping = {

            "silu": nn.SiLU,

            "tanh": nn.Tanh,

            "sigmoid": nn.Sigmoid,

        }



        act_class = act_class_mapping[activation]

        self.update_net = nn.Sequential(

            nn.Linear(hidden_channels * 2, intermediate_channels),

            act_class(),

            nn.Linear(intermediate_channels, out_channels * 2),

        )



        self.act = act_class() if scalar_activation else None



    # def reset_parameters(self):

    #     nn.init.xavier_uniform_(self.vec1_proj.weight)

    #     nn.init.xavier_uniform_(self.vec2_proj.weight)

    #     nn.init.xavier_uniform_(self.update_net[0].weight)

    #     self.update_net[0].bias.data.fill_(0)

    #     nn.init.xavier_uniform_(self.update_net[2].weight)

    #     self.update_net[2].bias.data.fill_(0)



    def forward(self, x, v):

        vec1 = torch.norm(self.vec1_proj(v), dim=-2)

        vec2 = self.vec2_proj(v)



        x = torch.cat([x, vec1], dim=-1)

        x, v = torch.split(self.update_net(x), self.out_channels, dim=-1)

        v = v.unsqueeze(-2) * vec2



        if self.act is not None:

            x = self.act(x)

        return x, v



class EquivariantVectorOutput(nn.Module):

    def __init__(self, hidden_channels=768, activation="silu", d_tilde=1):

        super(EquivariantVectorOutput, self).__init__()

        self.output_network = nn.ModuleList(

            [

                GatedEquivariantBlock(

                    hidden_channels,

                    hidden_channels // 2,

                    activation=activation,

                    scalar_activation=True,

                ),

                GatedEquivariantBlock(hidden_channels // 2, 1, activation=activation),

            ]

        )



        self.reset_parameters(d_tilde)



    def reset_parameters(self, d_tilde):

        for layer in self.output_network:

            layer.reset_parameters(d_tilde)



    def forward(self, x, v):

        for layer in self.output_network:

            x, v = layer(x, v)

        return v.squeeze(-1)



class SO3_LinearV2(torch.nn.Module):

    def __init__(self, in_features, out_features, lmax, bias=True,method=1,sph_args=None):

        '''

            1. Use `torch.einsum` to prevent slicing and concatenation

            2. Need to specify some behaviors in `no_weight_decay` and weight initialization.

        '''

        super().__init__()

        self.in_features = in_features

        self.out_features = out_features

        self.lmax = lmax

        self.method = method

        if self.method==1:

            self.weight = torch.nn.Parameter(torch.randn((self.lmax + 1), out_features, in_features))

            bound = 1 / math.sqrt(self.in_features)

            torch.nn.init.uniform_(self.weight, -bound, bound)

            if bias:

                self.bias = torch.nn.Parameter(torch.zeros(out_features))

   

            expand_index = torch.zeros([(lmax + 1) ** 2]).long()

            for l in range(lmax + 1):

                start_idx = l ** 2

                length = 2 * l + 1

                expand_index[start_idx : (start_idx + length)] = l

            self.register_buffer('expand_index', expand_index)

        elif self.method == 2:

            self.sph_num, self.sph_offset = sph_args

            self.linear = nn.ModuleList(

                [

                    nn.Linear(in_features, out_features, bias=(_==0))

                    for _ in range(self.sph_num)

                ]

            )

            for out_proj in self.linear:

                nn.init.xavier_uniform_(out_proj.weight, gain=1.0 / math.sqrt(1))

   

    def forward(self, input_embedding):

        if self.method==1:

            output_shape = input_embedding.shape[:-2]

            l_sum, hidden = input_embedding.shape[-2:]

            input_embedding = input_embedding.reshape(

                [output_shape.numel()] + [l_sum, hidden]

            )

            weight = torch.index_select(

                self.weight, dim=0, index=self.expand_index

            )  # [(L_max + 1) ** 2, C_out, C_in]

            out = torch.einsum(

                "bmi, moi -> bmo", input_embedding, weight

            ).to(self.bias.dtype)

            bias = self.bias.view(1, 1, self.out_features)

            out[:, 0:1, :] = out.narrow(1, 0, 1) + bias



            out = out.reshape(output_shape + (l_sum, self.out_features))



        elif self.method==2:

            attn_temp = []

            for i in range(self.sph_num):

                new_vec = self.linear[i](input_embedding[:,:,self.sph_offset[i]:self.sph_offset[i+1],:])

                if len(new_vec.shape) == 3:

                    new_vec = new_vec.unsqueeze(-2)

                attn_temp.append(new_vec)

            out = torch.cat(attn_temp, dim=-2)

       

        return out



class EquivariantLayerNorm(nn.Module):

    __constants__ = ["normalized_shape", "elementwise_linear"]

    normalized_shape: Tuple[int, ...]

    eps: float

    elementwise_linear: bool



    def __init__(

        self,

        normalized_shape: int,

        eps: float = 1e-5,

        elementwise_linear: bool = True,

        device=None,

        dtype=None,

        use_scale: bool = True,

        rank_num: int = 3,

        use_diff_liear_for_sph=False,

        sph_args=None,

    ) -> None:

        factory_kwargs = {"device": device, "dtype": dtype}

        super(EquivariantLayerNorm, self).__init__()



        self.normalized_shape = (int(normalized_shape),)

        self.eps = eps

        self.elementwise_linear = elementwise_linear



        use_diff_liear_for_sph=False # since it is too slow, we do not use it



        self.use_diff_liear_for_sph = use_diff_liear_for_sph

        self.sph_num, self.sph_offset = sph_args

        if self.elementwise_linear:

            if use_diff_liear_for_sph:

                self.weight=nn.ParameterList(

                    [

                        nn.Parameter(

                            torch.empty(self.normalized_shape, **factory_kwargs)

                        )

                        for _ in range(self.sph_num)

                    ]

                )

            else:

                self.weight = nn.Parameter(

                    torch.empty(self.normalized_shape, **factory_kwargs)

                )

        else:

            self.register_parameter(

                "weight", None

            )  # Without bias term to preserve equivariance!



        self.reset_parameters()

        self.use_scale = use_scale

        self.rank_num = rank_num

        if use_scale:

            self.eps = self.eps**2



        # self.linear = nn.Linear(self.rank_num*self.normalized_shape, self.rank_num*self.normalized_shape, bias=False)



    def reset_parameters(self) -> None:

        if self.elementwise_linear:

            if self.use_diff_liear_for_sph:

                for weight in self.weight:

                    nn.init.ones_(weight)

            else:

                nn.init.ones_(self.weight)



    def mean_center(self, input):

        return input - input.mean(-1, keepdim=True)



    def covariance(self, input):

        return 1 / self.normalized_shape[0] * input @ input.transpose(-1, -2)



    def symsqrtinv(self, matrix):

        """Compute the inverse square root of a positive definite matrix.

        Based on https://github.com/pytorch/pytorch/issues/25481

        """

        _, s, v = matrix.svd()

        good = s > s.max(-1, True).values * s.size(-1) * torch.finfo(s.dtype).eps

        components = good.sum(-1)

        common = components.max()

        unbalanced = common != components.min()

        if common < s.size(-1):

            s = s[..., :common]

            v = v[..., :common]

            if unbalanced:

                good = good[..., :common]

        if unbalanced:

            s = s.where(good, torch.zeros((), device=s.device, dtype=s.dtype))

        return (v * 1 / torch.sqrt(s + self.eps).unsqueeze(-2)) @ v.transpose(-2, -1)



    def forward(self, input: torch.Tensor) -> torch.Tensor:

        if self.use_scale:

            # input B x N x rank_num x d

            input = input.to(torch.float64)  # 180M

            input_norm = input.pow(2).sum(dim=-2, keepdim=True)  # B x N x 1 x C

            input_norm = torch.mean(input_norm, dim=-1)  # B x N x 1

            input_norm = (input_norm + self.eps).pow(-0.5)  # B x N x 1



            # input_norm = input.pow(2).sum(dim=-1) # B x N x 3

            # input_norm = torch.mean(input_norm,dim=2,keepdim=True) # B x N x 1

            # input_norm = (input_norm + self.eps).pow(-0.5) # B x N x 1



            if self.elementwise_linear:

                if self.use_diff_liear_for_sph:

                    input_temp = []

                    for i in range(self.sph_num):

                        new_vec = input[:,:,self.sph_offset[i]:self.sph_offset[i+1],:] * self.weight[i]

                        if len(new_vec.shape) == 3:

                            new_vec = new_vec.unsqueeze(-2)

                        input_temp.append(new_vec)

                    input = torch.cat(input_temp, dim=-2)



                    #     new_vec = input[:,:,self.sph_offset[i]:self.sph_offset[i+1],:]

                    #     if len(new_vec.shape) == 3:

                    #         new_vec = new_vec.unsqueeze(-2)



                    #     new_vec_norm = new_vec.pow(2).sum(dim=-2, keepdim=True)  # B x x N x 1 x d

                    #     new_vec_norm = torch.mean(new_vec_norm, dim=-1) # B x N x 1

                    #     new_vec_norm_mask = new_vec_norm < 1e-16 # B x N x 1

                    #     new_vec_norm = (new_vec_norm).pow(-0.5) # B x N x 1



                    #     # Avoid division by zero

                    #     new_vec_norm[new_vec_norm_mask] = 1.0  # Set to 1 if norm is too small to avoid division by zero



                    #     new_vec = new_vec * new_vec_norm.unsqueeze(-1) * self.weight[i]



                    #     new_vec[new_vec_norm_mask.unsqueeze(-1)] = 0.0  # Handle cases where norm was too small



                    #     input_temp.append(new_vec)

                    # output = torch.cat(input_temp, dim=-2)

                    del input_temp



                    dtype = self.weight[0].dtype

                else:

                    input = input * self.weight

                    dtype = self.weight.dtype

            else:

                input = input



            output = input * input_norm.unsqueeze(-1)  # B x N x 3 x d



            return output.to(dtype)

        else:

            input = input.to(

                torch.float64

            )  # Need double precision for accurate inversion.

            input = self.mean_center(input)

            # We use different diagonal elements in case input matrix is approximately zero,

            # in which case all singular values are equal which is problematic for backprop.

            # See e.g. https://pytorch.org/docs/stable/generated/torch.svd.html

            reg_matrix = (

                torch.diag(torch.tensor([1.0, 2.0, 3.0]))

                .unsqueeze(0)

                .to(input.device)

                .type(input.dtype)

            )

            covar = self.covariance(input) + self.eps * reg_matrix

            covar_sqrtinv = self.symsqrtinv(covar)

            return (covar_sqrtinv @ input).to(self.weight.dtype) * self.weight.reshape(

                1, 1, self.normalized_shape[0]

            )



    def extra_repr(self) -> str:

        return "{normalized_shape}, " "elementwise_linear={elementwise_linear}".format(

            **self.__dict__

        )



class InvariantAttention(nn.Module):

    def __init__(

        self,

        hidden_channels,

        head_dim,

        dropout,

        d_tilde=1,

        norm_eps: float = 1e-5,

    ):

        super().__init__()

        self.hidden_channels = hidden_channels

        self.head_dim = head_dim

        self.num_heads = hidden_channels // head_dim



        self.out_proj = nn.Linear(hidden_channels, hidden_channels, bias=True)



        self.dropout = dropout

        self.attn_ln = nn.LayerNorm(hidden_channels, eps=norm_eps)

        self.scaling = ((self.head_dim / d_tilde) ** 0.5) / self.head_dim



        self.reset_parameters(1)



    def reset_parameters(self, d_tilde):

        nn.init.xavier_uniform_(self.out_proj.weight, gain=1.0 / math.sqrt(d_tilde))

        self.out_proj.bias.data.fill_(0)



    def forward(

        self,

        q,

        k,

        v,

        attn_bias,

        key_padding_mask,

        pbc_expand_batched: Optional[Dict] = None,

        softmax_weights: Optional[Tensor] = None,

    ):



        # This is part of a workaround to get around fork/join parallelism

        # not supporting Optional types.

        local_attention_mask, expand_mask = key_padding_mask[0], key_padding_mask[1]

        if local_attention_mask is not None and local_attention_mask.dim() == 0:

            local_attention_mask = None



        q *= self.scaling



        if pbc_expand_batched is not None and softmax_weights is None:

            outcell_index = pbc_expand_batched["outcell_index"]

            embed_dim = q.size()[-1]

            outcell_index = outcell_index.unsqueeze(-1).expand(-1, -1, embed_dim)

            expand_k = torch.gather(k, dim=1, index=outcell_index)

            expand_v = torch.gather(v, dim=1, index=outcell_index)

            k = torch.cat([k, expand_k], dim=1)

            v = torch.cat([v, expand_v], dim=1)

        else:

            local_attention_mask = None



        bsz, tgt_len, src_len = q.shape[0], q.shape[1], k.shape[1]



        q = q.reshape(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)

        k = k.reshape(bsz, src_len, self.num_heads, self.head_dim).transpose(1, 2)

        v = v.reshape(bsz, src_len, self.num_heads, self.head_dim).transpose(1, 2)



        attn_weights = q.matmul(

            k.transpose(-1, -2)

        )  # (bsz, num_heads, tgt_len, src_len)



        if attn_bias is not None and softmax_weights is None:

            attn_weights = attn_weights + attn_bias



        if local_attention_mask is not None:

            if softmax_weights is None:

                attn_weights = attn_weights.masked_fill(

                    (local_attention_mask <= 1e-6).unsqueeze(1), float("-inf")

                )

        if expand_mask is not None:

            attn_weights = attn_weights.masked_fill(

                expand_mask.unsqueeze(1).unsqueeze(1).to(torch.bool),

                float("-inf"),

            )



        # attn_probs_float = F.dropout(

        #     SmoothSoftmax(attn_weights, local_attention_mask, return_dtype=torch.float32),

        #     # F.softmax(attn_weights, dim=-1, dtype=torch.float32),

        #     self.dropout,

        #     training=self.training,

        # )



        attn_probs_float = F.dropout(

            F.softmax(attn_weights, dim=-1, dtype=torch.float32),

            self.dropout,

            training=self.training,

        )



        # if local_attention_mask is not None:

        #     attn_probs_float = attn_probs_float * local_attention_mask.unsqueeze(1)

        #     attn_probs_float = attn_probs_float / (attn_probs_float.sum(dim=-1, keepdim=True) + 1e-10)

        # elif softmax_weights is not None:

        #     attn_probs_float = attn_probs_float * softmax_weights

        #     attn_probs_float = attn_probs_float / (attn_probs_float.sum(dim=-1, keepdim=True) + 1e-10)

       



        # if local_attention_mask is not None:

        #     attn_probs_float = (

        #         torch.mul(attn_probs_float, local_attention_mask.unsqueeze(1))

        #         .float()

        #         .type_as(attn_probs_float)

        #     )



        if local_attention_mask is not None:

            attn_probs_float=attn_probs_float.to(torch.float64)

            attn_probs_float = attn_probs_float * local_attention_mask.unsqueeze(1)

            temp_attn_probs_float = (

                torch.mul(attn_probs_float, local_attention_mask.unsqueeze(1))

                .float()

                .type_as(attn_probs_float)

            )

            attn_probs_float = temp_attn_probs_float / (attn_probs_float.sum(dim=-1, keepdim=True) + 1e-10)

        elif softmax_weights is not None:

            attn_probs_float=attn_probs_float.to(torch.float64)

            softmax_weights,weighted_softmax_weights = softmax_weights[0], softmax_weights[1]

            temp_attn_probs_float = attn_probs_float * weighted_softmax_weights

            attn_probs_float = attn_probs_float * softmax_weights

            attn_probs_float = temp_attn_probs_float / (attn_probs_float.sum(dim=-1, keepdim=True) + 1e-10)



        attn_probs = attn_probs_float.type_as(

            attn_weights

        )  # (bsz, num_heads, tgt_len, src_len)



        attn = attn_probs.matmul(v)  # (bsz, num_heads, tgt_len, head_dim)

        attn = attn.transpose(1, 2).reshape(bsz, tgt_len, self.hidden_channels)

        attn = self.attn_ln(attn)

        attn = self.out_proj(attn)



        return attn



class EquivariantAttention(nn.Module):

    def __init__(

        self,

        hidden_channels,

        head_dim,

        dropout,

        d_tilde=1,

        norm_eps: float = 1e-5,

        rank_num: int = 3,

        use_diff_liear_for_sph=False,

        sph_args=None,

    ):

        super().__init__()

        self.hidden_channels = hidden_channels

        self.head_dim = head_dim

        self.num_heads = hidden_channels // head_dim



        self.use_diff_liear_for_sph = use_diff_liear_for_sph

        self.sph_num, self.sph_offset = sph_args

        if self.use_diff_liear_for_sph:

            self.out_proj = SO3_LinearV2(

                hidden_channels, hidden_channels,lmax=self.sph_num-1,

            )

        else:

            self.out_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)



        self.dropout = dropout

        self.attn_ln = EquivariantLayerNorm(hidden_channels, eps=norm_eps, use_diff_liear_for_sph=use_diff_liear_for_sph, sph_args=sph_args)

        self.scaling = ((self.head_dim / (d_tilde * 3)) ** 0.5) / self.head_dim



        self.rank_num = rank_num



        self.reset_parameters(1)



    def reset_parameters(self, d_tilde):

        if self.use_diff_liear_for_sph:

            pass

        else:

            nn.init.xavier_uniform_(self.out_proj.weight, gain=1.0 / math.sqrt(d_tilde))



    def forward(

        self,

        q,

        k,

        v,

        attn_bias,

        key_padding_mask,

        pbc_expand_batched: Optional[Dict] = None,

        softmax_weights: Optional[Tensor] = None,

    ):

        # This is part of a workaround to get around fork/join parallelism

        # not supporting Optional types.

        local_attention_mask, expand_mask = key_padding_mask[0], key_padding_mask[1]

        if local_attention_mask is not None and local_attention_mask.dim() == 0:

            local_attention_mask = None



        q = q * self.scaling



        if pbc_expand_batched is not None and softmax_weights is None:

            outcell_index = pbc_expand_batched["outcell_index"]

            embed_dim = q.size()[-1]

            outcell_index = (

                outcell_index.unsqueeze(-1)

                .unsqueeze(-1)

                .expand(-1, -1, self.rank_num, embed_dim)

            )

            expand_k = torch.gather(k, dim=1, index=outcell_index)

            expand_v = torch.gather(v, dim=1, index=outcell_index)

            k = torch.cat([k, expand_k], dim=1)

            v = torch.cat([v, expand_v], dim=1)

        else:

            local_attention_mask = None



        bsz, tgt_len, src_len = q.shape[0], q.shape[1], k.shape[1]



        q = (

            q.reshape(bsz, tgt_len, self.rank_num, self.num_heads, self.head_dim)

            .transpose(2, 3)

            .reshape(bsz, tgt_len, self.num_heads, self.rank_num * self.head_dim)

            .transpose(1, 2)

        )

        k = (

            k.reshape(bsz, src_len, self.rank_num, self.num_heads, self.head_dim)

            .transpose(2, 3)

            .reshape(bsz, src_len, self.num_heads, self.rank_num * self.head_dim)

            .transpose(1, 2)

        )

        v = (

            v.reshape(bsz, src_len, self.rank_num, self.num_heads, self.head_dim)

            .transpose(2, 3)

            .reshape(bsz, src_len, self.num_heads, self.rank_num * self.head_dim)

            .transpose(1, 2)

        )



        attn_weights = q.matmul(

            k.transpose(-1, -2)

        )  # (bsz, num_heads, tgt_len, src_len)



        if attn_bias is not None and softmax_weights is None:

            attn_weights = attn_weights + attn_bias



        if local_attention_mask is not None:

            if softmax_weights is None:

                attn_weights = attn_weights.masked_fill(

                    (local_attention_mask <= 1e-6).unsqueeze(1), float("-inf")

                )

        if expand_mask is not None:

            attn_weights = attn_weights.masked_fill(

                expand_mask.unsqueeze(1).unsqueeze(1).to(torch.bool),

                float("-inf"),

            )



        # attn_probs_float = F.dropout(

        #     SmoothSoftmax(attn_weights, local_attention_mask, return_dtype=torch.float32),

        #     # F.softmax(attn_weights, dim=-1, dtype=torch.float32),

        #     self.dropout,

        #     self.training,

        # )

        attn_probs_float = F.dropout(

            F.softmax(attn_weights, dim=-1, dtype=torch.float32),

            self.dropout,

            training=self.training,

        )



        # if local_attention_mask is not None:

        #     attn_probs_float = attn_probs_float * local_attention_mask.unsqueeze(1)

        #     attn_probs_float = attn_probs_float / (attn_probs_float.sum(dim=-1, keepdim=True) + 1e-10)

        # elif softmax_weights is not None:

        #     attn_probs_float = attn_probs_float * softmax_weights

        #     attn_probs_float = attn_probs_float / (attn_probs_float.sum(dim=-1, keepdim=True) + 1e-10)



        # if local_attention_mask is not None:

        #     attn_probs_float = (

        #         torch.mul(attn_probs_float, local_attention_mask.unsqueeze(1))

        #         .float()

        #         .type_as(attn_probs_float)

        #     )

            # attn_probs_float = attn_probs_float / attn_probs_float.sum(dim=-1, keepdim=True)

        if local_attention_mask is not None:

            attn_probs_float=attn_probs_float.to(torch.float64)

            attn_probs_float = attn_probs_float * local_attention_mask.unsqueeze(1)

            temp_attn_probs_float = (

                torch.mul(attn_probs_float, local_attention_mask.unsqueeze(1))

                .float()

                .type_as(attn_probs_float)

            )

            attn_probs_float = temp_attn_probs_float / (attn_probs_float.sum(dim=-1, keepdim=True) + 1e-10)

        elif softmax_weights is not None:

            attn_probs_float=attn_probs_float.to(torch.float64)

            softmax_weights,weighted_softmax_weights = softmax_weights[0], softmax_weights[1]

            temp_attn_probs_float = attn_probs_float * weighted_softmax_weights

            attn_probs_float = attn_probs_float * softmax_weights

            attn_probs_float = temp_attn_probs_float / (attn_probs_float.sum(dim=-1, keepdim=True) + 1e-10)



        attn_probs = attn_probs_float.type_as(

            attn_weights

        )  # (bsz, num_heads, tgt_len, src_len)



        attn = attn_probs.matmul(v)  # (bsz, num_heads, tgt_len, 3 * head_dim)

        attn = (

            attn.transpose(1, 2)

            .reshape(bsz, tgt_len, self.num_heads, self.rank_num, self.head_dim)

            .transpose(2, 3)

            .reshape(bsz, tgt_len, self.rank_num, self.hidden_channels)

        )

        attn = self.attn_ln(attn)



        attn = self.out_proj(attn)



        return attn



class InvariantSelfAttention(nn.Module):

    def __init__(

        self,

        hidden_channels,

        head_dim,

        num_heads,

        dropout,

        d_tilde=1,

        norm_eps: float = 1e-5,

        use_linear_bias: bool = False,

    ):

        super().__init__()

        self.head_dim = head_dim

        self.num_heads = num_heads

        self.use_linear_bias = use_linear_bias

        self.q_proj = nn.Linear(hidden_channels, hidden_channels, bias=use_linear_bias)

        self.k_proj = nn.Linear(hidden_channels, hidden_channels, bias=use_linear_bias)

        self.v_proj = nn.Linear(hidden_channels, hidden_channels, bias=use_linear_bias)

        self.invariant_attention = InvariantAttention(

            hidden_channels, head_dim, dropout, norm_eps=norm_eps

        )



        self.reset_parameters(1)



    def reset_parameters(self, d_tilde):

        nn.init.xavier_uniform_(self.q_proj.weight, gain=1.0 / math.sqrt(d_tilde))

        nn.init.xavier_uniform_(self.k_proj.weight, gain=1.0 / math.sqrt(d_tilde))

        nn.init.xavier_uniform_(self.v_proj.weight, gain=1.0 / math.sqrt(d_tilde))

        if self.use_linear_bias:

            self.q_proj.bias.data.fill_(0)

            self.k_proj.bias.data.fill_(0)

            self.v_proj.bias.data.fill_(0)



    def forward(self, x,

                attn_bias,

                mask,

                pbc_expand_batched: Optional[Dict] = None,

                softmax_weights: Optional[Tensor] = None,):

        q = self.q_proj(x)

        k = self.k_proj(x)

        v = self.v_proj(x)



        attn = self.invariant_attention(q, k, v, attn_bias, mask, pbc_expand_batched,softmax_weights=softmax_weights,)



        return attn



class EquivariantSelfAttention(nn.Module):

    def __init__(

        self,

        hidden_channels,

        head_dim,

        num_heads,

        dropout,

        d_tilde=1,

        norm_eps: float = 1e-5,

        rank_num: int = 3,

        use_diff_liear_for_sph=False,

        sph_args=None,

    ):

        super().__init__()

        self.head_dim = head_dim

        self.num_heads = num_heads

        self.use_diff_liear_for_sph = use_diff_liear_for_sph

        self.sph_num,self.sph_offset = sph_args



        if self.use_diff_liear_for_sph:

            self.q_proj = SO3_LinearV2(

                hidden_channels, hidden_channels,lmax=self.sph_num-1,

            )

            self.k_proj = SO3_LinearV2(

                hidden_channels, hidden_channels,lmax=self.sph_num-1,

            )

            self.v_proj = SO3_LinearV2(

                hidden_channels, hidden_channels,lmax=self.sph_num-1,

            )



        else:

            self.q_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)

            self.k_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)

            self.v_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)

        self.equiariant_attention = EquivariantAttention(

            hidden_channels,

            head_dim,

            dropout,

            norm_eps=norm_eps,

            rank_num=rank_num,

            use_diff_liear_for_sph=use_diff_liear_for_sph,

            sph_args=sph_args

        )



        self.reset_parameters(1)



    def reset_parameters(self, d_tilde):

        if self.use_diff_liear_for_sph:

            pass

        else:

            nn.init.xavier_uniform_(self.q_proj.weight, gain=1.0 / math.sqrt(d_tilde))

            nn.init.xavier_uniform_(self.k_proj.weight, gain=1.0 / math.sqrt(d_tilde))

            nn.init.xavier_uniform_(self.v_proj.weight, gain=1.0 / math.sqrt(d_tilde))



    def forward(self,

                vec,

                attn_bias,

                mask,

                pbc_expand_batched: Optional[Dict] = None,

                softmax_weights: Optional[Tensor] = None,):

        q = self.q_proj(vec)

        k = self.k_proj(vec)

        v = self.v_proj(vec)



        attn = self.equiariant_attention(q, k, v, attn_bias, mask, pbc_expand_batched,softmax_weights=softmax_weights,)



        return attn



class Invariant2EquivariantAttention(nn.Module):

    def __init__(

        self,

        hidden_channels,

        head_dim,

        num_heads,

        dropout,

        d_tilde=1,

        norm_eps: float = 1e-5,

        rank_num: int = 3,

        use_linear_bias: bool = False,

        use_diff_liear_for_sph=False,

        sph_args=None,

    ):

        super().__init__()

        self.head_dim = head_dim

        self.num_heads = num_heads

        self.use_linear_bias = use_linear_bias

        self.q_proj = nn.Linear(hidden_channels, hidden_channels, bias=use_linear_bias)



        self.use_diff_liear_for_sph = use_diff_liear_for_sph

        self.sph_num, self.sph_offset = sph_args

        if self.use_diff_liear_for_sph:

            self.k1_proj = SO3_LinearV2(

                hidden_channels, hidden_channels,lmax=self.sph_num-1,

            )

            self.k2_proj = SO3_LinearV2(

                hidden_channels, hidden_channels,lmax=self.sph_num-1,

            )

            self.v1_proj = SO3_LinearV2(

                hidden_channels, hidden_channels,lmax=self.sph_num-1,

            )

            self.v2_proj = SO3_LinearV2(

                hidden_channels, hidden_channels,lmax=self.sph_num-1,

            )

        else:

            self.k1_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)

            self.k2_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)

            self.v1_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)

            self.v2_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)

        self.invariant_attention = InvariantAttention(

            hidden_channels, head_dim, dropout, norm_eps=norm_eps

        )

        self.rank_num = rank_num



        self.reset_parameters(1)



    def reset_parameters(self, d_tilde):

        nn.init.xavier_uniform_(self.q_proj.weight, gain=1.0 / math.sqrt(d_tilde))

        if self.use_linear_bias:

            self.q_proj.bias.data.fill_(0)



        if self.use_diff_liear_for_sph:

            pass

        else:

            nn.init.xavier_uniform_(self.k1_proj.weight, gain=1.0 / math.sqrt(d_tilde))

            nn.init.xavier_uniform_(self.k2_proj.weight, gain=1.0 / math.sqrt(d_tilde))

            nn.init.xavier_uniform_(self.v1_proj.weight, gain=1.0 / math.sqrt(d_tilde))

            nn.init.xavier_uniform_(self.v2_proj.weight, gain=1.0 / math.sqrt(d_tilde))



    def forward(

        self,

        x,

        vec,

        attn_bias,

        mask,

        pbc_expand_batched: Optional[Dict] = None,

        softmax_weights: Optional[Tensor] = None,

    ):

        q = self.q_proj(x)

       

        k1 = self.k1_proj(vec)

        k2 = self.k2_proj(vec)

        v1 = self.v1_proj(vec)

        v2 = self.v2_proj(vec)



        k = (k1 * k2).sum(dim=-2) * (

            self.rank_num**-0.5

        )  # (n_graph, n_node, feat_dim)



        v = (v1 * v2).sum(dim=-2) * (

            self.rank_num**-0.5

        )  # (n_graph, n_node, feat_dim)



        attn = self.invariant_attention(q, k, v, attn_bias, mask, pbc_expand_batched,softmax_weights=softmax_weights,)



        return attn



class Equivariant2InvariantAttention(nn.Module):

    def __init__(

        self,

        hidden_channels,

        head_dim,

        num_heads,

        dropout,

        eQi_choice,

        gbf_args,

        d_tilde=1,

        norm_eps: float = 1e-5,

        rank_num: int = 3,

        use_linear_bias: bool = False,

        use_diff_liear_for_sph=False,

        sph_args=None,

    ):

        super().__init__()

        self.head_dim = head_dim

        self.num_heads = num_heads

        self.eQi_choice = eQi_choice

        self.K = gbf_args[0]

        self.edge_types = gbf_args[1]

        self.use_linear_bias = use_linear_bias



        self.use_diff_liear_for_sph = use_diff_liear_for_sph

        self.sph_num, self.sph_offset = sph_args



        if self.use_diff_liear_for_sph:

            self.q_proj = SO3_LinearV2(

                hidden_channels, hidden_channels,lmax=self.sph_num-1,

            )

        else:

            self.q_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)

        self.k1_proj = nn.Linear(hidden_channels, hidden_channels, bias=use_linear_bias)

        self.v1_proj = nn.Linear(hidden_channels, hidden_channels, bias=use_linear_bias)

        if eQi_choice == "original":

            self.k2_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)

            self.v2_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)

        elif "gbf" in eQi_choice:

            self.gbf = GaussianLayer(self.K, self.edge_types)

            self.gbf_proj = nn.Linear(self.K, hidden_channels, bias=use_linear_bias)



        self.equiariant_attention = EquivariantAttention(

            hidden_channels,

            head_dim,

            dropout,

            norm_eps=norm_eps,

            rank_num=rank_num,

            use_diff_liear_for_sph=use_diff_liear_for_sph,

            sph_args=sph_args

        )

        self.rank_num = rank_num



        self.reset_parameters(1)



    def reset_parameters(self, d_tilde):

        if self.use_diff_liear_for_sph:

            pass

        else:

            nn.init.xavier_uniform_(self.q_proj.weight, gain=1.0 / math.sqrt(d_tilde))

        nn.init.xavier_uniform_(self.k1_proj.weight, gain=1.0 / math.sqrt(d_tilde))

        nn.init.xavier_uniform_(self.k2_proj.weight, gain=1.0 / math.sqrt(d_tilde))

        nn.init.xavier_uniform_(self.v1_proj.weight, gain=1.0 / math.sqrt(d_tilde))

        nn.init.xavier_uniform_(self.v2_proj.weight, gain=1.0 / math.sqrt(d_tilde))

        if self.use_linear_bias:

            self.k1_proj.bias.data.fill_(0)

            self.v1_proj.bias.data.fill_(0)



    def forward(

        self,

        x,

        vec,

        attn_bias,

        mask,

        pos_unit,

        gbf_args,

        pbc_expand_batched: Optional[Dict] = None,

        softmax_weights: Optional[Tensor] = None,

    ):

        q = self.q_proj(vec)



        k1 = self.k1_proj(x)

        v1 = self.v1_proj(x)



        pos_mean_centered_unit, pos_relative_unit = pos_unit

        if "gbf" in self.eQi_choice:

            dist, edge_type = gbf_args

            gbf_feature = self.gbf_proj(self.gbf(dist, edge_type))

            edge_feature = gbf_feature.masked_fill(

                mask.unsqueeze(1).unsqueeze(-1), 0.0

            )  # (n_graph, n_node, n_node, feat_dim)



        if self.eQi_choice == "original":

            k2 = self.k2_proj(vec)

            k = k1.unsqueeze(2) * k2  # (n_graph, n_node, 3, feat_dim)

            v2 = self.v2_proj(vec)

            v = v1.unsqueeze(2) * v2  # (n_graph, n_node, 3, feat_dim)



        elif self.eQi_choice == "meanCentered_vanilla":

            k = pos_mean_centered_unit.unsqueeze(-1) * k1.unsqueeze(

                -2

            )  # (n_graph, n_node, 3, feat_dim)

            v = pos_mean_centered_unit.unsqueeze(-1) * v1.unsqueeze(

                -2

            )  # (n_graph, n_node, 3, feat_dim)



        elif self.eQi_choice == "sumRelative_vanilla":

            k_edge = pos_relative_unit.unsqueeze(-1) * k1.unsqueeze(2).unsqueeze(

                -2

            )  # (n_graph, n_node, n_node, 3, feat_dim)

            k_edge = k_edge.masked_fill(

                mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1), 0.0

            )

            k = k_edge.sum(dim=2)  # (n_graph, n_node, 3, feat_dim)

            k = k.masked_fill(mask.unsqueeze(-1).unsqueeze(-1), 0.0)

            v_edge = pos_relative_unit.unsqueeze(-1) * v1.unsqueeze(2).unsqueeze(

                -2

            )  # (n_graph, n_node, n_node, 3, feat_dim)

            v_edge = v_edge.masked_fill(

                mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1), 0.0

            )

            v = v_edge.sum(dim=2)  # (n_graph, n_node, 3, feat_dim)

            v = v.masked_fill(mask.unsqueeze(-1).unsqueeze(-1), 0.0)



        elif self.eQi_choice == "meanCentered_gbf":

            gbf_sum = pos_mean_centered_unit.unsqueeze(-1) * edge_feature.sum(

                dim=2

            ).unsqueeze(

                -2

            )  # (n_graph, n_node, 3, feat_dim)

            gbf_sum = gbf_sum.masked_fill(mask.unsqueeze(-1).unsqueeze(-1), 0.0)

            k = k1.unsqueeze(-2) * gbf_sum  # (n_graph, n_node, 3, feat_dim)

            v = v1.unsqueeze(-2) * gbf_sum  # (n_graph, n_node, 3, feat_dim)



        elif self.eQi_choice == "sumRelative_gbf":

            feat_edge = pos_relative_unit.unsqueeze(-1) * edge_feature.unsqueeze(

                -2

            )  # (n_graph, n_node, n_node, 3, feat_dim)

            feat_edge = feat_edge.masked_fill(

                mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1), 0.0

            )

            feat_sum = feat_edge.sum(dim=2)  # (n_graph, n_node, 3, feat_dim)

            feat_sum = feat_sum.masked_fill(mask.unsqueeze(-1).unsqueeze(-1), 0.0)

            k = k1.unsqueeze(-2) * feat_sum  # (n_graph, n_node, 3, feat_dim)

            v = v1.unsqueeze(-2) * feat_sum  # (n_graph, n_node, 3, feat_dim)



        attn = self.equiariant_attention(q, k, v, attn_bias, mask, pbc_expand_batched,softmax_weights=softmax_weights,)



        return attn



def gelu(x: torch.Tensor) -> torch.Tensor:

    return torch.nn.functional.gelu(x.float()).type_as(x)



class EncoderLayer(nn.Module):

    def __init__(

        self,

        hidden_channels,

        num_heads,

        activation_fn,

        ffn_embedding_dim,

        dropout: float = 0.1,

        attention_dropout: float = 0.1,

        activation_dropout: float = 0.1,

        eQi_choice: str = "original",

        gbf_args=None,

        layer_index=0,

        d_tilde=1,

        norm_eps: float = 1e-10,

        use_linear_bias: bool = False,

        rank_num: int = 3,

        use_diff_liear_for_sph: bool = False,

        sph_args=None,

    ):

        super().__init__()



        head_dim = hidden_channels // num_heads



        self.layer_index = layer_index



        self.use_linear_bias = use_linear_bias



        self.use_diff_liear_for_sph = use_diff_liear_for_sph

        self.sph_num, self.sph_offset = sph_args



        if self.layer_index % 2 == 0:

            self.invariant_self_attention = InvariantSelfAttention(

                hidden_channels,

                head_dim,

                num_heads,

                dropout=attention_dropout,

                norm_eps=norm_eps,

                use_linear_bias=use_linear_bias,

            )

            self.equivariant_self_attention = EquivariantSelfAttention(

                hidden_channels,

                head_dim,

                num_heads,

                dropout=attention_dropout,

                norm_eps=norm_eps,

                rank_num=rank_num,

                use_diff_liear_for_sph=use_diff_liear_for_sph,

                sph_args=sph_args,

            )

        else:

            self.invariant2equivariant_attention = Invariant2EquivariantAttention(

                hidden_channels,

                head_dim,

                num_heads,

                dropout=attention_dropout,

                norm_eps=norm_eps,

                rank_num=rank_num,

                use_linear_bias=use_linear_bias,

                use_diff_liear_for_sph=use_diff_liear_for_sph,

                sph_args=sph_args,

            )

            self.equivaiant2invariant_attention = Equivariant2InvariantAttention(

                hidden_channels,

                head_dim,

                num_heads,

                dropout=attention_dropout,

                eQi_choice=eQi_choice,

                gbf_args=gbf_args,

                norm_eps=norm_eps,

                rank_num=rank_num,

                use_linear_bias=use_linear_bias,

                use_diff_liear_for_sph=use_diff_liear_for_sph,

                sph_args=sph_args,

            )



        self.invariant_attn_layer_norm = nn.LayerNorm(hidden_channels, eps=norm_eps)

        self.equivariant_attn_layer_norm = EquivariantLayerNorm(

            hidden_channels, eps=norm_eps, use_diff_liear_for_sph=use_diff_liear_for_sph, sph_args=sph_args

            )



        self.activation_fn = nn.SiLU() if activation_fn == "silu" else gelu

        self.invariant_fc1 = nn.Linear(

            hidden_channels, ffn_embedding_dim, bias=use_linear_bias

        )

        self.invariant_fc2 = nn.Linear(

            ffn_embedding_dim, hidden_channels, bias=use_linear_bias

        )

        self.equivariant_fc1 = nn.Linear(

            hidden_channels, ffn_embedding_dim, bias=use_linear_bias

        )

        if self.use_diff_liear_for_sph:

            self.equivariant_fc2 = SO3_LinearV2(

                hidden_channels, ffn_embedding_dim,lmax=self.sph_num-1,

            )

            self.equivariant_fc3 = SO3_LinearV2(

                ffn_embedding_dim, hidden_channels,lmax=self.sph_num-1,

            )



        else:

            self.equivariant_fc2 = nn.Linear(hidden_channels, ffn_embedding_dim, bias=False)

            self.equivariant_fc3 = nn.Linear(ffn_embedding_dim, hidden_channels, bias=False)



        self.invariant_ffn_layer_norm = nn.LayerNorm(hidden_channels, eps=norm_eps)

        self.equivariant_ffn_layer_norm = EquivariantLayerNorm(

            hidden_channels, eps=norm_eps, use_diff_liear_for_sph=use_diff_liear_for_sph, sph_args=sph_args

        )



        self.invariant_ffn_layer_norm_2 = nn.LayerNorm(ffn_embedding_dim, eps=norm_eps)

        self.equivariant_ffn_layer_norm_2 = EquivariantLayerNorm(

            ffn_embedding_dim, eps=norm_eps, use_diff_liear_for_sph=use_diff_liear_for_sph, sph_args=sph_args

        )



        self.dropout = dropout

        self.activation_dropout = activation_dropout

        self.attention_dropout = attention_dropout



        self.reset_parameters()



    def reset_parameters(self, d_tilde=1.0):

        self.invariant_attn_layer_norm.reset_parameters()

        self.equivariant_attn_layer_norm.reset_parameters()



        nn.init.xavier_uniform_(

            self.invariant_fc1.weight, gain=1.0 / math.sqrt(d_tilde)

        )

        nn.init.xavier_uniform_(

            self.invariant_fc2.weight, gain=1.0 / math.sqrt(d_tilde)

        )

        nn.init.xavier_uniform_(

            self.equivariant_fc1.weight, gain=1.0 / math.sqrt(d_tilde)

        )



        if self.use_diff_liear_for_sph:

            pass

        else:

            nn.init.xavier_uniform_(

                self.equivariant_fc2.weight, gain=1.0 / math.sqrt(d_tilde)

            )

            nn.init.xavier_uniform_(

                self.equivariant_fc3.weight, gain=1.0 / math.sqrt(d_tilde)

            )



        self.invariant_ffn_layer_norm.reset_parameters()

        self.equivariant_ffn_layer_norm.reset_parameters()

        self.invariant_ffn_layer_norm_2.reset_parameters()

        self.equivariant_ffn_layer_norm_2.reset_parameters()



        if self.use_linear_bias:

            self.invariant_fc1.bias.data.fill_(0)

            self.invariant_fc2.bias.data.fill_(0)

            self.equivariant_fc1.bias.data.fill_(0)



    def forward(

        self,

        x,

        vec,

        attn_bias_iself,

        attn_bias_i2e,

        attn_bias_eself,

        attn_bias_e2i,

        mask,

        pos_unit,

        gbf_args,

        pbc_expand_batched: Optional[Dict] = None,

        softmax_weights: Optional[Tensor] = None,

    ):

        # attetion

        dx = self.invariant_attn_layer_norm(x)

        dvec = self.equivariant_attn_layer_norm(vec)



        if self.layer_index % 2 == 0:

            dx_invariant = self.invariant_self_attention(

                dx, attn_bias_iself, mask, pbc_expand_batched,softmax_weights=softmax_weights,

            )

            dx_invariant = F.dropout(

                dx_invariant, p=self.dropout, training=self.training

            )

            dvec_equivariant = self.equivariant_self_attention(

                dvec, attn_bias_eself, mask, pbc_expand_batched,softmax_weights=softmax_weights,

            )

            x = x + dx_invariant

            vec = vec + dvec_equivariant

        else:

            dx_equivariant = self.invariant2equivariant_attention(

                dx, dvec, attn_bias_i2e, mask, pbc_expand_batched,softmax_weights=softmax_weights,

            )

            dx_equivariant = F.dropout(

                dx_equivariant, p=self.dropout, training=self.training

            )

            dvec_invariant = self.equivaiant2invariant_attention(

                dx, dvec, attn_bias_e2i, mask, pos_unit, gbf_args, pbc_expand_batched,softmax_weights=softmax_weights,

            )



            x = x + dx_equivariant

            vec = vec + dvec_invariant



        # FFN

        dx = self.invariant_ffn_layer_norm(x)

        dvec = self.equivariant_ffn_layer_norm(vec)



        dx_ffn = self.activation_fn(self.invariant_fc1(dx))

        dx_ffn = F.dropout(dx_ffn, p=self.activation_dropout, training=self.training)

        dx_ffn = self.invariant_ffn_layer_norm_2(dx_ffn)

        dx_ffn = self.invariant_fc2(dx_ffn)

        dx_ffn = F.dropout(dx_ffn, p=self.dropout, training=self.training)

        dvec_ffn = self.activation_fn(self.equivariant_fc1(dx)).unsqueeze(

            -2

        )



        dvec_ffn = dvec_ffn * self.equivariant_fc2(dvec)

        dvec_ffn = self.equivariant_ffn_layer_norm_2(dvec_ffn)

        dvec_ffn = self.equivariant_fc3(dvec_ffn)

       



        # add & norm

        x = x + dx_ffn

        vec = vec + dvec_ffn



        return x, vec



class UnifiedDecoder(nn.Module):

    def __init__(

        self,

        args,

        num_pred_attn_layer: int,

        embedding_dim: int,

        num_attention_heads: int,

        ffn_embedding_dim: int,

        dropout: float,

        attention_dropout: float,

        activation_dropout: float,

        num_3d_bias_kernel: int,

        num_edges: int,

        num_atoms: int,

        pbc_multigraph_cutoff: float,

        use_linear_bias: bool = False,

        rank_num: int = 4,

        use_diff_liear_for_sph: bool = False,

        sph_args=None,

        activation_fn: str = "gelu",

    ):

        super().__init__()

        self.args = args

        self.pbc_multigraph_cutoff = pbc_multigraph_cutoff

        self.unified_encoder_layers = nn.ModuleList()



        sph_num,sph_offset = sph_args

        for _ in range(num_pred_attn_layer):

            layer = EncoderLayer(

                embedding_dim,

                num_attention_heads,

                activation_fn,

                ffn_embedding_dim,

                dropout=dropout,

                attention_dropout=attention_dropout,

                activation_dropout=activation_dropout,

                eQi_choice="original",

                gbf_args=[num_3d_bias_kernel, num_edges],

                layer_index=_,

                use_linear_bias=use_linear_bias,

                rank_num=rank_num,

                use_diff_liear_for_sph=use_diff_liear_for_sph,

                sph_args=[sph_num,sph_offset]

            )

            self.unified_encoder_layers.append(layer)



        self.unified_gbf_attn_bias = GaussianLayer(num_3d_bias_kernel, num_edges)



        self.unified_gbf_pos = NodeGaussianLayer(num_3d_bias_kernel, num_atoms)



        # self.unified_bias_proj = nn.Linear(num_3d_bias_kernel, num_attention_heads)



        if use_diff_liear_for_sph:

            self.unified_vec_proj=SO3_LinearV2(

                num_3d_bias_kernel, embedding_dim,lmax=sph_num-1,method=2,sph_args=sph_args

            )

        else:

            self.unified_vec_proj = nn.Linear(num_3d_bias_kernel, embedding_dim, bias=False)



        self.unified_final_equivariant_ln = EquivariantLayerNorm(embedding_dim, use_diff_liear_for_sph=use_diff_liear_for_sph, sph_args=sph_args)



        self.unified_final_invariant_ln = nn.LayerNorm(embedding_dim)



        self.unified_final_feature_ln = nn.LayerNorm(embedding_dim)



        self.unified_output_layer = nn.Linear(embedding_dim, 1, bias=False)



    def forward(self, batched_data, x, pos, padding_mask, mask_pos=None,softmax_weights=None):

        node_type_edge, node_type = (

            # batched_data["pos"],

            batched_data["node_type_edge"],

            batched_data["x"][:, :, 0],

        )

        node_mask = batched_data["node_mask"]

        n_node = pos.shape[1]



        uni_delta_pos = pos.unsqueeze(1) - pos.unsqueeze(2)

        dist = uni_delta_pos.norm(dim=-1).view(-1, n_node, n_node)

        uni_delta_pos /= dist.unsqueeze(-1) + 1e-5



        # r_i/||r_i|| * gbf(||r_i||)

        pos_norm = pos.norm(dim=-1)

        gbf_node_type_mask = node_mask.squeeze(-1)

        node_type = node_type.masked_fill(gbf_node_type_mask.bool(), 0).to(node_type)



        uni_gbf_pos_feature = self.unified_gbf_pos(pos_norm, node_type.unsqueeze(-1))

        uni_pos_feature = uni_gbf_pos_feature.masked_fill(

            padding_mask[:, 1:].unsqueeze(-1), 0.0

        )

        uni_vec_value = self.unified_vec_proj(uni_pos_feature).unsqueeze(-2)

        vec = pos.unsqueeze(-1) * uni_vec_value



        vec = vec.masked_fill(padding_mask[:, 1:].unsqueeze(-1).unsqueeze(-1), 0.0)

        pos_mean_centered_dist = pos.norm(dim=-1)

        pos_mean_centered_unit = pos / (pos_mean_centered_dist.unsqueeze(-1) + 1e-5)



        # attn_bias

        node_mask_i = node_mask.unsqueeze(-2).repeat(1, 1, n_node, 1)

        node_mask_j = node_mask.unsqueeze(1).repeat(1, n_node, 1, 1)

        new_node_mask = torch.cat([node_mask_i, node_mask_j], dim=-1).bool()

        node_type_edge = node_type_edge.masked_fill(new_node_mask, 0).to(node_type_edge)



        uni_gbf_feature = self.unified_gbf_attn_bias(dist, node_type_edge)



        uni_graph_attn_bias = (

            self.unified_bias_proj(uni_gbf_feature).permute(0, 3, 1, 2).contiguous()

        )

        uni_graph_attn_bias = uni_graph_attn_bias.masked_fill(

            padding_mask[:, 1:].unsqueeze(1).unsqueeze(2), float("-inf")

        )



        output = x.contiguous().transpose(0, 1)[:, 1:, :]

        output = output.masked_fill(padding_mask[:, 1:].unsqueeze(-1), 0.0)

        for layer in self.unified_encoder_layers:

            output, vec = layer(

                output,

                vec,

                uni_graph_attn_bias,

                uni_graph_attn_bias,

                uni_graph_attn_bias,

                uni_graph_attn_bias,

                padding_mask[:, 1:],

                [pos_mean_centered_unit, uni_delta_pos],

                [dist, node_type_edge],

                softmax_weights=softmax_weights,

            )



        node_output = self.unified_final_equivariant_ln(vec)

        output = self.unified_final_invariant_ln(output)



        node_output = self.unified_output_layer(node_output).squeeze(-1)

        node_output = node_output.masked_fill(padding_mask[:, 1:].unsqueeze(-1), 0.0)



        return node_output

