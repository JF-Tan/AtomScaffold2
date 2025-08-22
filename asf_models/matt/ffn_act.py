import torch
import torch.nn as nn
from .norms import XixianEquivariantLayerNorm

class GateActivation(torch.nn.Module):
    def __init__(self, lmax: int, mmax: int, num_channels: int) -> None:
        super().__init__()

        self.lmax = lmax
        self.mmax = mmax
        self.num_channels = num_channels

        # compute `expand_index` based on `lmax` and `mmax`
        num_components = 0
        for lval in range(1, self.lmax + 1):
            num_m_components = min((2 * lval + 1), (2 * self.mmax + 1))
            num_components = num_components + num_m_components
        expand_index = torch.zeros([num_components]).long()
        start_idx = 0
        for lval in range(1, self.lmax + 1):
            length = min((2 * lval + 1), (2 * self.mmax + 1))
            expand_index[start_idx : (start_idx + length)] = lval - 1
            start_idx = start_idx + length
        self.register_buffer("expand_index", expand_index)

        self.scalar_act = (
            torch.nn.SiLU()
        )  # SwiGLU(self.num_channels, self.num_channels)  # #
        self.gate_act = torch.nn.Sigmoid()  # torch.nn.SiLU() # #

    def forward(self, gating_scalars, input_tensors):
        """
        `gating_scalars`: shape [N, lmax * num_channels]
        `input_tensors`: shape  [N, (lmax + 1) ** 2, num_channels]
        """

        gating_scalars = self.gate_act(gating_scalars)
        gating_scalars = gating_scalars.reshape(
            gating_scalars.shape[0], self.lmax, self.num_channels
        )
        gating_scalars = torch.index_select(
            gating_scalars, dim=1, index=self.expand_index
        )

        input_tensors_scalars = input_tensors.narrow(1, 0, 1)
        input_tensors_scalars = self.scalar_act(input_tensors_scalars)

        input_tensors_vectors = input_tensors.narrow(1, 1, input_tensors.shape[1] - 1)
        input_tensors_vectors = input_tensors_vectors * gating_scalars

        return torch.cat((input_tensors_scalars, input_tensors_vectors), dim=1)


class S2Activation(torch.nn.Module):
    """
    Assume we only have one resolution
    """

    def __init__(self, lmax: int, mmax: int, SO3_grid) -> None:
        super().__init__()
        self.lmax = lmax
        self.mmax = mmax
        self.act = torch.nn.SiLU()
        self.SO3_grid = SO3_grid

    def forward(self, inputs):
        to_grid_mat = self.SO3_grid["lmax_mmax"].get_to_grid_mat()
        from_grid_mat = self.SO3_grid["lmax_mmax"].get_from_grid_mat()
        x_grid = torch.einsum("bai, zic -> zbac", to_grid_mat, inputs)
        x_grid = self.act(x_grid)
        return torch.einsum("bai, zbac -> zic", from_grid_mat, x_grid)


class SeparableS2Activation(torch.nn.Module):
    def __init__(self, lmax: int, mmax: int, SO3_grid) -> None:
        super().__init__()
        self.lmax = lmax
        self.mmax = mmax
        self.scalar_act = torch.nn.SiLU()
        self.s2_act = S2Activation(self.lmax, self.mmax, SO3_grid)

    def forward(self, input_scalars, input_tensors):
        output_scalars = self.scalar_act(input_scalars)
        output_scalars = output_scalars.reshape(
            output_scalars.shape[0], 1, output_scalars.shape[-1]
        )
        output_tensors = self.s2_act(input_tensors)
        return torch.cat(
            (
                output_scalars,
                output_tensors.narrow(1, 1, output_tensors.shape[1] - 1),
            ),
            dim=1,
        )


import math
import e3nn.o3 as o3
import torch


class SO3_Linear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, lmax: int) -> None:
        """
        1. Use `torch.einsum` to prevent slicing and concatenation
        2. Need to specify some behaviors in `no_weight_decay` and weight initialization.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lmax = lmax

        self.weight = torch.nn.Parameter(
            torch.randn((self.lmax + 1), out_features, in_features)
        )
        bound = 1 / math.sqrt(self.in_features)
        torch.nn.init.uniform_(self.weight, -bound, bound)
        self.bias = torch.nn.Parameter(torch.zeros(out_features))

        expand_index = torch.zeros([(lmax + 1) ** 2]).long()
        for lval in range(lmax + 1):
            start_idx = lval**2
            length = 2 * lval + 1
            expand_index[start_idx : (start_idx + length)] = lval
        self.register_buffer("expand_index", expand_index)

    def forward(self, input_embedding):
        weight = torch.index_select(
            self.weight, dim=0, index=self.expand_index
        )  # [(L_max + 1) ** 2, C_out, C_in]
        out = torch.einsum(
            "bmi, moi -> bmo", input_embedding, weight
        )  # [N, (L_max + 1) ** 2, C_out]
        bias = self.bias.view(1, 1, self.out_features)
        out[:, 0:1, :] = out.narrow(1, 0, 1) + bias
        return out

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(in_features={self.in_features}, out_features={self.out_features}, lmax={self.lmax})"


class SO3_Linear_method_version(torch.nn.Module):
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

class InvFFN(nn.Module):
    def __init__(self, 
                 sphere_channels: int,
                  hidden_channels: int,
                  hidden_dropout_prob=0.0):
        super().__init__()
        self.linear_1 = nn.Linear(sphere_channels, hidden_channels)
        self.linear_2 = nn.Linear(hidden_channels, sphere_channels)
        self.silu = nn.SiLU()
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.silu(x)
        x = self.linear_2(x)
        x = self.dropout(x)
        return x

class XixianEquFFN(nn.Module):
    def __init__(self,embedding_ch: int,hidden_channels: int,lmax: int,):
        super().__init__()
        self.inv_fc1 = nn.Linear(embedding_ch,hidden_channels,lmax)
        self.equ_fc2 = SO3_Linear(embedding_ch,hidden_channels,lmax)
        self.equ_fc3 = SO3_Linear(hidden_channels,embedding_ch,lmax)

        self.activation_fn = nn.SiLU()

        sph_irrps = (o3.Irreps.spherical_harmonics(lmax)  ).regroup()
        self.edge_irreps = o3.Irreps(sph_irrps)
        self.sph_num = len(self.edge_irreps)
        self.sph_offset = [0]
        for i in range(self.sph_num):
            self.sph_offset.append((i+1)**2)
        self.equ_norm =  XixianEquivariantLayerNorm(embedding_ch, eps=1e-7, use_diff_liear_for_sph=True, sph_args=[self.sph_num,self.sph_offset])
    def forward(self,inv_feat,equ_feat):
        dvec_ffn = self.activation_fn(self.inv_fc1(inv_feat)).unsqueeze(-2) 

        dvec_ffn = dvec_ffn * self.equ_fc2(equ_feat)
        dvec_ffn = self.equ_norm(dvec_ffn)
        dvec_ffn = self.equ_fc3(dvec_ffn)
        return dvec_ffn







class SpectralAtomwise(torch.nn.Module):
    def __init__(
        self,
        sphere_channels: int,
        hidden_channels: int,
        lmax: int,
        mmax: int,
    ):
        super().__init__()
        self.sphere_channels = sphere_channels
        self.hidden_channels = hidden_channels
        self.lmax = lmax
        self.mmax = mmax

        self.scalar_mlp = nn.Sequential(
            nn.Linear(
                self.sphere_channels,
                self.lmax * self.hidden_channels,
                bias=True,
            ),
            nn.SiLU(),
        )

        self.so3_linear_1 = SO3_Linear(
            self.sphere_channels, self.hidden_channels, lmax=self.lmax
        )
        self.act = GateActivation(
            lmax=self.lmax, mmax=self.lmax, num_channels=self.hidden_channels
        )
        self.so3_linear_2 = SO3_Linear(
            self.hidden_channels, self.sphere_channels, lmax=self.lmax
        )

    def forward(self, x):
        gating_scalars = self.scalar_mlp(x.narrow(1, 0, 1))
        x = self.so3_linear_1(x)
        x = self.act(gating_scalars, x)
        return self.so3_linear_2(x)