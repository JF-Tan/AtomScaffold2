
import torch
import torch.nn as nn
from torch_scatter import scatter_softmax, scatter_add

from typing import Optional

from torch_scatter import scatter_sum, scatter_max
from torch_scatter.utils import broadcast


def smooth_scatter_softmax(src: torch.Tensor, index: torch.Tensor,envelope: torch.Tensor,
                    dim: int = -1,
                    dim_size: Optional[int] = None) -> torch.Tensor:
    if not torch.is_floating_point(src):
        raise ValueError('`scatter_softmax` can only be computed over tensors '
                         'with floating point data types.')

    index = broadcast(index, src, dim)

    max_value_per_index = scatter_max(
        src, index, dim=dim, dim_size=dim_size)[0]
    max_per_src_element = max_value_per_index.gather(dim, index)

    recentered_scores = src - max_per_src_element
    recentered_scores_exp = recentered_scores.exp_() * envelope.unsqueeze(0)

    sum_per_index = scatter_sum(
        recentered_scores_exp, index, dim, dim_size=dim_size)
    normalizing_constants = sum_per_index.gather(dim, index)

    return recentered_scores_exp.div(normalizing_constants+1e-16)


@torch.jit.script
def SmoothSoftmax(input,envelope, return_dtype: Optional[torch.dtype] = None,eps: float = 1e-16,):
    # input = input.to(torch.float64)
    # Compute e_ij
    max_value = input.max(dim=-1, keepdim=True).values
    envelope = envelope.to(input.dtype)

    max_value = torch.where(max_value == float("-inf"), torch.zeros_like(max_value), max_value)

    input = input - max_value
    e_ij = torch.exp(input) * (envelope.unsqueeze(0)) #[h,n,e]

    # Compute softmax along the last dimension
    softmax = e_ij / (torch.sum(e_ij, dim=-1, keepdim=True) + eps)

    if return_dtype is not None:
        softmax = softmax.to(return_dtype)

    return softmax


class InvAttentionGATv2(nn.Module):
    def __init__(self, in_channels, h_channel, n_heads ):
        super().__init__()
        assert h_channel % n_heads == 0, "out_channels must be divisible by n_heads"
        self.n_heads = n_heads
        self.head_dim = h_channel // n_heads

        # Single linear to produce both "left" and "right" projections
        self.linear_l = nn.Linear(in_channels, h_channel, bias=False)
        self.linear_r = nn.Linear(in_channels, h_channel, bias=False)

        self.attn = nn.Linear(h_channel, 1, bias=False)
        self.activation = nn.SiLU()

    def forward(self, q,k,v,envelope,edge_index):
        # x: [N, in_channels], edge_index: [2, E]
        src, dst = edge_index  # [n_edges], [n_edges]
        N, E = q.size(0),  edge_index.size(1)
        


        g_l = self.linear_l(q) 
        g_r = self.linear_r(q) 


        # Gather per-edge features
        g_l_src = g_l[src]  # [n_edges, n_heads, n_hidden]
        g_r_dst = g_r[dst]  # [n_edges, n_heads, n_hidden]

        # Compute attention logits
        g_sum = g_l_src + g_r_dst # [n_edges, n_heads, n_hidden]
        e = self.attn(self.activation(g_sum)).squeeze(-1)  # [n_edges, n_heads]

        
        # Compute normalized attention coefficients per target node
        e += torch.log(envelope + 1e-7 )
        attn = scatter_softmax(e, dst, dim=0)  # [n_edges, n_heads]

        # Message passing
        out = attn.unsqueeze(-1) * g_l_src  # [n_edges, n_heads, n_hidden]
        out = scatter_add(out, dst, dim=0, dim_size=N)  # [n_nodes, n_heads, n_hidden

        # Aggregate back to nodes: [E*H, head_dim] -> [N*H, head_dim]
        out = out.view(N, -1)

        # Reshape to [N, out_channels]
        return out



import torch 
import torch.nn as nn
import math
from .ffn_act import SO3_Linear
from .norms import get_normalization_layer,XixianEquivariantLayerNorm

class EquAttentionGATv2(nn.Module):
    def __init__(self,in_channels, h_channel, n_heads ,lmax):
        super().__init__()
        assert h_channel % n_heads == 0, "out_channels must be divisible by n_heads"
        self.n_heads = n_heads
        self.head_dim = h_channel // n_heads

        # Single linear to produce both "left" and "right" projections
        self.linear_l = SO3_Linear(in_channels,h_channel,lmax)
        self.linear_r = SO3_Linear(in_channels,h_channel,lmax)

        self.attn = nn.Linear(h_channel, 1, bias=False)
        self.activation = nn.SiLU()

    def forward(self, q,k,v,envelope,edge_index):
        # x: [N, S, in_channels], edge_index: [2, E]
        src, dst = edge_index  # [n_edges], [n_edges]
        N,S, E = q.size(0),q.size(1),edge_index.size(1)


        g_l = self.linear_l(q) 
        g_r = self.linear_r(q) 


        # Gather per-edge features
        g_l_src = g_l[src]  # [n_edges,S, n_hidden]
        g_r_dst = g_r[dst]  # [n_edges,S, n_hidden]

        # Compute attention logits
        g_sum = g_l_src + g_r_dst # [n_edges, S, n_hidden]
        e = self.attn(self.activation(g_sum)).squeeze(-1)  # [n_edges, S]

        
        # Compute normalized attention coefficients per target node
        e += torch.log(envelope.unsqueeze(-1) + 1e-7 )
        attn = scatter_softmax(e, dst, dim=0)  # [n_edges, S]

        # Message passing
        out = attn.unsqueeze(-1) * g_l_src  # [n_edges, S, n_hidden]
        out = scatter_add(out, dst, dim=0, dim_size=N)  # [n_nodes, S, n_hidden

        # Aggregate back to nodes: [E*H, head_dim] -> [N*H, head_dim]
        out = out.view(N,S, -1)

        # Reshape to [N, out_channels]
        return out



import torch 
import torch.nn as nn
import math
import e3nn.o3 as o3

class InvAttention(nn.Module):
    def __init__(self,in_channels, h_channel, n_heads,ca = False ):
        super().__init__()
        self.ca = ca
        self.n_heads = n_heads
        self.d = h_channel // n_heads
        if ca == False:
            self.linear_q = nn.Linear(in_channels,h_channel)
            self.linear_k = nn.Linear(in_channels,h_channel)
            self.linear_v = nn.Linear(in_channels,h_channel)
        self.attn_ln = nn.LayerNorm(h_channel, eps=1e-7)
        self.final_proj = nn.Linear(h_channel,in_channels)

    def forward(self,q,k,v,envelope,attn_bias,expand_dict):
        '''
        q/k/v [n,D]
        '''
        atom_index = expand_dict['atom_index']
        batch_index = expand_dict['batch_index']
        edge_map_tab = expand_dict['edge_map_tab']
        N,E = q.shape[0], atom_index.shape[0]
        if self.ca == False:
            q = self.linear_q(q)
            k = self.linear_k(k)
            v = self.linear_v(v)

        # expand egdes
        k = k[atom_index] #[e,D]
        v = v[atom_index] #[e,D]

        # reshape
        q = q.view(N,self.n_heads,self.d).permute(1,0,2) # [N,H,D] -> [H,N,D]
        k = k.view(E,self.n_heads,self.d).permute(1,2,0) # [E,H,D] -> [H,D,E]
        v = v.view(E,self.n_heads,self.d).permute(1,0,2) # [E,H,D] -> [H,E,D]

        attention_weights = torch.matmul(q* math.sqrt(self.d) / self.d, k)  #[h,n,e]
        attention_weights += attn_bias[: , edge_map_tab]  #[h,n,e]
        attention_weights = smooth_scatter_softmax(attention_weights,batch_index,envelope[ edge_map_tab])
        attention_weights = attention_weights * envelope[ edge_map_tab].unsqueeze(0)


        out = torch.matmul(attention_weights,v).permute(1,0,2).contiguous().view(N,-1)
        out = self.attn_ln(out)
        out = self.final_proj(out)
        
        return out



class EquAttention(nn.Module):
    def __init__(self,in_channels, h_channel, n_heads ,lmax,ca =False):
        super().__init__()
        self.ca = ca
        self.n_heads = n_heads
        self.d = h_channel // n_heads

        if self.ca == False:
            self.linear_q = SO3_Linear(in_channels,h_channel,lmax)
            self.linear_k = SO3_Linear(in_channels,h_channel,lmax)
            self.linear_v = SO3_Linear(in_channels,h_channel,lmax)
        # self.attn_ln = nn.LayerNorm(h_channel, eps=1e-7)

        sph_irrps = (o3.Irreps.spherical_harmonics(lmax)  ).regroup()
        self.edge_irreps = o3.Irreps(sph_irrps)
        self.sph_num = len(self.edge_irreps)
        self.sph_offset = [0]
        for i in range(self.sph_num):
            self.sph_offset.append((i+1)**2)

        self.attn_ln = XixianEquivariantLayerNorm(h_channel, eps=1e-7, use_diff_liear_for_sph=True, sph_args=[self.sph_num,self.sph_offset])
        self.final_proj = SO3_Linear(h_channel,in_channels,lmax)

    def forward(self,q,k,v,envelope,attn_bias, expand_dict):
        '''
        q/k/v [n,S,D]
        '''
        atom_index = expand_dict['atom_index']
        batch_index = expand_dict['batch_index']
        edge_map_tab = expand_dict['edge_map_tab']
        N,S,E = q.shape[0], q.shape[1], atom_index.shape[0]
        if self.ca == False:
            q = self.linear_q(q)
            k = self.linear_k(k)
            v = self.linear_v(v)

        # expand egdes
        k = k[atom_index] #[e,S,D]
        v = v[atom_index] #[e,S,D]

        # reshape
        q = q.view(N,S,self.n_heads,self.d).transpose(1,2).contiguous().view(N,self.n_heads,-1).transpose(0,1)
        k = k.view(E,S,self.n_heads,self.d).transpose(1,2).contiguous().view(E,self.n_heads,-1).transpose(0,1)
        v = v.view(E,S,self.n_heads,self.d).transpose(1,2).contiguous().view(E,self.n_heads,-1).transpose(0,1)



        attention_weights = torch.matmul(q* math.sqrt(self.d /3) / self.d, k.transpose(-1, -2)  )
        attention_weights += attn_bias[: , edge_map_tab]
        attention_weights = smooth_scatter_softmax(attention_weights,batch_index,envelope[ edge_map_tab])
        attention_weights = attention_weights * envelope[ edge_map_tab].unsqueeze(0)

        out = torch.matmul(attention_weights,v).transpose(0,1).reshape(N,self.n_heads,S,-1).transpose(1,2).reshape(N,S,-1).contiguous()
        out = self.attn_ln(out.unsqueeze(0)).squeeze(0)
        out = self.final_proj(out)

        return out