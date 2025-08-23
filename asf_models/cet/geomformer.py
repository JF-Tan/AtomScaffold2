import math
from typing import Dict, Optional, Tuple
from ase import Atom
import e3nn
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from engine.logging import logger
from models.geomformer.config import GraphormerConfig
from models.geomformer.pbe_bias import PBE_bias
from models.geomformer.pbc2 import CellExpander
from models.geomformer.blocks import (
EncoderLayer,
UnifiedDecoder,
)

from models.base_model import ModelOutput

# from sfm.pipeline.accelerator.trainer import Model
# from sfm.models.graphormer.modules.task_heads import BandGapHead

@torch.jit.script
def _spherical_harmonics(lmax: int, x: torch.Tensor) -> torch.Tensor:
    sh_0_0 = torch.ones_like(x) * 0.5 * math.sqrt(1.0 / math.pi)
    if lmax == 0:
        return torch.stack([sh_0_0,],dim=-1,)

    sh_1_1 = math.sqrt(3.0 / (4.0 * math.pi)) * x
    if lmax == 1:
        return torch.stack([sh_0_0, sh_1_1], dim=-1)

    sh_2_2 = math.sqrt(5.0 / (16.0 * math.pi)) * (3.0 * x**2 - 1.0)
    if lmax == 2:
        return torch.stack([sh_0_0, sh_1_1, sh_2_2], dim=-1)

    sh_3_3 = math.sqrt(7.0 / (16.0 * math.pi)) * x * (5.0 * x**2 - 3.0)
    if lmax == 3:
        return torch.stack([sh_0_0, sh_1_1, sh_2_2, sh_3_3], dim=-1)

    raise ValueError("lmax must be less than 8")



class NoneBasis(torch.nn.Module):
    def __init__(
    self,
    num_radial: int,
    edge_types=512 * 3,
    ) -> None:
        super().__init__()
        self.num_radial = num_radial
        self.linear = nn.Linear(1, num_radial)
        self.activation = nn.SiLU()
        self.linear2 = nn.Linear(num_radial, num_radial)
        # self.mul = nn.Embedding(edge_types, 1, padding_idx=0)
        # self.bias = nn.Embedding(edge_types, 1, padding_idx=0)

    def forward(self, x: torch.Tensor, edge_types=None) -> torch.Tensor:
        x = x.unsqueeze(-1).type_as(self.linear.weight)
        x = self.linear(x)
        x = F.gelu(x)
        x = self.linear2(x)
        # mul = self.mul(edge_types).sum(dim=-2)
        # bias = self.bias(edge_types).sum(dim=-2)
        # x = mul * x + bias
        return x



class SphericalBesselBasis(torch.nn.Module):

    """

    1D spherical Bessel basis

    ```Plain Text
    Parameters
    ----------
    num_radial: int
        Controls maximum frequency.
    cutoff: float
        Cutoff distance in Angstrom.
    """

    def __init__(
        self,
        num_radial: int,
        edge_types=512 * 3,
        cutoff: float = 5.0,
    ) -> None:
        super().__init__()
        # cutoff: float = 5.0
        self.cutoff = cutoff
        self.norm_const = math.sqrt(2 / (cutoff**3))
        # cutoff ** 3 to counteract dividing by d_scaled = d / cutoff

        # Initialize frequencies at canonical positions
        self.frequencies = torch.nn.Parameter(
            data=torch.tensor(np.pi * np.arange(1, num_radial + 1, dtype=np.float32)),
            requires_grad=True,
        )
        self.mul = nn.Embedding(edge_types, 1, padding_idx=0)
        self.bias = nn.Embedding(edge_types, 1, padding_idx=0)
        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1)

    def forward(self, x: torch.Tensor, edge_types=None) -> torch.Tensor:
        x = x / self.cutoff

        basis = (
            self.norm_const
            / (x.unsqueeze(-1))  # )
            * torch.sin(self.frequencies * x.unsqueeze(-1))
        ).type_as(
            self.frequencies
        )  # (num_edges, num_radial)

        mul = self.mul(edge_types).sum(dim=-2)
        bias = self.bias(edge_types).sum(dim=-2)
        basis = mul * basis + bias
        return basis



class SphericalBesselBasis_(torch.nn.Module):
    """
    Equation (7)
    """
    def __init__(self,
                num_basis=8, 
                edge_types=512 * 3,
                r_max: float=5.0,
                trainable=False):
        super().__init__()
        bessel_weights = (
            np.pi
            / r_max
            * torch.linspace(
                start=1.0,
                end=num_basis,
                steps=num_basis,
                dtype=torch.get_default_dtype(),
            )
        )
        if trainable:
            self.bessel_weights = torch.nn.Parameter(bessel_weights)
        else:
            self.register_buffer("bessel_weights", bessel_weights)

        self.register_buffer(
            "r_max", torch.tensor(r_max, dtype=torch.get_default_dtype())
        )
        self.register_buffer(
            "prefactor",
            torch.tensor(np.sqrt(2.0 / r_max), dtype=torch.get_default_dtype()),
        )

        self.mul = nn.Embedding(edge_types, 1, padding_idx=0)
        self.bias = nn.Embedding(edge_types, 1, padding_idx=0)
        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1)

    def forward(self, x: torch.Tensor, edge_types=None) -> torch.Tensor:
        x= x.unsqueeze(-1)
        numerator = torch.sin(self.bessel_weights * x) 
        basis = (self.prefactor * (numerator / x) ).type_as(self.mul.weight)

        mul = self.mul(edge_types).sum(dim=-2)
        bias = self.bias(edge_types).sum(dim=-2)
        basis = mul * basis + bias
        return basis



@torch.jit.script
def gaussian(x, mean, std):
    pi = 3.14159
    a = (2 * pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)

class GaussianLayer(nn.Module):
    def __init__(self, K=128, edge_types=512 * 3):
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
        std = self.stds.weight.float().view(-1).abs() + 8e-1
        return gaussian(x.float(), mean, std).type_as(self.means.weight)



class NonLinear(nn.Module):
    def __init__(self, input, output_size, hidden=None):
        super(NonLinear, self).__init__()
        if hidden is None:
            hidden = input
        self.layer1 = nn.Linear(input, hidden)
        self.layer2 = nn.Linear(hidden, output_size)

    def forward(self, x):
        x = self.layer1(x)
        x = F.gelu(x)
        x = self.layer2(x)
        return x



class MultiheadAttention(nn.Module):

    """Multi-headed attention.

    ```Plain Text
    See "Attention Is All You Need" for more details.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        kdim=None,
        vdim=None,
        dropout=0.0,
        self_attention=False,
        d_tilde=1,
        k_bias=False,
        q_bias=False,
        v_bias=False,
        o_bias=False,
        layer_norm=False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout_module = nn.Dropout(dropout)
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = (
            (self.head_dim / d_tilde) ** 0.5
        ) / self.head_dim  # when d_tilt == 1, match with original transformer scale

        self.self_attention = self_attention

        assert self.self_attention, "Only support self attention"

        assert not self.self_attention or self.qkv_same_dim, (
            "Self-attention requires query, key and " "value to be of the same size"
        )
        self.k_proj = nn.Linear(self.kdim, embed_dim, bias=k_bias)
        self.v_proj = nn.Linear(self.vdim, embed_dim, bias=q_bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=v_bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=o_bias)

        self.layer_norm = nn.LayerNorm(embed_dim) if layer_norm else None

        self.reset_parameters(d_tilde)

    def reset_parameters(self, d_tilde=1):
        if self.qkv_same_dim:
            # Empirically observed the convergence to be much better with
            # the scaled initialization
            nn.init.xavier_uniform_(
                self.k_proj.weight, gain=1.0 / (math.sqrt(2 * d_tilde))
            )
            nn.init.xavier_uniform_(
                self.v_proj.weight, gain=1.0 / (math.sqrt(2 * d_tilde))
            )
            nn.init.xavier_uniform_(
                self.q_proj.weight, gain=1.0 / (math.sqrt(2 * d_tilde))
            )
            if self.q_proj.bias is not None:
                nn.init.constant_(self.out_proj.bias, 0.0)
            if self.k_proj.bias is not None:
                nn.init.constant_(self.out_proj.bias, 0.0)
            if self.v_proj.bias is not None:
                nn.init.constant_(self.out_proj.bias, 0.0)
        else:
            nn.init.xavier_uniform_(self.k_proj.weight, gain=1.0 / math.sqrt(d_tilde))
            nn.init.xavier_uniform_(self.v_proj.weight, gain=1.0 / math.sqrt(d_tilde))
            nn.init.xavier_uniform_(self.q_proj.weight, gain=1.0 / math.sqrt(d_tilde))

        nn.init.xavier_uniform_(self.out_proj.weight, gain=1.0 / math.sqrt(d_tilde))
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)
        if self.layer_norm is not None:
            self.layer_norm.reset_parameters()

    def forward(
        self,
        query,
        key: Optional[Tensor],
        value: Optional[Tensor],
        attn_bias: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        before_softmax: bool = False,
        need_head_weights: bool = False,
        pbc_expand_batched: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Input shape: Time x Batch x Channel

        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
            before_softmax (bool, optional): return the raw attention
                weights and values before the attention softmax.
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default:
                return the average attention weights over all heads.
        """
        if need_head_weights:
            need_weights = True

        tgt_len, bsz, embed_dim = query.size()

        src_len = tgt_len
        assert embed_dim == self.embed_dim, f"query dim {embed_dim} != {self.embed_dim}"
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        if key is not None:
            src_len, key_bsz, _ = key.size()
            if not torch.jit.is_scripting():
                assert key_bsz == bsz
                assert value is not None
                assert src_len, bsz == value.shape[:2]

        q = self.q_proj(query)
        k = self.k_proj(query)
        v = self.v_proj(query)

        # ic(q, k, v, self.v_proj.weight.data, self.v_proj.bias.data)

        q *= self.scaling

        if pbc_expand_batched is not None:
            outcell_index = pbc_expand_batched["outcell_index"]
            expand_mask = pbc_expand_batched["expand_mask"]
        else:
            outcell_index = None
            expand_mask = None

        if outcell_index is not None:
            outcell_index = (
                outcell_index.transpose(1, 0).unsqueeze(-1).expand(-1, -1, embed_dim)
            )
            expand_k = torch.gather(k, dim=0, index=outcell_index)
            expand_v = torch.gather(v, dim=0, index=outcell_index)

            k = torch.cat([k, expand_k], dim=0)
            v = torch.cat([v, expand_v], dim=0)

            src_len = k.size()[0]

        q = (
            q.contiguous()
            .view(tgt_len, bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )
        if k is not None:
            k = (
                k.contiguous()
                .view(-1, bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )
        if v is not None:
            v = (
                v.contiguous()
                .view(-1, bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )

        assert k is not None
        assert k.size(1) == src_len

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

            if outcell_index is not None:
                assert expand_mask is not None
                key_padding_mask = torch.cat([key_padding_mask, expand_mask], dim=1)
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        attn_weights = self.apply_sparse_mask(attn_weights, tgt_len, src_len, bsz)

        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_bias is not None:
            attn_weights += attn_bias.contiguous().view(
                bsz * self.num_heads, tgt_len, src_len
            )

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = (attn_weights + 20.0) * attn_mask.unsqueeze(1) - 20.0
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if before_softmax:
            return attn_weights, v

        attn_weights_float = nn.functional.softmax(attn_weights, dim=-1)

        attn_weights = attn_weights_float.type_as(attn_weights)

        if attn_mask is not None:
            attn_weights = (
                attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
                * attn_mask.unsqueeze(1)
            ).view(bsz * self.num_heads, tgt_len, src_len)
            # attn_weights = attn_weights / attn_weights.sum(dim=-1, keepdim=True)

        attn_probs = self.dropout_module(attn_weights).type_as(attn_weights)

        assert v is not None
        attn = torch.bmm(attn_probs, v)

        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]

        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)

        if self.layer_norm is not None:
            attn = self.layer_norm(attn)

        attn = self.out_proj(attn)

        attn_weights: Optional[Tensor] = None
        if need_weights:
            attn_weights = attn_weights_float.view(
                bsz, self.num_heads, tgt_len, src_len
            ).transpose(1, 0)
            if not need_head_weights:
                # average attention weights over heads
                attn_weights = attn_weights.mean(dim=0)

        return attn, attn_weights

    def apply_sparse_mask(self, attn_weights, tgt_len: int, src_len: int, bsz: int):
        return attn_weights



class GraphormerEncoderLayer(nn.Module):
    """
    Implements a Graphormer Encoder Layer used in BERT/XLM style pre-trained
    models.
    """
    def __init__(
        self,
        graphormer_config: GraphormerConfig,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.dropout_module = nn.Dropout(graphormer_config.dropout)
        self.attention_dropout_module = nn.Dropout(graphormer_config.attention_dropout)
        self.activation_dropout_module = nn.Dropout(
            graphormer_config.activation_dropout
        )

        # Initialize blocks
        self.self_attn = MultiheadAttention(
            graphormer_config.embedding_dim,
            graphormer_config.num_attention_heads,
            dropout=graphormer_config.attention_dropout,
            self_attention=True,
            d_tilde=1,
            k_bias=bias,
            q_bias=bias,
            v_bias=bias,
            o_bias=bias,
            layer_norm=False,
        )

        # layer norm associated with the self attention layer
        self.self_attn_layer_norm = nn.LayerNorm(graphormer_config.embedding_dim)

        self.fc1 = nn.Linear(
            graphormer_config.embedding_dim,
            graphormer_config.ffn_embedding_dim,
            bias=bias,
        )
        self.fc2 = nn.Linear(
            graphormer_config.ffn_embedding_dim,
            graphormer_config.embedding_dim,
            bias=bias,
        )

        # layer norm associated with the position wise feed-forward NN
        self.final_layer_norm = nn.LayerNorm(graphormer_config.embedding_dim)
        self.final_layer_norm_2 = nn.LayerNorm(graphormer_config.ffn_embedding_dim)

        self.activation_function = nn.SiLU()

        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.self_attn_layer_norm.reset_parameters()
        self.final_layer_norm.reset_parameters()
        self.final_layer_norm_2.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        self_attn_bias: Optional[torch.Tensor] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        pbc_expand_batched: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer implementation.
        """

        # x: T x B x C
        residual = x
        x = self.self_attn_layer_norm(x)

        x, attn = self.self_attn(
            query=x,
            key=x,
            value=x,
            attn_bias=self_attn_bias,
            key_padding_mask=self_attn_padding_mask,
            need_weights=False,
            attn_mask=self_attn_mask,
            pbc_expand_batched=pbc_expand_batched,
        )

        x = self.dropout_module(x)
        x = residual + x

        residual = x
        x = self.final_layer_norm(x)
        x = self.activation_function(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.final_layer_norm_2(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = residual + x

        return x, attn



class Graph3DBias(nn.Module):
    """
    Compute 3D attention bias according to the position information for each head.
    """

    def __init__(
        self,
        graphormer_config: GraphormerConfig,
        use_emb_bias: bool = False,
        basis_type: str = "gaussian",
    ):
        super(Graph3DBias, self).__init__()
        # num_rpe_heads = graphormer_config.num_attention_heads * (
        #     graphormer_config.num_encoder_layers + 1
        # )
        self.use_emb_bias = use_emb_bias
        if use_emb_bias:
            if basis_type == "gaussian":
                self.gbf = GaussianLayer(
                    graphormer_config.num_3d_bias_kernel, graphormer_config.num_edges
                )
            elif basis_type == "spherical_bessel":
                self.gbf = SphericalBesselBasis(
                    graphormer_config.num_3d_bias_kernel, graphormer_config.num_edges,cutoff=graphormer_config.pbc_multigraph_cutoff
                )
            elif basis_type == "none":
                self.gbf = NoneBasis(
                    graphormer_config.num_3d_bias_kernel, graphormer_config.num_edges
                )
            # self.gbf_proj = NonLinear(graphormer_config.num_3d_bias_kernel, num_rpe_heads)
            self.edge_proj = nn.Linear(
                graphormer_config.num_3d_bias_kernel, graphormer_config.embedding_dim
            )
            self.pbc_multigraph_cutoff = graphormer_config.pbc_multigraph_cutoff

    def polynomial(self, dist: torch.Tensor, cutoff: float) -> torch.Tensor:
        """
        Polynomial cutoff function,ref: https://arxiv.org/abs/2204.13639
        Args:
            dist (tf.Tensor): distance tensor
            cutoff (float): cutoff distance
        Returns: polynomial cutoff functions
        """
        ratio = torch.div(dist, cutoff)
        result = (
            1
            - 6 * torch.pow(ratio, 5)
            + 15 * torch.pow(ratio, 4)
            - 10 * torch.pow(ratio, 3)
        )
        return torch.clamp(result, min=0.0)

    def forward(self, batched_data, pos, pbc_expand_batched=None, pbc=None, cell=None):
        x, node_type_edge = (
            batched_data["x"],
            batched_data["node_type_edge"],
        )  # pos shape: [n_graphs, n_nodes, 3]

        padding_mask = x.eq(0).all(dim=-1)
        n_node = pos.size()[1]

        if self.use_emb_bias:
            if pbc_expand_batched is not None:
                expand_pos = pbc_expand_batched["expand_pos"]
                expand_mask = pbc_expand_batched["expand_mask"]

                outcell_index = pbc_expand_batched["outcell_index"]
                if len(outcell_index.size()) == 3:
                    total_pos = torch.cat(
                        [pos[:, None, :, :].repeat(1, n_node, 1, 1), expand_pos], dim=2
                    )
                    expand_n_node = total_pos.size(2)
                    delta_pos = pos.unsqueeze(2) - total_pos
                else:
                    expand_pos = expand_pos.masked_fill(
                        expand_mask.unsqueeze(-1).to(torch.bool), 0.0
                    )
                    total_pos = torch.cat([pos, expand_pos], dim=1)
                    expand_n_node = total_pos.size()[1]
                    delta_pos = pos.unsqueeze(2) - total_pos.unsqueeze(
                        1
                    )   # B x T x (expand T) x 3

                unit_mask = delta_pos.norm(dim=-1) < 1e-6
                unit_mask = unit_mask.unsqueeze(-1).expand(-1, -1, -1, 3)
                delta_pos[unit_mask] = torch.abs(delta_pos[unit_mask]) + 20.0

                dist = delta_pos.norm(dim=-1).view(-1, n_node, expand_n_node) # 
                attn_mask = pbc_expand_batched["local_attention_mask"]
            else:
                delta_pos = pos.unsqueeze(1) - pos.unsqueeze(2)
                dist = delta_pos.norm(dim=-1).view(-1, n_node, n_node) # 
                attn_mask = self.polynomial(dist, self.pbc_multigraph_cutoff)

        n_graph, n_node, _ = pos.shape

        atomic_numbers = x[:, :, 0]
        if node_type_edge is None:
            node_type_edge = atomic_numbers.unsqueeze(
                -1
            ) * 128 + atomic_numbers.unsqueeze(1)

            if pbc_expand_batched is not None:
                outcell_index = pbc_expand_batched["outcell_index"]
                node_type_edge = torch.cat(
                    [
                        node_type_edge,
                        torch.gather(
                            node_type_edge,
                            dim=-1,
                            index=outcell_index.unsqueeze(1).repeat(1, n_node, 1),
                        ),
                    ],
                    dim=-1,
                )
            node_type_edge = node_type_edge.unsqueeze(-1)

        merge_edge_features = None
        merge_edge_features_three_body = None
        if self.use_emb_bias:
            edge_feature = self.gbf(
                dist,
                node_type_edge.long(),
            )

            if pbc_expand_batched is None:
                edge_feature = edge_feature.masked_fill(
                    padding_mask.unsqueeze(1).unsqueeze(-1).to(torch.bool), 0.0
                )
            else:
                if len(expand_mask.size()) == 3:
                    full_mask = torch.cat([padding_mask[:, None, :].repeat(1, n_node, 1), expand_mask], dim=-1)
                    edge_feature = edge_feature.masked_fill(
                        full_mask.unsqueeze(-1), 0.0
                    )
                else:
                    full_mask = torch.cat([padding_mask, expand_mask], dim=-1)
                    edge_feature = edge_feature.masked_fill(
                        full_mask.unsqueeze(1).unsqueeze(-1).to(torch.bool), 0.0
                    )
                edge_feature = edge_feature.masked_fill(
                    padding_mask.unsqueeze(-1).unsqueeze(-1).to(torch.bool), 0.0
                )

                edge_feature = (
                    torch.mul(edge_feature, attn_mask.unsqueeze(-1))
                    .float()
                    .type_as(edge_feature)
                )

            sum_edge_features = edge_feature.sum(dim=-2)
            merge_edge_features = self.edge_proj(sum_edge_features)

            merge_edge_features = merge_edge_features.masked_fill_(
                padding_mask.unsqueeze(-1), 0.0
            )

        return None, merge_edge_features, merge_edge_features_three_body, node_type_edge



class UnifiedDecoderNoMask(UnifiedDecoder):
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
    use_attn_bias: bool = False,
    l_max: str = "1x0e+1x1o",
    concat_zero: bool = True,
    basis_type: str = "gaussian",
    use_diff_liear_for_sph: bool = False,
    activation_fn=None,
    ):
        self.l_max = l_max
        self.concat_zero = concat_zero
        self.edge_irreps = e3nn.o3.Irreps(l_max)
        self.use_diff_liear_for_sph = use_diff_liear_for_sph
        self.sph_num = len(self.edge_irreps)
        self.sph_offset = [0]
        for i in range(self.sph_num):
            self.sph_offset.append((i+1)**2)
        # self.sph_offset = [0,1,4,9,16,25]
        rank_num = self.edge_irreps.dim


        super().__init__(
            args,
            num_pred_attn_layer,
            embedding_dim,
            num_attention_heads,
            ffn_embedding_dim,
            dropout,
            attention_dropout,
            activation_dropout,
            num_3d_bias_kernel,
            num_edges,
            num_atoms,
            pbc_multigraph_cutoff,
            use_linear_bias=use_linear_bias,
            rank_num=rank_num,
            use_diff_liear_for_sph=use_diff_liear_for_sph,
            sph_args = [self.sph_num, self.sph_offset],
            activation_fn=activation_fn,
        )
        if basis_type == "gaussian":
            self.unified_gbf_vec = GaussianLayer(num_3d_bias_kernel, num_edges)
        elif basis_type == "spherical_bessel":
            self.unified_gbf_vec = SphericalBesselBasis(num_3d_bias_kernel, num_edges,cutoff=pbc_multigraph_cutoff)
        elif basis_type == "none":
            self.unified_gbf_vec = NoneBasis(num_3d_bias_kernel, num_edges)
        del self.unified_gbf_pos
        self.unified_gbf_pos = None

        del self.unified_gbf_attn_bias
        self.use_attn_bias = use_attn_bias
        if use_attn_bias:
            if basis_type == "gaussian":
                self.unified_gbf_attn_bias = GaussianLayer(
                    num_3d_bias_kernel, num_edges
                )
            elif basis_type == "spherical_bessel":
                self.unified_gbf_attn_bias = SphericalBesselBasis(
                    num_3d_bias_kernel, num_edges,cutoff=pbc_multigraph_cutoff
                )
            elif basis_type == "none":
                self.unified_gbf_attn_bias = NoneBasis(num_3d_bias_kernel, num_edges)
            self.unified_bias_proj = nn.Linear(num_3d_bias_kernel, num_attention_heads)
        self.embedding_dim = embedding_dim
        self.edge_num_irreps = self.edge_irreps.num_irreps
        self.sph_harm = e3nn.o3.SphericalHarmonics(
            self.edge_irreps, normalize=True, normalization="component"
        )

        if use_diff_liear_for_sph:
            pass
        else:
            self.unified_vec_proj_concat = nn.Linear(
                num_3d_bias_kernel, embedding_dim, bias=use_linear_bias
            )

        self.use_precomputed_softmax_weight = args.use_precomputed_softmax_weight

        self.num_attention_heads = num_attention_heads


    def polynomial(self, dist: torch.Tensor, cutoff: float) -> torch.Tensor:
        ratio = torch.div(dist, cutoff)
        result = (
            1
            - 6 * torch.pow(ratio, 5)
            + 15 * torch.pow(ratio, 4)
            - 10 * torch.pow(ratio, 3)
        )
        return torch.clamp(result, min=0.0)


    def compute_softmax_weights(self, pos, mixed_attn_bias, pbc_expand_batched, padding_mask):
        if pbc_expand_batched is not None and pbc_expand_batched["local_attention_mask"] is not None:
            batch_size = pos.size(0)
            num_atoms = pos.size(1)
            num_heads = mixed_attn_bias.size(1)
            attn_bias_softmax = torch.softmax(mixed_attn_bias, dim=-1).to(torch.float64)
            # attn_bias_softmax = torch.ones_like(mixed_attn_bias, dtype=torch.float, device=mixed_attn_bias.device)
            local_attnetion_weight = pbc_expand_batched["local_attention_mask"].to(torch.float64)
            outcell_index = pbc_expand_batched["outcell_index"]
            expand_mask = pbc_expand_batched["expand_mask"]
            if len(expand_mask.size()) == 3: # per atom outcell index
                all_mask = torch.cat([padding_mask[:, None, :].repeat(1, num_atoms, 1), expand_mask], dim=-1) # B x T x expand T
                in_cell_index = torch.arange(0, num_atoms, device=pos.device, dtype=outcell_index.dtype)[None, None, :].repeat(batch_size, num_atoms, 1)
                full_index = torch.cat([in_cell_index, outcell_index], dim=-1)
                full_index = full_index.masked_fill(all_mask, num_atoms)[:, None, :, :].repeat(1, num_heads, 1, 1)
                softmax_weights = torch.zeros([pos.size(0), num_heads, num_atoms, num_atoms + 1], dtype=torch.float64, device=attn_bias_softmax.device) # B x T x T
                local_attnetion_weight = local_attnetion_weight.masked_fill(all_mask, 0.0) # B x T x expand T
                softmax_weights = softmax_weights.scatter_add(dim=-1, index=full_index, src=attn_bias_softmax * local_attnetion_weight[:, None, :, :].repeat(1, num_heads, 1, 1)) # B x T x T
                weighetd_softmax_weights = torch.zeros([pos.size(0), num_heads, num_atoms, num_atoms + 1], dtype=attn_bias_softmax.dtype, device=attn_bias_softmax.device) # B x T x T
                weighetd_softmax_weights = weighetd_softmax_weights.scatter_add(dim=-1, index=full_index, src=attn_bias_softmax * local_attnetion_weight[:, None, :, :].repeat(1, num_heads, 1, 1)* local_attnetion_weight[:, None, :, :].repeat(1, num_heads, 1, 1)) # B x T x T
            else:
                all_mask = torch.cat([padding_mask, expand_mask], dim=-1) # B x expand T
                local_attnetion_weight = local_attnetion_weight.masked_fill(all_mask[:, None, :].repeat(1, num_atoms, 1), 0.0) # B x T x expand T
                in_cell_index = torch.arange(0, num_atoms, device=pos.device, dtype=outcell_index.dtype)[None, :].repeat(batch_size, 1)
                full_index = torch.cat([in_cell_index, outcell_index], dim=-1)
                full_index = full_index.masked_fill(all_mask, num_atoms)[:, None, None, :].repeat(1, num_heads, num_atoms, 1)
                softmax_weights = torch.zeros([pos.size(0), num_heads, num_atoms, num_atoms + 1], dtype=torch.float64, device=attn_bias_softmax.device) # B x T x T
                softmax_weights = softmax_weights.scatter_add(dim=-1, index=full_index, src=attn_bias_softmax * local_attnetion_weight[:, None, :, :].repeat(1, num_heads, 1, 1)) # B x T x T
                weighetd_softmax_weights = torch.zeros([pos.size(0), num_heads, num_atoms, num_atoms + 1], dtype=attn_bias_softmax.dtype, device=attn_bias_softmax.device) # B x T x T
                weighetd_softmax_weights = weighetd_softmax_weights.scatter_add(dim=-1, index=full_index, src=attn_bias_softmax * local_attnetion_weight[:, None, :, :].repeat(1, num_heads, 1, 1)* local_attnetion_weight[:, None, :, :].repeat(1, num_heads, 1, 1)) # B x T x T
            return [softmax_weights[:, :, :, :-1],weighetd_softmax_weights[:, :, :, :-1]]
        else:
            return None


    def forward(
        self,
        x,
        pos,
        node_type_edge,
        node_type,
        padding_mask,
        pbc_expand_batched: Optional[Dict] = None,
    ):
        n_node = pos.shape[1]

        # x = x.contiguous().transpose(0, 1)

        if pbc_expand_batched is not None:
            pbc_expand_batched["outcell_index"]
            outcell_index = pbc_expand_batched["outcell_index"]
            if len(outcell_index.size()) == 3:
                expand_pos = torch.cat([pos[:, None, :, :].repeat(1, n_node, 1, 1), pbc_expand_batched["expand_pos"]], dim=2)
                expand_n_node = expand_pos.size(2)
                uni_delta_pos = (pos.unsqueeze(2) - expand_pos).to(
                    dtype=x.dtype
                )
            else:
                expand_pos = torch.cat([pos, pbc_expand_batched["expand_pos"]], dim=1)
                expand_n_node = expand_pos.shape[1]
                uni_delta_pos = pos.unsqueeze(2) - expand_pos.unsqueeze(1) 

            unit_mask = uni_delta_pos.norm(dim=-1) < 1e-6
            unit_mask = unit_mask.unsqueeze(-1).expand(-1, -1, -1, 3)
            uni_delta_pos[unit_mask] = torch.abs(uni_delta_pos[unit_mask]) + 20.0
            
            dist = uni_delta_pos.norm(dim=-1).view(-1, n_node, expand_n_node) # 
        else:
            uni_delta_pos = pos.unsqueeze(1) - pos.unsqueeze(2)
            dist = uni_delta_pos.norm(dim=-1).view(-1, n_node, n_node) # 

        # attn_mask = self.polynomial(dist, self.pbc_multigraph_cutoff)
        # uni_delta_pos = uni_delta_pos / (dist.unsqueeze(-1) + 1e-5)

        # r_ij/||r_ij|| * gbf(||r_ij||)
        vec_gbf = self.unified_gbf_vec(
            dist, node_type_edge.long()
        )  # n_graph x n_node x expand_n_node x num_kernel
        # vec_gbf = torch.mul(torch.ones_like(vec_gbf,device=vec_gbf.device),dist.unsqueeze(-1))
        sph = self.sph_harm(uni_delta_pos)
        vec_gbf = sph.unsqueeze(-1) * vec_gbf.unsqueeze(
            -2
        )  # n_graph x n_node x expand_n_node x 4 x num_kernel

        # reduce ij -> i by \sum_j vec_ij * x_j
        if pbc_expand_batched is not None:
            if len(outcell_index.size()) == 3:
                expand_mask = torch.cat(
                    [padding_mask[:, None, :].repeat(1, n_node, 1), pbc_expand_batched["expand_mask"]], dim=-1
                )
            else:
                expand_mask = pbc_expand_batched["expand_mask"]
                expand_mask = torch.cat([padding_mask, expand_mask], dim=-1)
            attn_mask = pbc_expand_batched["local_attention_mask"]
            if len(expand_mask.size()) == 3:
                vec_gbf = vec_gbf.masked_fill(
                    expand_mask.unsqueeze(-1).unsqueeze(-1), 0.0
                )
            else:
                vec_gbf = vec_gbf.masked_fill(
                    expand_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1), 0.0
                )

            vec_gbf = vec_gbf * attn_mask.unsqueeze(-1).unsqueeze(-1)
        else:
            attn_mask = self.polynomial(dist, self.pbc_multigraph_cutoff)
            vec_gbf = vec_gbf.masked_fill(
                padding_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1), 0.0
            )

        vec_gbf = vec_gbf.sum(dim=2).float().type_as(x)

        if self.use_diff_liear_for_sph:
            vec = self.unified_vec_proj(vec_gbf)  # n_graph x n_node x num_kernel
        else:
            if self.concat_zero:
                vec_gbf_concat = vec_gbf[:, :, 0, :].unsqueeze(-2)
                vec_concat = self.unified_vec_proj_concat(vec_gbf_concat)

                vec_gbf = vec_gbf[:, :, 1:, :]
                vec = self.unified_vec_proj(vec_gbf)
                # vec = vec_gbf_feature * x.unsqueeze(-2)

                vec = torch.cat([vec_concat, vec], dim=-2)
            else:
                vec = self.unified_vec_proj(vec_gbf)

        if pbc_expand_batched["noise_add_forces"] is not None:
            vec = vec + pbc_expand_batched["noise_add_forces"]
        
        pos_mean_centered_dist = pos.norm(dim=-1)
        pos_mean_centered_unit = pos / (pos_mean_centered_dist.unsqueeze(-1) + 1e-2)

        if self.use_attn_bias:
            uni_gbf_feature = self.unified_gbf_attn_bias(dist, node_type_edge.long())

            uni_graph_attn_bias = (
                self.unified_bias_proj(uni_gbf_feature).permute(0, 3, 1, 2).contiguous()
            )
        else:
            uni_graph_attn_bias = None

        if pbc_expand_batched is not None:
            if len(expand_mask.size()) == 3:
                attn_mask = attn_mask.masked_fill(
                    expand_mask, 0.0
                )  # other nodes don't attend to padding nodes
            else:
                attn_mask = attn_mask.masked_fill(
                    expand_mask.unsqueeze(1), 0.0
                )  # other nodes don't attend to padding nodes
        else:
            attn_mask = attn_mask.masked_fill(
                padding_mask.unsqueeze(1), 0.0
            )  # other nodes don't attend to padding nodes

        output = x
        output = output.masked_fill(padding_mask.unsqueeze(-1), 0.0)


        if self.use_precomputed_softmax_weight:
            softmax_weights = self.compute_softmax_weights(pos, uni_graph_attn_bias, pbc_expand_batched, padding_mask)
            mask_couple=[attn_mask, padding_mask]
        else:
            softmax_weights = None
            mask_couple = [attn_mask, expand_mask]

        last_vec = None
        last_output = None
        num_pred_attn_layer = len(self.unified_encoder_layers)
        for i, layer in enumerate(self.unified_encoder_layers):
            output, vec = layer(
                output,
                vec,
                uni_graph_attn_bias,
                uni_graph_attn_bias,
                uni_graph_attn_bias,
                uni_graph_attn_bias,
                mask_couple,
                [pos_mean_centered_unit, uni_delta_pos],
                [dist, node_type_edge],
                pbc_expand_batched=pbc_expand_batched,
                softmax_weights=softmax_weights,
            )

            if i == num_pred_attn_layer - 2:
                last_vec = vec
                last_output = output

        if self.use_diff_liear_for_sph:
            output = output + vec[:,:,0,:]
        vec = vec[:, :, 1:4, :]
        node_output = self.unified_final_equivariant_ln(vec)
        output = self.unified_final_invariant_ln(output)

        node_output = self.unified_output_layer(node_output).squeeze(-1)
        node_output = node_output.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        return node_output, output, last_vec, last_output



class GradientHead(nn.Module):
    def __init__(self, force_std, stress_std,use_hessian=False,hessian_num=0.1):
        super(GradientHead, self).__init__()
        self.force_std = force_std
        self.stress_std = stress_std
        self.use_hessian = use_hessian
        self.hessian_num = hessian_num


    def forward(self, 
                energy, 
                pos, 
                strain, 
                volume,
                orig_hessian_mask=None,
                natoms=None,
                batch_data=None):
        grad_outputs = [torch.ones_like(energy)]

        grad = torch.autograd.grad(
            outputs=energy,
            inputs=[pos, strain],
            grad_outputs=grad_outputs,
            create_graph=self.training or self.use_hessian, 
            retain_graph=True
        )
        force_grad = grad[0] / self.force_std
        stress_grad = grad[1] / self.stress_std

        if force_grad is not None:
            forces = torch.neg(force_grad)

        if stress_grad is not None:
            stresses = 1 / volume[:, None, None] * stress_grad * 160.21766208

        hessian = None
        hessian_mask = None
        valid_hessian_natoms=natoms[orig_hessian_mask]
        if self.use_hessian and valid_hessian_natoms.shape[0] > 0:
            N = pos.shape[1]
            hessian = torch.zeros([pos.shape[0], pos.shape[1],pos.shape[1], 3, 3], device=pos.device).type_as(pos)
            hessian_mask = torch.zeros((N, 3), dtype=torch.bool, device=pos.device)
            if self.hessian_num > 0 and N > 0:
                total_positions = torch.max(valid_hessian_natoms).item()
                num_selected = min(self.hessian_num, total_positions)
                flat_indices = torch.randperm(total_positions, device=pos.device)[:num_selected]
                atoms = flat_indices
                hessian_mask[atoms, :] = True


            for i in range(forces.shape[1]): # B x N x 3
                hessian_row = []
                for alpha in range(forces.shape[2]):
                    if hessian_mask[i, alpha] == 0:
                        continue 
                    hessian_row=torch.autograd.grad(
                        outputs=forces[:, i, alpha].unsqueeze(-1).unsqueeze(-1),
                        inputs=pos,
                        grad_outputs=torch.ones_like(forces[:, i, alpha].unsqueeze(-1).unsqueeze(-1)),
                        create_graph=True,
                        retain_graph=True
                    )[0] # [B,N,3]
                    # B x N x 3 x N x 3 
                    hessian[:,i,:,alpha,:] = hessian_row # B x N x N x 3 x 3
            hessian = -hessian

        return forces, stresses,hessian,hessian_mask



class AtomEnergies(nn.Module):
    def __init__(
    self, energy_mean, energy_std, offset: int = 2
    ):  # H in ase is 1,and in preprocess,we add 2 to atomic number,so offset = 2
        """
        AtomEnergies is a PyTorch module that provides atomic energies for given atomic numbers.
        Parameters:
        energy_file (str): Path to the JSON file containing the atomic energies.
        offset (int): An offset to apply to the atomic numbers. Default is 2.
        """
        self.energy_mean = energy_mean
        self.energy_std = energy_std
        super(AtomEnergies, self).__init__()
        e0s = PBE_bias
        from ase.data import atomic_numbers

        self.e0s_tensor = torch.zeros(
            len(atomic_numbers) + 4, dtype=torch.float32, device="cuda"
        )
        for Z in list(e0s.keys()):
            self.e0s_tensor[atomic_numbers[Z] + offset] = e0s[Z]

        self.e0s_tensor = self.e0s_tensor / energy_std

    def forward(self, atomic_numbers: torch.Tensor) -> torch.Tensor:
        """
        Compute the energies for the given atomic numbers.
        Parameters:
        atomic_numbers (torch.Tensor): A tensor of atomic numbers.
        Returns:
        torch.Tensor: A tensor of energies corresponding to the atomic numbers.
        """
        energies = self.e0s_tensor[atomic_numbers]

        return energies



class Geomformer(nn.Module):
    def __init__(
    self,
    cli_args,
    energy_mean=0,
    energy_std=1,
    force_mean=0,
    force_std=1,
    force_loss_factor=1,
    stress_mean=0,
    stress_std=1,
    stress_loss_factor=1,
    use_stress_loss=1,
    return_all_info=False,
    skip_load=False,
    *args,
    **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        graphormer_config: GraphormerConfig = GraphormerConfig(cli_args)
        self.use_multi_modal = cli_args.use_multi_modal
        self.return_all_info = return_all_info
        # atom embedding
        if self.use_multi_modal:
            self.atom_encoder = nn.Embedding(
            graphormer_config.num_atoms + 1,
            graphormer_config.embedding_dim - 2,
            padding_idx=0,
            )
        else:
            self.atom_encoder = nn.Embedding(
            graphormer_config.num_atoms + 1,
            graphormer_config.embedding_dim,
            padding_idx=0,
            )

        # GBF encoder
        self.graph_3d_pos_encoder = Graph3DBias(
            graphormer_config=graphormer_config,
            use_emb_bias=cli_args.use_emb_bias,
            basis_type=cli_args.basis_type,
        )

        self.use_autograd_force = cli_args.use_autograd_force
        self.cell_expander = CellExpander(
            graphormer_config.pbc_cutoff,
            graphormer_config.pbc_expanded_token_cutoff,
            graphormer_config.pbc_expanded_num_cell_per_direction,
            graphormer_config.pbc_multigraph_cutoff,
            backprop=cli_args.use_autograd_force,
            original_token_count=False,
        )


        self.gradient_head = GradientHead(force_std, stress_std,
                                        use_hessian=cli_args.use_hessian,
                                        hessian_num=cli_args.hessian_num)

        # self.energy_loss = nn.SmoothL1Loss(reduction="mean")
        # self.force_loss = nn.SmoothL1Loss(reduction="none")
        # self.force_mae_loss = nn.SmoothL1Loss(reduction="none")
        # self.stress_loss = nn.SmoothL1Loss(reduction="none")

        self.energy_loss = nn.L1Loss(reduction="mean")
        self.force_loss = nn.L1Loss(reduction="none")
        self.force_mae_loss = nn.L1Loss(reduction="none")
        self.stress_loss = nn.L1Loss(reduction="mean")

        self.energy_huberloss = nn.HuberLoss(reduction="mean", delta=0.01)
        self.force_huberloss = nn.HuberLoss(reduction="none", delta=0.01)
        self.stress_huberloss = nn.HuberLoss(reduction="mean", delta=0.01)

        # self.hessian_loss = nn.SmoothL1Loss(reduction="none")
        self.hessian_loss = nn.HuberLoss(reduction="none", delta=0.01)

        # if cli_args.use_simple_head:
        #     self.node_head = NodeTaskHead(
        #         graphormer_config.embedding_dim, graphormer_config.num_attention_heads
        #     )
        # else:
        self.node_head = UnifiedDecoderNoMask(
            cli_args,
            graphormer_config.num_pred_attn_layer,
            graphormer_config.embedding_dim,
            graphormer_config.num_attention_heads,
            graphormer_config.ffn_embedding_dim,
            graphormer_config.dropout,
            graphormer_config.attention_dropout,
            graphormer_config.activation_dropout,
            graphormer_config.num_3d_bias_kernel,
            graphormer_config.num_edges,
            graphormer_config.num_atoms,
            graphormer_config.pbc_multigraph_cutoff,
            use_linear_bias=cli_args.use_linear_bias,
            use_attn_bias=cli_args.use_attn_bias,
            l_max=cli_args.l_max,
            concat_zero=cli_args.concat_zero,
            basis_type=cli_args.basis_type,
            use_diff_liear_for_sph=cli_args.use_diff_liear_for_sph,
            activation_fn=cli_args.act_type,
        )

        self.energy_mean = energy_mean
        self.energy_std = energy_std
        self.force_mean = force_mean
        self.force_std = force_std
        self.force_loss_factor = force_loss_factor
        self.energy_loss_factor = cli_args.energy_loss_factor
        self.stress_mean = stress_mean
        self.stress_std = stress_std
        self.stress_loss_factor = stress_loss_factor
        self.hessian_loss_factor = cli_args.hessian_loss_factor

        self.activation_function = nn.SiLU() if cli_args.act_type == "silu" else nn.GELU()
        self.layer_norm = nn.LayerNorm(graphormer_config.embedding_dim, eps=1e-10)
        self.lm_head_transform_weight = nn.Linear(
            graphormer_config.embedding_dim, graphormer_config.embedding_dim
        )
        self.energy_out = nn.Linear(graphormer_config.embedding_dim, 1)

        self.use_e0s = cli_args.use_e0s
        self.e0s = AtomEnergies(energy_mean=energy_mean, energy_std=energy_std)

        self.usehessian = cli_args.use_hessian
        self.hessian_num = cli_args.hessian_num
        # self.heads = nn.ModuleDict()
        # self.heads.update(
        #     {
        #         "Bandgap": BandGapHead(
        #             graphormer_config=graphormer_config,
        #         )
        #     }
        # )

        if not skip_load:
            if cli_args.loadcheck_path != "":
                self.load_state_dict(
                    torch.load(cli_args.loadcheck_path, map_location="cpu",weights_only=False)["model"],
                    strict=False,
                )

        self.use_simple_head = cli_args.use_simple_head
        self.graphormer_config = graphormer_config
        self.use_stress_loss = use_stress_loss

        self.use_predict_forces = cli_args.use_predict_forces
        self.use_force = cli_args.use_force
        self.test_memory = cli_args.test_memory

        self.large_model = cli_args.large_model

        self.use_precomputed_softmax_weight = cli_args.use_precomputed_softmax_weight
        self.use_pbc_on_cpu = cli_args.use_pbc_on_cpu
        self.use_per_atom_outcell_index = cli_args.use_per_atom_outcell_index


    def construct_pbc_expand_batched(self, batched_data, use_local_attention, use_per_atom_outcell_index):
        expand_len = batched_data["expand_len"]
        outcell_index = batched_data["outcell_index"]
        outcell_cell_index = batched_data["outcell_cell_index"]
        pos = batched_data["pos"]
        cell = batched_data["cell"]
        pair_token_type = batched_data["node_type_edge"]
        atomic_numbers = batched_data["x"][:, :, 0]  # B x T
        if pair_token_type is None:
            pair_token_type = atomic_numbers.unsqueeze(
                -1
            ) * 128 + atomic_numbers.unsqueeze(1)
            pair_token_type = pair_token_type.unsqueeze(-1)
        num_atoms = batched_data["natoms"]
        return self.cell_expander.construct_pbc_expand_batched(expand_len, num_atoms, outcell_index, outcell_cell_index, pos, cell, pair_token_type, use_local_attention, use_per_atom_outcell_index)


    def forward(self, batched_data):
        atomic_numbers = batched_data["x"]
        pbc = batched_data["pbc"]
        cell = batched_data["cell"]
        pos = batched_data["pos"]

        if self.test_memory:
            padding_to = 200
            padding_num = padding_to - pos.size(1)
            if padding_num > 0:
                padding_pos = torch.zeros(
                    pos.size(0), padding_num, 3, device=pos.device, dtype=pos.dtype
                )
                pos = torch.cat([pos, padding_pos], dim=1)
                atomic_numbers = torch.cat(
                    [
                        atomic_numbers,
                        torch.ones(
                            atomic_numbers.size(0), padding_num, atomic_numbers.size(2)
                        )
                        .to(atomic_numbers.device)
                        .to(atomic_numbers.dtype),
                    ],
                    dim=1,
                )
                batched_data["natoms"] = torch.full_like(
                    batched_data["natoms"],
                    padding_to,
                    device=batched_data["natoms"].device,
                )
                atomic_numbers = torch.where(
                    atomic_numbers.eq(0),
                    torch.tensor(1, device=atomic_numbers.device).to(
                        atomic_numbers.dtype
                    ),
                    atomic_numbers,
                )
                batched_data["x"] = atomic_numbers
                batched_data["pos"] = pos

        # pos = torch.where(pos.eq(0), torch.tensor(1e-8, device=pos.device), pos)

        padding_mask = (atomic_numbers[:, :, 0]).eq(0)  # B x T
        n_graph, n_node = atomic_numbers.size()[:2]
        # stress
        # if self.use_stress_loss:
        volume = torch.abs(torch.linalg.det(cell))  # to avoid negative volume

        pbc_expand_batched = {}
        if self.use_pbc_on_cpu:
            pbc_dict = self.construct_pbc_expand_batched(batched_data, True, self.use_per_atom_outcell_index)
        else:
            pbc_dict = self.cell_expander.expand(
                pos,
                pbc,
                atomic_numbers[:, :, 0],
                cell,
                batched_data["natoms"],
                use_local_attention=True,
            )

        pos = pbc_dict["pos"]
        # print(pos.shape)
        cell = pbc_dict["cell"]
        expand_pos = pbc_dict["expand_pos"]
        # print(expand_pos.shape)
        outcell_index = pbc_dict["outcell_index"]
        expand_mask = pbc_dict["expand_mask"]
        local_attention_mask = pbc_dict["local_attention_weight"]
        strain = pbc_dict["strain"]

        pbc_expand_batched["expand_pos"] = expand_pos
        pbc_expand_batched["outcell_index"] = outcell_index
        pbc_expand_batched["expand_mask"] = expand_mask
        pbc_expand_batched["noise_add_forces"]=batched_data["noise_add_forces"] if "noise_add_forces" in batched_data else None
        batched_data["node_type_edge"] = pbc_dict["expand_node_type_edge"] if "expand_node_type_edge" in pbc_dict else None

        x: Tensor = self.atom_encoder(atomic_numbers).sum(dim=-2)

        if self.use_multi_modal:
            # batched_data["bader_charge"] = torch.zeros_like(batched_data["bader_charge"])
            # batched_data["magmom"] = torch.zeros_like(batched_data["magmom"])
            # if self.training:
            #     # Randomly set magmom and bader_charge to zero with 50% probability
            #     if torch.rand(1).item() < 0.5:
            #         batched_data["magmom"] = torch.zeros_like(batched_data["magmom"])
            #     if torch.rand(1).item() < 0.5:
            #         batched_data["bader_charge"] = torch.zeros_like(batched_data["bader_charge"])

            x = torch.cat(
                [
                    x,
                    batched_data["magmom"].float(),
                    batched_data["bader_charge"].float(),
                ],
                dim=-1,
            )
            # x = torch.cat([x, batched_data["magmom"].float(), batched_data["bader_charge"].float()], dim=-1)
            # x = torch.cat([x, batched_data["magmom"].float()], dim=-1)
            # x = torch.cat([x,batched_data["bader_charge"].float()], dim=-1)
        pbc_expand_batched["local_attention_mask"] = local_attention_mask

        (
            attn_bias,
            merged_edge_features,
            merge_edge_features_three_body,
            node_type_edge,
        ) = self.graph_3d_pos_encoder(
            batched_data, pos, pbc_expand_batched=pbc_expand_batched, pbc=pbc, cell=cell
        )

        if "noise_add_forces_inv" in batched_data:
            x = x + batched_data["noise_add_forces_inv"]

        if merged_edge_features is not None:
            x = x + merged_edge_features
        if merge_edge_features_three_body is not None:
            x = x + merge_edge_features_three_body

        node_type = atomic_numbers[:, :, 0]

        forces, x, last_vec,last_output = self.node_head(
            x,
            pos,
            node_type_edge,
            node_type,
            padding_mask,
            pbc_expand_batched,
        )


        # use mean pooling
        energy_per_atom = self.layer_norm(
            self.activation_function(self.lm_head_transform_weight(x))
        )
        energy_per_atom = self.energy_out(
            energy_per_atom
        )  # per atom energy, get from graph token

        if self.use_e0s:
            atomic_energy = self.e0s(atomic_numbers)
            energy_per_atom = energy_per_atom + atomic_energy

        energy = torch.sum(
            energy_per_atom.masked_fill(padding_mask.unsqueeze(-1), 0.0), dim=1
        ) / batched_data["natoms"].unsqueeze(-1)

        # --------bandgap-------
        # bandgap = self.heads["Bandgap"](x,last_vec,padding_mask,batched_data,pbc_expand_batched)
        # energy = bandgap
        # ----------------

        max_natoms = torch.max(batched_data["natoms"])
        # # additional stress:use energy gradient
        hessian = None
        hessian_mask = None
        direct_pred_forces = forces if self.use_predict_forces else None

        if self.use_force is False:
            grad_forces = None
            grad_stresses = None
            self.useforce = False
            self.usestress = False
        elif self.large_model and self.use_stress_loss and max_natoms > 512:
            grad_forces = None
            grad_stresses = None
            self.useforce = False
            self.usestress = False
        elif (
            self.use_stress_loss is False
            and max_natoms < 512
            and self.use_autograd_force
        ):  # 230 1.3B:216  finetune:260
            grad_forces, grad_stresses,hessian,hessian_mask = self.gradient_head(
                torch.mul(
                    energy * self.energy_std + self.energy_mean,
                    batched_data["natoms"].unsqueeze(-1),
                ),
                pos,
                strain,
                volume,
                orig_hessian_mask=batched_data["hessian_mask"],
                natoms=batched_data["natoms"],
                batch_data=batched_data,
            )
            self.useforce = True
            self.usestress = False
        elif (
            self.use_stress_loss
            and self.use_autograd_force
            and (not self.training or max_natoms < 512)
        ):
            grad_forces, grad_stresses,hessian,hessian_mask = self.gradient_head(
                torch.mul(
                    energy * self.energy_std + self.energy_mean,
                    batched_data["natoms"].unsqueeze(-1),
                ),
                pos,
                strain,
                volume,
                orig_hessian_mask=batched_data["hessian_mask"],
                natoms=batched_data["natoms"],
                batch_data=batched_data,
            )
            self.useforce = True
            self.usestress = True
        else:
            grad_forces = None
            grad_stresses = None
            self.useforce = True
            self.usestress = False

        if self.use_simple_head:
            forces = grad_forces
        else:
            forces = grad_forces
            stresses = grad_stresses
            forces = forces[:, :n_node] if forces is not None else None
            forces = (
                forces.masked_fill(padding_mask.unsqueeze(-1), 0.0)
                if forces is not None
                else None
            )
            if hessian is not None:
                hessian = hessian.masked_fill(padding_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1), 0.0)
                hessian = hessian.masked_fill(padding_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1), 0.0)

        self.usehessian = True if hessian is not None else False
        if self.return_all_info:
            return_dict = {}
            return_dict["energy"] = energy
            return_dict["forces"] = forces
            return_dict["stresses"] = stresses if self.use_stress_loss else None
            return_dict["padding_mask"] = padding_mask
            return_dict["direct_pred_forces"] = direct_pred_forces
            return_dict["last_vec"] = last_vec
            return_dict["x"] = last_output
            return_dict["useforce"] = self.useforce
            return_dict["usestress"] = self.usestress
            return_dict["pbc_expand_batched"] = pbc_expand_batched
            return_dict["node_type_edge"] = node_type_edge
            return return_dict


        if self.use_stress_loss:
            # if torch.any(stresses.isnan()):
            #     logger.info(f"found nan in stress: {torch.any(stresses.isnan())}")
            return energy, forces, stresses, padding_mask, direct_pred_forces,hessian,hessian_mask

        return energy, forces, padding_mask, direct_pred_forces,hessian,hessian_mask

    def compute_loss(self, model_output, batch_data) -> ModelOutput:
        if self.use_stress_loss:
            energy, forces, stress, padding_mask, direct_pred_forces, hessian,hessian_mask = model_output
        else:
            energy, forces, padding_mask, direct_pred_forces, hessian,hessian_mask = model_output

        bs = energy.shape[0]
        natoms = batch_data["natoms"]
        L1loss = nn.L1Loss(reduction="none")

        pred_energy = energy * self.energy_std + self.energy_mean
        if self.useforce:
            pred_forces = (
                forces * self.force_std + self.force_mean
                if forces is not None
                else None
            )
            direct_pred_forces = (
                direct_pred_forces * self.force_std + self.force_mean
                if direct_pred_forces is not None
                else None
            )
        if self.usestress:
            pred_stress = stress * self.stress_std + self.stress_mean
        energy_loss = self.energy_huberloss(
            batch_data["y"].reshape(-1).float(),
            pred_energy.reshape(-1).float(),
        )

        if self.large_model and natoms.max() > 310:
            energy_loss = torch.tensor(0.0).to(energy_loss.device).requires_grad_()
        # self.total_energy_loss.extend(pred_energy.cpu().detach().numpy().flatten().tolist())
        # self.ture_total_energy_loss.extend(batch_data["y"].cpu().detach().numpy().flatten().tolist())
        energy_mae_loss = L1loss(
            batch_data["y"].reshape(-1),
            pred_energy.reshape(-1),
        )

        energy_mae_loss = energy_mae_loss.mean()

        if self.useforce:
            if self.test_memory:
                batch_data["forces"] = (
                    forces.detach() * 2
                    if forces is not None
                    else direct_pred_forces.detach() * 2
                )

            pad_seq_len = (
                forces.size()[1] if forces is not None else direct_pred_forces.size()[1]
            )
            # self.total_force_loss.extend(pred_forces.cpu().detach().numpy().flatten().tolist())
            # self.true_total_force_loss.extend(batch_data["forces"].cpu().detach().numpy().flatten().tolist())
            if self.use_autograd_force:
                force_loss = self.force_huberloss(
                    batch_data["forces"].reshape(-1).float(),
                    pred_forces.reshape(-1).float(),
                ).reshape(bs, pad_seq_len, 3)
                # force_loss = force_loss.masked_fill(padding_mask.unsqueeze(-1), 0.0).sum() / (
                #     natoms.sum() * 3.0
                # )
                force_loss = (
                    (force_loss.masked_fill(padding_mask.unsqueeze(-1), 0.0))
                    .sum(dim=-1)
                    .sum(dim=-1)
                    / (natoms * 3.0)
                ).mean()
                force_mae_loss = L1loss(
                    batch_data["forces"].reshape(-1).float().type_as(energy),
                    pred_forces.reshape(-1),
                ).reshape(bs, pad_seq_len, 3)

                force_mae_loss = (
                    force_mae_loss.masked_fill(padding_mask.unsqueeze(-1), 0.0)
                    .norm(dim=-1)
                    .sum()
                    / natoms.sum()
                )
                if torch.any(batch_data["forces"].isnan()):
                    logger.warning("nan found in force labels")
            else:
                force_loss = 0.0
                force_mae_loss = 0.0
            if self.use_predict_forces:
                direct_force_loss = self.force_huberloss(
                    batch_data["forces"].reshape(-1).float(),
                    direct_pred_forces.reshape(-1).float(),
                ).reshape(bs, pad_seq_len, 3)
                direct_force_loss = (
                    (direct_force_loss.masked_fill(padding_mask.unsqueeze(-1), 0.0))
                    .sum(dim=-1)
                    .sum(dim=-1)
                    / (natoms * 3.0)
                ).mean()
            else:
                direct_force_loss = 0.0
        else:
            force_loss = 0.0
            force_mae_loss = 0.0
            direct_force_loss = 0.0

        if self.use_stress_loss:
            if self.usestress:
                # self.total_stress_loss.extend(pred_stress.cpu().detach().numpy().flatten().tolist())
                # self.true_total_stress_loss.extend(batch_data["stress"].cpu().detach().numpy().flatten().tolist())
                stress_mae_loss = L1loss(
                    batch_data["stress"].reshape(-1),
                    pred_stress.reshape(-1),
                ).reshape(bs, 3, 3)
                stress_mae_loss = stress_mae_loss.norm(dim=-1).sum(dim=-1) / natoms
                stress_loss = self.stress_huberloss(
                    batch_data["stress"].reshape(-1).float(),
                    pred_stress.reshape(-1).float(),
                )

                stress_mae_loss = stress_mae_loss.sum() / bs
                if self.use_stress_loss and torch.any(batch_data["stress"].isnan()):
                    logger.warning("nan found in stress labels")
                    stress_non_flag = True
                    stress_loss = 0.0
                    stress_mae_loss = 0.0
            else:
                stress_loss = 0.0
                stress_mae_loss = 0.0

        if self.usehessian:
            # # -----------------use finite difference to compute hessian---------------------
            # pos = batch_data["pos"]
            # valid_hessian_natoms=batch_data["natoms"][batch_data["hessian_mask"]]
            # if self.usehessian:
            #     hessian = torch.zeros([pos.shape[0], pos.shape[1],pos.shape[1], 3, 3], device=pos.device).type_as(pos)
            #     N = pos.shape[1]
            #     hessian_mask = torch.zeros((N, 3), dtype=torch.bool, device=pos.device)
            #     if self.hessian_num > 0 and N > 0:
            #         total_positions = torch.max(valid_hessian_natoms).item()
            #         num_selected = min(self.hessian_num, total_positions)
            #         flat_indices = torch.randperm(total_positions, device=pos.device)[:num_selected]
            #         atoms = flat_indices
            #         hessian_mask[atoms, :] = True
            #     N = batch_data["pos"].shape[1]
            #     batch_data["pos"]=batch_data["pos"].detach()
            #     batch_data["cell"]=batch_data["cell"].detach()

            #     grad_outputs = torch.zeros_like(hessian).to(forces.device)
            #     indices = torch.arange(N)
            #     grad_outputs[:,indices,indices,:,:] = torch.eye(3)[None,:,:].to(forces.device)

            #     for idx in range(N):
            #         # Create forward perturbation
            #         for j in range(3):
            #             if hessian_mask[idx,j] == 0:
            #                 continue
            #             import copy
            #             perturbed_batch_forward = copy.deepcopy(batch_data)
            #             perturbed_batch_forward["pos"] = perturbed_batch_forward["pos"] + 0.03 * grad_outputs[:,idx,:,j,:]

            #             hessian[:,idx,:,j,:] = (self.forward(perturbed_batch_forward)[1] - forces) / 0.03
            #     hessian=-hessian
            #     hessian = hessian.masked_fill(padding_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1), 0.0)
            #     hessian = hessian.masked_fill(padding_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1), 0.0)
            #---------------------------------------------------------------------------------
            mask_tensor = torch.ones_like(hessian, device=hessian.device, dtype=torch.bool)
            mask_tensor = mask_tensor.masked_fill(padding_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1), 0.0)
            mask_tensor = mask_tensor.masked_fill(padding_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1), 0.0)
            mask_tensor = mask_tensor & hessian_mask.unsqueeze(0).unsqueeze(2).unsqueeze(-1)
            mask_tensor = mask_tensor.masked_fill(batch_data["hessian_mask"].eq(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1), 0.0)

            pred_hessian = hessian[mask_tensor].reshape(-1).float()
            
            if pred_hessian.shape[0] > 0:
                hessian_loss = self.hessian_loss(
                    hessian[mask_tensor].reshape(-1).float(),
                    batch_data["hessian"][mask_tensor].reshape(-1).float(),
                )
                force_loss=0.0
                energy_loss=0.0
                stress_loss=0.0
            else:
                hessian_loss= 0.0

            hessian_loss = hessian_loss.mean()
        else:
            hessian_loss = 0.0

        if torch.any(batch_data["y"].isnan()):
            logger.warning("nan found in energy labels")
        stress_non_flag = False

        return ModelOutput(
            loss=(
                force_loss * self.force_loss_factor
                + energy_loss * self.energy_loss_factor
                + direct_force_loss
                + hessian_loss * self.hessian_loss_factor
            ).float()
            if not self.use_stress_loss or stress_non_flag
            else (
                force_loss * self.force_loss_factor
                + energy_loss * self.energy_loss_factor
                + stress_loss * self.stress_loss_factor
                + direct_force_loss
                + hessian_loss * self.hessian_loss_factor
            ).float(),
            log_output={
                "energy_loss": energy_loss,
                "energy_mae_loss": energy_mae_loss,
                "force_loss": force_loss,
                "force_mae_loss": force_mae_loss,
                "direct_force_loss": direct_force_loss,
                "stress_loss": stress_loss if self.use_stress_loss else 0,
                "stress_mae_loss": stress_mae_loss if self.use_stress_loss else 0,
                "hessian_loss": hessian_loss,
            },
            num_examples=bs,
        )

    def config_optimizer(self):
        pass



