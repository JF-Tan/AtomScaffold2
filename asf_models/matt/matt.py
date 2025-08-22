import torch.nn as nn
from asf_models.matt.cell_expand import radius_graph_pbc, get_pbc_distances
import math
import torch
from torch_scatter import scatter_sum, scatter_mean
import e3nn.o3 as o3

from .ffn_act import SO3_Linear, GateActivation, SpectralAtomwise
from .sa import EquAttention
from .norms import EquivariantRMSNormArraySphericalHarmonicsV2

class BesselBasis(nn.Module):
    def __init__(self, args):
        super(BesselBasis, self).__init__()
        self.cutoff = args.cutoff
        self.norm_const = math.sqrt(2 / (self.cutoff**3))
        self.frequencies = torch.nn.Parameter(
            data=torch.tensor(torch.pi * torch.arange(1, args.num_basis + 1, dtype=torch.float32)),
            requires_grad=True,
        )
        self.mul = nn.Embedding(args.max_atoms_number**2, 1, padding_idx=0)
        self.bias = nn.Embedding(args.max_atoms_number**2, 1, padding_idx=0)
        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1)
        
    def forward(self, edge_distancec, edge_types):
        x = edge_distancec / self.cutoff 
        # print(self.frequencies.shape)
        basis = (self.norm_const / (x.unsqueeze(-1)))*torch.sin(self.frequencies * x.unsqueeze(-1))
        mul = self.mul(edge_types)
        bias = self.bias(edge_types)
        basis = basis * mul + bias
        
        return basis

    
def envelope_fn(dist: torch.Tensor, cutoff: float) -> torch.Tensor:
        ratio = torch.div(dist, cutoff)
        result = (
            1
            - 6 * torch.pow(ratio, 5)
            + 15 * torch.pow(ratio, 4)
            - 10 * torch.pow(ratio, 3)
        )
        return torch.clamp(result, min=0.0)
    


class MattBlock(nn.Module):
    def __init__(self,args):
        super().__init__()
       
        self.equ_selfatt = EquAttention(args.edge_embedding_dim,args.edge_embedding_dim,args.n_heads,args.lmax)
        
        self.equ_norm1 =  EquivariantRMSNormArraySphericalHarmonicsV2(lmax=args.lmax,  num_channels=args.edge_embedding_dim, eps=1e-7)
        self.equ_norm2 =  EquivariantRMSNormArraySphericalHarmonicsV2(lmax=args.lmax,  num_channels=args.edge_embedding_dim, eps=1e-7)
      
       
        self.equ_ffn1 = SpectralAtomwise(args.edge_embedding_dim,args.edge_embedding_dim*4,lmax=args.lmax, mmax=args.lmax)
        

    
    def forward(self, equ_feat, envelope, attn_bias, expand_dict ):
        ################## SA ###################################
        # residual 
        equ_feat_res = equ_feat.clone()
        # norm 
        equ_feat = self.equ_norm1(equ_feat)
        
        
        # self attention 
        equ_feat = self.equ_selfatt(equ_feat,equ_feat,equ_feat,envelope,attn_bias,expand_dict)
        
        # add
        equ_feat = equ_feat + equ_feat_res
        # residual 
        equ_feat_res = equ_feat.clone()
        # norm 
        equ_feat = self.equ_norm2(equ_feat)
       
        # ffn
        equ_feat = self.equ_ffn1(equ_feat)
        
       
        # add 
        equ_feat = equ_feat + equ_feat_res

        return equ_feat

class MattNet(nn.Module):
    def __init__(self, args):
        super(MattNet, self).__init__()
        self.max_atoms_number = args.max_atoms_number
        self.atoms_number_fn = nn.Embedding(args.max_atoms_number, args.lmax * args.atom_embedding_dim)
        self.edge_radial_fn = BesselBasis(args)
        self.edge_radial_proj = nn.Linear(args.num_basis, args.lmax * args.edge_embedding_dim)
        self.sph_fn = o3.SphericalHarmonics(o3.Irreps.spherical_harmonics(args.lmax), normalize=True, normalization='component')
        self.edge_equ_proj = SO3_Linear(1, args.edge_embedding_dim,lmax=args.lmax, )
        self.cutoff = args.cutoff
        self.gate_inv_equ = GateActivation(lmax=args.lmax, mmax=args.lmax, num_channels=args.edge_embedding_dim)
        
        self.main_blocks = nn.ModuleList()
        for _ in range(args.num_blocks):
            self.main_blocks.append(
                MattBlock(args)
            )
            
            
    def forward(self, data):
        data.cell = data.cell.view(-1, 3, 3)
        edge_index, cell_offsets, neighbors = radius_graph_pbc(
                    data,
                    5.0,
                    max_num_neighbors_threshold=99999,
                )
        out = get_pbc_distances(
                        data.pos,
                        edge_index,
                        data.cell,
                        cell_offsets,
                        neighbors,
                        return_offsets=True,
                        return_distance_vec=True,
                    )
        ######## parse the input ##############
        cell = data.cell 
        positions = data.pos 
        atoms_numbers = data.atomic_numbers
        edge_index = out['edge_index']
        offsets = out['offsets']
        distances = out['distances']
        distance_vec = out['distance_vec']
        
        ######### atom type embedding ##################  
        atoms_embedding = self.atoms_number_fn(atoms_numbers)
        
        ######### edge inv embedding ##################  
        src_node_type = atoms_numbers[edge_index[0]]
        dst_node_type = atoms_numbers[edge_index[1]]
        egde_type = src_node_type * self.max_atoms_number + dst_node_type

        ENV = envelope_fn(distances, self.cutoff).unsqueeze(-1)
        edge_embedding = self.edge_radial_fn(distances, egde_type) *  ENV
        sumed_edge_embedding = scatter_sum(edge_embedding, edge_index[1],dim=0)
        sumed_edge_embedding = self.edge_radial_proj(sumed_edge_embedding)
        
        inv_feat = atoms_embedding + sumed_edge_embedding
        
        ######### edge equ embedding ##################  
        equ_feat = self.sph_fn(distance_vec) * ENV
        equ_feat = scatter_sum(equ_feat, edge_index[1], dim=0)
        equ_feat = self.edge_equ_proj(equ_feat.unsqueeze(-1))
        
        node_feat = self.gate_inv_equ(inv_feat, equ_feat)

        ######### interactions ##################
        for block in self.main_blocks:
            node_feat = block(node_feat, ENV, torch.ones_like(ENV), data)
        
        
        ######### head projection ##################
        return node_feat
        
    