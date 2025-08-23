import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.utils.rnn import pad_sequence
from .pbc2 import CellExpander

#################### Data Padding#########################
def padding_cell(data):
    pos = data.pos
    cell = data.cell.view(-1, 3, 3)
    natoms = data.natoms
    
    atoms_positions  = []
    atoms_numbers = []
    for i in range(len(data.ptr)-1):
        atoms_positions.append(pos[data.ptr[i]:data.ptr[i+1]])
        atoms_numbers.append(data.atomic_numbers[data.ptr[i]:data.ptr[i+1]])
    
    atoms_positions = pad_sequence(atoms_positions, batch_first=True, padding_value=0)
    atoms_numbers = pad_sequence(atoms_numbers, batch_first=True, padding_value=0)
    
    padding_mask = (atoms_numbers[:, :]).eq(0)
    
    return {
        'atoms_positions': atoms_positions,
        'atoms_numbers': atoms_numbers,
        'cell': cell,
        'natoms': natoms,
        'ptr': data.ptr,
        'padding_mask':padding_mask,
    }
##########################################################


class BesselBasis(torch.nn.Module):
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
            data=torch.tensor(torch.pi * torch.arange(1, num_radial + 1, dtype=torch.float32)),
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
       
        mul = self.mul(edge_types)
        bias = self.bias(edge_types)
        basis = mul * basis + bias
        return basis

def cal_distance(data):
    pos = data["pos"]
    n_node = pos.size(1)
    expand_pos = data["expand_pos"]
    expand_mask = data["expand_mask"]
    outcell_index = data["outcell_index"]
    
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
    return delta_pos, dist

def polynomial(dist: torch.Tensor, cutoff: float) -> torch.Tensor:
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

class Cet(nn.Module):
    def __init__(self, args):
        super(Cet, self).__init__()
        self.args = args
        self.atom_number_fn = nn.Embedding(args.max_numbers_of_atoms, args.lmax * args.embedding_dim, padding_idx=0,)
        self.cell_expander = CellExpander(cutoff = args.cutoff, pbc_multigraph_cutoff = args.cutoff,)
        
        self.radial_basis_fn = BesselBasis(args.num_basis, args.max_numbers_of_atoms**2 ,cutoff=args.cutoff)
        self.radial_proj = nn.Linear(args.num_basis, args.lmax * args.embedding_dim)
        
        

    def forward(self, x):
        # padding the data
        padding_dict = padding_cell(x)
        
        # expand the cell
        expand_dict = self.cell_expander.expand(
            padding_dict['atoms_positions'],
            torch.ones((padding_dict['cell'].shape[0],3),device=padding_dict['cell'].device),
            padding_dict['atoms_numbers'],
            padding_dict['cell'],
            padding_dict["natoms"],
            use_local_attention=True,
        )
        
        padding_mask = padding_dict['padding_mask']
        expand_mask = expand_dict['expand_mask']
        
        # encode the atom types
        atoms_numbers = padding_dict['atoms_numbers']
        atoms_embedding = self.atom_number_fn(atoms_numbers)
        
        # encode the radial edge features
        unicell_edge_type = atoms_numbers.unsqueeze(-1) * self.args.max_numbers_of_atoms + atoms_numbers.unsqueeze(1)
        outcell_index = expand_dict["outcell_index"]
        edge_type = torch.cat([
                unicell_edge_type,
                torch.gather(
                    unicell_edge_type,
                    dim=-1,
                    index=outcell_index.unsqueeze(1).repeat(1, padding_dict['atoms_positions'].size(1), 1),
                ),
            ],dim=-1,)
        
        vectors , distances = cal_distance(expand_dict)
        
        edge_radial_feature = self.radial_basis_fn(distances, edge_type.long())

        full_mask = torch.cat([padding_mask, expand_mask], dim=-1)
        edge_radial_feature = edge_radial_feature.masked_fill(full_mask.unsqueeze(1).unsqueeze(-1).to(torch.bool), 0.0)
        edge_radial_feature = edge_radial_feature.masked_fill(padding_mask.unsqueeze(-1).unsqueeze(-1).to(torch.bool), 0.0)
        edge_radial_feature = (torch.mul(edge_radial_feature, polynomial(distances,self.args.cutoff).unsqueeze(-1) ).float().type_as(edge_radial_feature))
        node_radial_feat = edge_radial_feature.sum(dim=-2)
        node_radial_feat = self.radial_proj(node_radial_feat)

        node_radial_feat = node_radial_feat.masked_fill_(padding_mask.unsqueeze(-1), 0.0)
        
        node_embedding = atoms_embedding + node_radial_feat
        
        return vectors