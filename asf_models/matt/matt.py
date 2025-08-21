import torch.nn as nn
from asf_models.matt.cell_expand import radius_graph_pbc, get_pbc_distances

class MattNet(nn.Module):
    def __init__(self, args):
        super(MattNet, self).__init__()

    def forward(self, data):
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

    