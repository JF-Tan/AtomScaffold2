# -*- coding: utf-8 -*-

import numpy as np

import torch



@torch.jit.script

def mask_after_k_persample(n_sample: int, n_len: int, persample_k: torch.Tensor):

    assert persample_k.shape[0] == n_sample

    assert persample_k.max() <= n_len

    device = persample_k.device

    mask = torch.zeros([n_sample, n_len + 1], device=device)

    mask[torch.arange(n_sample, device=device), persample_k] = 1

    mask = mask.cumsum(dim=1)[:, :-1]

    return mask.type(torch.bool)



def create_2d_expand_mask(batch_size: int, max_num_node: int, max_expand_num_node: int, num_atoms: torch.Tensor, num_expand_atoms: torch.Tensor):

    assert num_atoms.size(0) == batch_size

    assert num_expand_atoms.size(0) == batch_size

    assert num_atoms.max() <= max_num_node

    assert num_expand_atoms.max() <= max_expand_num_node

    device = num_atoms.device

    mask = torch.zeros([batch_size, max_num_node, max_expand_num_node + 1], device=device)

    ones = torch.ones([batch_size, max_num_node, 1], dtype=mask.dtype, device=device)

    mask = mask.scatter(src=ones, dim=-1, index=num_expand_atoms[:, :, None])

    mask = mask.cumsum(dim=-1)[:, :, :-1]

    return mask.type(torch.bool)



class CellExpander:

    def __init__(

        self,

        cutoff=10.0,

        expanded_token_cutoff=256,

        pbc_expanded_num_cell_per_direction=10,

        pbc_multigraph_cutoff=5.0,

        backprop=False,

        original_token_count=True,

        use_three_body=False,

    ):

        self.cells = []

        for i in range(

            -pbc_expanded_num_cell_per_direction,

            pbc_expanded_num_cell_per_direction + 1,

        ):

            for j in range(

                -pbc_expanded_num_cell_per_direction,

                pbc_expanded_num_cell_per_direction + 1,

            ):

                for k in range(

                    -pbc_expanded_num_cell_per_direction,

                    pbc_expanded_num_cell_per_direction + 1,

                ):

                    if i == 0 and j == 0 and k == 0:

                        continue

                    self.cells.append([i, j, k])



        self.cells = torch.tensor(self.cells)



        self.cell_mask_for_pbc = self.cells != 0



        self.cutoff = cutoff



        self.expanded_token_cutoff = expanded_token_cutoff



        self.pbc_multigraph_cutoff = pbc_multigraph_cutoff



        self.pbc_expanded_num_cell_per_direction = pbc_expanded_num_cell_per_direction



        self.conflict_cell_offsets = []

        for i in range(-1, 2):

            for j in range(-1, 2):

                for k in range(-1, 2):

                    if i != 0 or j != 0 or k != 0:

                        self.conflict_cell_offsets.append([i, j, k])

        self.conflict_cell_offsets = torch.tensor(self.conflict_cell_offsets)  # 26 x 3



        conflict_to_consider = self.cells.unsqueeze(

            1

        ) - self.conflict_cell_offsets.unsqueeze(

            0

        )  # num_expand_cell x 26 x 3

        conflict_to_consider_mask = (

            ((conflict_to_consider * self.cells.unsqueeze(1)) >= 0)

            & (torch.abs(conflict_to_consider) <= self.cells.unsqueeze(1).abs())

        ).all(

            dim=-1

        )  # num_expand_cell x 26

        conflict_to_consider_mask &= (

            (conflict_to_consider <= pbc_expanded_num_cell_per_direction)

            & (conflict_to_consider >= -pbc_expanded_num_cell_per_direction)

        ).all(

            dim=-1

        )  # num_expand_cell x 26

        self.conflict_to_consider_mask = conflict_to_consider_mask



        self.backprop = backprop



        self.original_token_count = original_token_count



        self.use_three_body = use_three_body



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

        # unit_mask = torch.eye(dist.size(1), dist.size(2), device=dist.device).bool()

        # unit_mask = unit_mask.unsqueeze(0).expand(dist.size(0), -1, -1)

        # result[unit_mask] = 0.0

        return torch.clamp(result, min=0.0)



    def _get_cell_tensors(self, cell, use_local_attention):

        # fitler impossible offsets according to cell size and cutoff

        def _get_max_offset_for_dim(cell, dim):

            lattice_vec_0 = cell[:, dim, :]

            lattice_vec_1_2 = cell[

                :, torch.arange(3, dtype=torch.long, device=cell.device) != dim, :

            ]

            normal_vec = torch.cross(

                lattice_vec_1_2[:, 0, :], lattice_vec_1_2[:, 1, :], dim=-1

            )

            normal_vec = normal_vec / normal_vec.norm(dim=-1, keepdim=True)

            cutoff = self.pbc_multigraph_cutoff if use_local_attention else self.cutoff



            max_offset = int(

                torch.max(

                    torch.ceil(

                        cutoff

                        / torch.abs(torch.sum(normal_vec * lattice_vec_0, dim=-1))

                    )

                )

            )

            return max_offset



        max_offsets = []

        for i in range(3):

            try:

                max_offset = _get_max_offset_for_dim(cell, i)

            except Exception as e:

                print(f"{e} with cell {cell}")

                max_offset = self.pbc_expanded_num_cell_per_direction

            max_offsets.append(max_offset)

        max_offsets = torch.tensor(max_offsets, device=cell.device)

        self.cells = self.cells.to(device=cell.device)

        self.cell_mask_for_pbc = self.cell_mask_for_pbc.to(device=cell.device)

        mask = (self.cells.abs() <= max_offsets).all(dim=-1)

        selected_cell = self.cells[mask, :]

        return selected_cell, self.cell_mask_for_pbc[mask, :], mask



    def _get_conflict_mask(self, cell, pos, atoms):

        batch_size, max_num_atoms = pos.size()[:2]

        self.conflict_cell_offsets = self.conflict_cell_offsets.to(device=pos.device)

        self.conflict_to_consider_mask = self.conflict_to_consider_mask.to(

            device=pos.device

        )

        offset = torch.bmm(

            self.conflict_cell_offsets.unsqueeze(0)

            .repeat(batch_size, 1, 1)

            .to(dtype=cell.dtype),

            cell,

        )  # batch_size x 26 x 3

        expand_pos = (pos.unsqueeze(1) + offset.unsqueeze(2)).reshape(

            batch_size, -1, 3

        )  # batch_size x max_num_atoms x 3,

        # batch_size x 26 x 3 -> batch_size x (26 x max_num_atoms) x 3

        expand_dist = (pos.unsqueeze(2) - expand_pos.unsqueeze(1)).norm(

            p=2, dim=-1

        ) # 1e-6  # batch_size x max_num_atoms x (26 x max_num_atoms)



        expand_atoms = atoms.repeat(

            1, self.conflict_cell_offsets.size()[0]

        )  # batch_size x (26 x max_num_atoms)

        atoms_identical_mask = atoms.unsqueeze(-1) == expand_atoms.unsqueeze(

            1

        )  # batch_size x max_num_atoms x (26 x max_num_atoms)



        conflict_mask = (

            ((expand_dist < 1e-5) & atoms_identical_mask)

            .any(dim=1)

            .reshape(batch_size, -1, max_num_atoms)

        )  # batch_size x 26 x max_num_atoms

        all_conflict_mask = (

            torch.bmm(

                self.conflict_to_consider_mask.unsqueeze(0)

                .to(dtype=cell.dtype)

                .repeat(batch_size, 1, 1),

                conflict_mask.to(dtype=cell.dtype),

            )

            .long()

            .bool()

        )  # batch_size x num_expand_cell x 26,

        # batch_size x 26 x max_num_atoms ->

        # batch_size x num_expand_cell x max_num_atoms

        return all_conflict_mask



    def expand(

        self,

        pos,

        pbc,

        atoms,

        cell,

        natoms=None,

        use_local_attention=False,

        skip_wrap=False,

    ):

        """

        Args:

            pos: (B, T, 3) tensor, atom positions

            pbc: (B, 3) tensor, periodic boundary conditions

            atoms: (B, T) tensor, atom types

            cell: (B, 3, 3) tensor, unit cell

            natoms: (B,) tensor, number of atoms in each sample

            use_local_attention: bool, whether to use local attention

        """

        if self.backprop:

            if skip_wrap:

                return self._expand(

                    pos, pbc, atoms, cell, natoms, use_local_attention

                )

            else:

                pos_copy = pos.clone().requires_grad_(True)

                cell_copy = cell.clone()  # avoid replacing the original tensor



                strain = torch.zeros_like(cell_copy, device=cell_copy.device)  # B x 3 x 3

                strain.requires_grad_(True)

                strain = 0.5 * (strain + strain.transpose(-2,-1))



                strain_augment = strain.unsqueeze(1).expand(-1, pos_copy.size(1), -1, -1)

                cell_copy = torch.matmul(

                    cell_copy, (torch.eye(3, device=cell_copy.device).unsqueeze(0) + strain)

                )



                pos_copy = torch.einsum(

                    "bki, bkij -> bkj",

                    pos_copy,

                    (

                        torch.eye(3, device=pos_copy.device).unsqueeze(0).unsqueeze(0)

                        + strain_augment

                    ),

                )



                pbc_dict = self._expand(

                    pos_copy, pbc, atoms, cell_copy, natoms, use_local_attention

                )

                pbc_dict["strain"] = strain

        else:

            with torch.no_grad():

                pbc_dict = self._expand(

                    pos, pbc, atoms, cell, natoms, use_local_attention

                )

            pbc_dict["strain"] = None



        return pbc_dict



    def _expand(

        self,

        pos,

        pbc,

        atoms,

        cell,

        num_atoms,

        use_local_attention=True,

    ):

        with torch.set_grad_enabled(True):

            # pos = pos.float()

            # cell = cell.float()

            batch_size, max_num_atoms = pos.size()[:2]

            cell_tensor, cell_mask, selected_cell_mask = self._get_cell_tensors(

                cell, use_local_attention

            )



            if not use_local_attention:

                all_conflict_mask = self._get_conflict_mask(cell, pos, atoms)

                all_conflict_mask = all_conflict_mask[:, selected_cell_mask, :].reshape(

                    batch_size, -1

                )

            cell_tensor = (

                cell_tensor.unsqueeze(0).repeat(batch_size, 1, 1).to(dtype=cell.dtype)

            )

            num_expanded_cell = cell_tensor.size()[1]

            offset = torch.bmm(cell_tensor, cell)  # B x num_expand_cell x 3



            # ----------------du

            unique_offsets, inverse_indices, counts = torch.unique(

                offset, return_inverse=True, return_counts=True, dim=1

            )



            du_mask = torch.zeros_like(offset, dtype=torch.bool)



            inverse_indices_expanded = inverse_indices.unsqueeze(-1).expand(

                batch_size, num_expanded_cell, 3

            )

            counts_expanded = (

                counts[inverse_indices]

                .unsqueeze(-1)

                .expand(batch_size, num_expanded_cell, 3)

            )



            du_mask.scatter_(1, inverse_indices_expanded, counts_expanded > 1)

            du_mask = (counts_expanded > 1) & ~du_mask

            offset[du_mask] = 0.0



            # ---------------



            expand_pos = pos.unsqueeze(1) + offset.unsqueeze(

                2

            )  # B x num_expand_cell x T x 3

            expand_pos = expand_pos.view(

                batch_size, -1, 3

            )  # B x (num_expand_cell x T) x 3



            # eliminate duplicate atoms of expanded atoms,

            # comparing with the original unit cell

            expand_dist = (

                torch.norm(pos.unsqueeze(2) - expand_pos.unsqueeze(1), p=2, dim=-1)

                # 1e-6

            )  # B x T x (num_expand_cell x T)

            expand_atoms = atoms.repeat(1, num_expanded_cell)

            expand_atom_identical = atoms.unsqueeze(-1) == expand_atoms.unsqueeze(1)

            expand_mask = (expand_dist < self.cutoff) & (

                (expand_dist > 1e-5) | ~expand_atom_identical

            )  # B x T x (num_expand_cell x T)

            expand_mask = torch.masked_fill(

                expand_mask, atoms.eq(0).unsqueeze(-1), False

            )

            expand_mask = torch.sum(expand_mask, dim=1) > 0

            if not use_local_attention:

                expand_mask = expand_mask & (~all_conflict_mask)

            expand_mask = expand_mask & (

                ~(atoms.eq(0).repeat(1, num_expanded_cell))

            )  # B x (num_expand_cell x T)



            cell_mask = (

                torch.all(pbc.unsqueeze(1) >= cell_mask.unsqueeze(0), dim=-1)

                .unsqueeze(-1)

                .repeat(1, 1, max_num_atoms)

                .reshape(expand_mask.size())

            )  # B x (num_expand_cell x T)

            expand_mask &= cell_mask

            expand_len = torch.sum(expand_mask, dim=-1)



            if self.original_token_count:  # B

                threshold_num_expanded_token = torch.clamp(

                    self.expanded_token_cutoff - num_atoms, min=0

                )

            else:

                threshold_num_expanded_token = torch.full(

                    (batch_size,),

                    self.expanded_token_cutoff,

                    device=pos.device,

                    dtype=torch.long,

                )



            max_expand_len = torch.max(expand_len)



            # cutoff within expanded_token_cutoff tokens

            need_threshold = expand_len > threshold_num_expanded_token



            if need_threshold.any():

                min_expand_dist = expand_dist.masked_fill(expand_dist <= 1e-5, np.inf)

                expand_dist_mask = (

                    atoms.eq(0).unsqueeze(-1) | atoms.eq(0).unsqueeze(1)

                ).repeat(1, 1, num_expanded_cell)

                min_expand_dist = min_expand_dist.masked_fill_(expand_dist_mask, np.inf)

                min_expand_dist = min_expand_dist.masked_fill_(

                    ~cell_mask.unsqueeze(1), np.inf

                )

                min_expand_dist = torch.min(min_expand_dist, dim=1)[0]



                need_threshold_distances = min_expand_dist[

                    need_threshold

                ]  # B x (num_expand_cell x T)

                threshold_num_expanded_token = threshold_num_expanded_token[

                    need_threshold

                ]

                threshold_dist = torch.sort(

                    need_threshold_distances, dim=-1, descending=False

                )[0]



                threshold_dist = torch.gather(

                    threshold_dist, 1, threshold_num_expanded_token.unsqueeze(-1)

                )



                new_expand_mask = min_expand_dist[need_threshold] < threshold_dist

                expand_mask[need_threshold] &= new_expand_mask

                expand_len = torch.sum(expand_mask, dim=-1)

                max_expand_len = torch.max(expand_len)



            outcell_index = torch.zeros(

                [batch_size, max_expand_len], dtype=torch.long, device=pos.device

            )

            expand_pos_compressed = torch.zeros(

                [batch_size, max_expand_len, 3], dtype=pos.dtype, device=pos.device

            )

            outcell_all_index = torch.arange(

                max_num_atoms, dtype=torch.long, device=pos.device

            ).repeat(num_expanded_cell)

            for i in range(batch_size):

                outcell_index[i, : expand_len[i]] = outcell_all_index[expand_mask[i]]

                # assert torch.all(outcell_index[i, :expand_len[i]] < natoms[i])

                expand_pos_compressed[i, : expand_len[i], :] = expand_pos[

                    i, expand_mask[i], :

                ]



            if use_local_attention:

                n_node = pos.shape[1]

                expand_pos = torch.cat([pos, expand_pos_compressed], dim=1)

                expand_n_node = expand_pos.shape[1]

                uni_delta_pos = pos.unsqueeze(2) - expand_pos.unsqueeze(1)



                unit_mask = uni_delta_pos.norm(dim=-1) < 1e-6

                unit_mask = unit_mask.unsqueeze(-1).expand(-1, -1, -1, 3)

                uni_delta_pos[unit_mask] = torch.abs(uni_delta_pos[unit_mask]) + 20.0



                total_dist = uni_delta_pos.norm(dim=-1).view(-1, n_node, expand_n_node)



                local_attention_weight = self.polynomial(

                    total_dist,

                    cutoff=self.pbc_multigraph_cutoff,

                )

                is_periodic = pbc.any(dim=-1)

                local_attention_weight = local_attention_weight.masked_fill(

                    ~is_periodic.unsqueeze(-1).unsqueeze(-1), 1.0

                )

                expand_mask = mask_after_k_persample(

                    batch_size, max_expand_len, expand_len

                )

                full_mask = torch.cat([atoms.eq(0), expand_mask], dim=-1)

                local_attention_weight = local_attention_weight.masked_fill(

                    atoms.eq(0).unsqueeze(-1), 1.0

                )

                local_attention_weight = local_attention_weight.masked_fill(

                    full_mask.unsqueeze(1), 0.0

                )

                pbc_expand_batched = {

                    "expand_pos": expand_pos_compressed,

                    "outcell_index": outcell_index,

                    "expand_mask": expand_mask,

                    "local_attention_weight": local_attention_weight,

                }

                # print(local_attention_weight.shape[2])

            else:

                pbc_expand_batched = {

                    "expand_pos": expand_pos_compressed,

                    "outcell_index": outcell_index,

                    "expand_mask": mask_after_k_persample(

                        batch_size, max_expand_len, expand_len

                    ),

                    "local_attention_weight": None,

                }



            expand_pos_no_offset = torch.gather(

                pos, dim=1, index=outcell_index.unsqueeze(-1)

            )

            offset = expand_pos_compressed - expand_pos_no_offset



            pbc_expand_batched["pos"] = pos

            pbc_expand_batched["cell"] = cell

            return pbc_expand_batched



    def expand_cpu(

        self,

        pos,

        pbc,

        num_atoms,

        atoms,

        cell,

        use_local_attention=True,

        use_per_atom_outcell_index=False,

    ):

        with torch.set_grad_enabled(True):

            pos = pos.float()

            cell = cell.float()

            batch_size, max_num_atoms = pos.size()[:2]

            cell_tensor, cell_mask, selected_cell_mask = self._get_cell_tensors(

                cell, use_local_attention

            )



            encode_factor = self.pbc_expanded_num_cell_per_direction * 2 + 1

            cell_index = cell_tensor + self.pbc_expanded_num_cell_per_direction

            cell_index = cell_index[:, 0] * encode_factor * encode_factor + cell_index[:, 1] * encode_factor + cell_index[:, 2]



            cell_tensor = (

                cell_tensor.unsqueeze(0).repeat(batch_size, 1, 1).to(dtype=cell.dtype)

            )



            num_expanded_cell = cell_tensor.size()[1]

            offset = torch.bmm(cell_tensor, cell)  # B x num_expand_cell x 3

            expand_pos = pos.unsqueeze(1) + offset.unsqueeze(

                2

            )  # B x num_expand_cell x T x 3

            expand_pos = expand_pos.view(

                batch_size, -1, 3

            )  # B x (num_expand_cell x T) x 3



            # eliminate duplicate atoms of expanded atoms, comparing with the original unit cell

            expand_dist = torch.norm(

                pos.unsqueeze(2) - expand_pos.unsqueeze(1), p=2, dim=-1

            )  # B x T x (num_expand_cell x T)

            # expand_atoms = atoms.repeat(1, num_expanded_cell)

            # expand_atom_identical = atoms.unsqueeze(-1) == expand_atoms.unsqueeze(1)

            cutoff = self.cutoff if (not use_local_attention) or (not use_per_atom_outcell_index) else self.pbc_multigraph_cutoff

            expand_mask = (expand_dist < cutoff)

            # & (

            #     # (expand_dist > 1e-5) | ~expand_atom_identical

            #     ~expand_atom_identical

            # )  # B x T x (num_expand_cell x T)

            expand_mask = torch.masked_fill(

                expand_mask, atoms.eq(0).unsqueeze(-1), False

            )

            if not use_per_atom_outcell_index:

                expand_mask = torch.sum(expand_mask, dim=1) > 0

            # if not use_local_attention:

            #     expand_mask = expand_mask & (~all_conflict_mask)

            if use_per_atom_outcell_index:

                expand_mask = expand_mask & (

                    ~(atoms.eq(0).repeat(1, num_expanded_cell)[:, None, :])

                )  # B x T x (num_expand_cell x T)

            else:

                expand_mask = expand_mask & (

                    ~(atoms.eq(0).repeat(1, num_expanded_cell))

                )  # B x (num_expand_cell x T)



            if use_per_atom_outcell_index:

                cell_mask = (

                    torch.all(pbc.unsqueeze(1) >= cell_mask.unsqueeze(0), dim=-1)

                    .unsqueeze(-1)

                    .repeat(1, 1, max_num_atoms)

                    .reshape([expand_mask.size(0), expand_mask.size(2)])

                )  # B x (num_expand_cell x T)

            else:

                cell_mask = (

                    torch.all(pbc.unsqueeze(1) >= cell_mask.unsqueeze(0), dim=-1)

                    .unsqueeze(-1)

                    .repeat(1, 1, max_num_atoms)

                    .reshape(expand_mask.size())

                )  # B x (num_expand_cell x T)



            if use_per_atom_outcell_index:

                expand_mask &= cell_mask[:, None, :]

            else:

                expand_mask &= cell_mask



            expand_len = torch.sum(expand_mask, dim=-1)



            threshold_num_expanded_token = torch.clamp(

                self.expanded_token_cutoff - num_atoms, min=0

            )



            max_expand_len = torch.max(expand_len)



            #TODO(shiyu): fix threshold with per atom mask, for now no limit of expanded atoms will be applied when use_per_atom_outcell_index

            # cutoff within expanded_token_cutoff tokens

            need_threshold = (expand_len > threshold_num_expanded_token)

            if need_threshold.any() and (not use_per_atom_outcell_index):

                min_expand_dist = expand_dist # .masked_fill(expand_dist <= 1e-5, np.inf)

                expand_dist_mask = (

                    atoms.eq(0).unsqueeze(-1) | atoms.eq(0).unsqueeze(1)

                ).repeat(1, 1, num_expanded_cell)

                min_expand_dist = min_expand_dist.masked_fill_(expand_dist_mask, np.inf)

                min_expand_dist = min_expand_dist.masked_fill_(

                    ~cell_mask.unsqueeze(1), np.inf

                )

                min_expand_dist = torch.min(min_expand_dist, dim=1)[0]



                need_threshold_distances = min_expand_dist[

                    need_threshold

                ]  # B x (num_expand_cell x T)

                threshold_num_expanded_token = threshold_num_expanded_token[

                    need_threshold

                ]

                threshold_dist = torch.sort(

                    need_threshold_distances, dim=-1, descending=False

                )[0]



                threshold_dist = torch.gather(

                    threshold_dist, 1, threshold_num_expanded_token.unsqueeze(-1)

                )



                new_expand_mask = min_expand_dist[need_threshold] < threshold_dist

                expand_mask[need_threshold] &= new_expand_mask

                expand_len = torch.sum(expand_mask, dim=-1)

                max_expand_len = torch.max(expand_len)



            if use_per_atom_outcell_index:

                outcell_index = torch.zeros(

                    [batch_size, max_num_atoms, max_expand_len], dtype=torch.long, device=pos.device

                )

                outcell_cell_index = torch.zeros(

                    [batch_size, max_num_atoms, max_expand_len], dtype=torch.long, device=pos.device

                )

            else:

                outcell_index = torch.zeros(

                    [batch_size, max_expand_len], dtype=torch.long, device=pos.device

                )

                outcell_cell_index = torch.zeros(

                    [batch_size, max_expand_len], dtype=torch.long, device=pos.device

                )

            outcell_all_index = torch.arange(

                max_num_atoms, dtype=torch.long, device=pos.device

            ).repeat(num_expanded_cell)



            cell_all_index = cell_index[:, None].repeat(1, max_num_atoms).reshape(-1)



            for i in range(batch_size):

                if use_per_atom_outcell_index:

                    for j in range(int(num_atoms[i])):

                        outcell_index[i, j, : expand_len[i, j]] = outcell_all_index[expand_mask[i, j]]

                        outcell_cell_index[i, j, : expand_len[i, j]] = cell_all_index[expand_mask[i, j]]

                else:

                    outcell_index[i, : expand_len[i]] = outcell_all_index[expand_mask[i]]

                    outcell_cell_index[i, : expand_len[i]] = cell_all_index[expand_mask[i]]



            pbc_expand_batched = {

                "outcell_index": outcell_index,

                "expand_len": expand_len,

                "outcell_cell_index": outcell_cell_index,

            }



            return pbc_expand_batched

       

    def wrap_input(self, pos,cell):

        pos.requires_grad_(True)

        batch_size = pos.size(0)

        device = pos.device

        dtype = pos.dtype



        strain = torch.zeros([batch_size, 3, 3], device=device, dtype=dtype)

        strain.requires_grad_(True)

        strain = 0.5 * (strain + strain.transpose(-2,-1))

        strain_augment = strain.unsqueeze(1).expand(-1, pos.size(1), -1, -1)



        cell = torch.matmul(

            cell, torch.eye(3, device=device, dtype=dtype)[None, :, :] + strain

        )

        pos = torch.einsum(

            "bki, bkij -> bkj",

            pos,

            (torch.eye(3, device=device)[None, None, :, :] + strain_augment),

        )

        return pos,cell,strain



    def construct_pbc_expand_batched(self, expand_len, num_atoms, outcell_index, outcell_cell_index, pos, cell, pair_token_type,use_local_attention, use_per_atom_outcell_index):

        batch_size = pos.size(0)

        max_num_atoms = pos.size(1)

        max_expand_len = int(torch.max(expand_len))

        pos, cell, strain = self.wrap_input(pos, cell)

        if use_per_atom_outcell_index:

            expand_mask = create_2d_expand_mask(batch_size, max_num_atoms, max_expand_len, num_atoms, expand_len)

            outcell_cell_matrix = torch.zeros([batch_size, max_num_atoms, max_expand_len, 3], dtype=torch.int64, device=pos.device)

            encode_factor = self.pbc_expanded_num_cell_per_direction * 2 + 1

            outcell_cell_matrix[:, :, :, 0] = (outcell_cell_index // (encode_factor ** 2)) - self.pbc_expanded_num_cell_per_direction

            outcell_cell_matrix[:, :, :, 1] = ((outcell_cell_index % (encode_factor ** 2)) // encode_factor) - self.pbc_expanded_num_cell_per_direction

            outcell_cell_matrix[:, :, :, 2] = (outcell_cell_index % encode_factor) - self.pbc_expanded_num_cell_per_direction

            offset = torch.matmul(outcell_cell_matrix.to(dtype=cell.dtype), cell[:, None, :, :]) # B x num_atoms x num_expand_atoms x 3, B x 1 x 3 x 3 -> B x num_atoms x num_expand_atoms x 3

            expand_pos = torch.gather(pos[:, None, :, :].repeat(1, max_num_atoms, 1, 1), dim=2, index=outcell_index[:, :, :, None].repeat(1, 1, 1, 3)) # B x num_atoms x num_expand_atoms x 3

            expand_pos = expand_pos + offset # B x num_atoms x num_expand_atoms x 3

            expand_pos = expand_pos.masked_fill(expand_mask[:, :, :, None], 0.0) # B x num_atoms x num_expand_atoms x 3

            full_pos = torch.cat([pos[:, None, :, :].repeat(1, max_num_atoms, 1, 1), expand_pos], dim=2) # B x num_atoms x (num_atoms + num_expand_atoms) x 3

            full_dist = (pos[:, :, None, :] - full_pos)

           

            unit_mask = full_dist.norm(dim=-1) < 1e-6

            unit_mask = unit_mask.unsqueeze(-1).expand(-1, -1, -1, 3)

            full_dist[unit_mask] = torch.abs(full_dist[unit_mask]) + 20.0



            full_dist = full_dist.norm(p=2, dim=-1)

            if use_local_attention:

                local_attention_weight = self.polynomial(full_dist, self.pbc_multigraph_cutoff)

                padding_mask = mask_after_k_persample(batch_size, max_num_atoms, num_atoms)

                full_mask = torch.cat([padding_mask[:, :, None] | padding_mask[:, None, :], expand_mask], dim=-1)

                local_attention_weight = local_attention_weight.masked_fill(full_mask, 0.0)

                local_attention_weight = local_attention_weight.masked_fill(padding_mask[:, :, None], 0.0)

            else:

                local_attention_weight = None

            expand_pair_token_type = torch.gather(

                pair_token_type, # B x num_atoms x num_atoms x k

                dim=2,

                index=outcell_index

                .unsqueeze(-1)

                .repeat(1, 1, 1, pair_token_type.size()[-1]),

            )

            expand_node_type_edge = torch.cat(

                [pair_token_type, expand_pair_token_type], dim=2

            ) # B x num_atoms x (num_atoms + num_expand_atoms) x k

        else:

            expand_mask = mask_after_k_persample(batch_size, max_expand_len, expand_len)

            outcell_cell_matrix = torch.zeros([batch_size, max_expand_len, 3], dtype=torch.int64, device=pos.device)

            encode_factor = self.pbc_expanded_num_cell_per_direction * 2 + 1

            outcell_cell_matrix[:, :, 0] = (outcell_cell_index // (encode_factor ** 2)) - self.pbc_expanded_num_cell_per_direction

            outcell_cell_matrix[:, :, 1] = ((outcell_cell_index % (encode_factor ** 2)) // encode_factor) - self.pbc_expanded_num_cell_per_direction

            outcell_cell_matrix[:, :, 2] = (outcell_cell_index % encode_factor) - self.pbc_expanded_num_cell_per_direction

            offset = torch.bmm(outcell_cell_matrix.to(dtype=cell.dtype), cell) # B x num_expand_atoms x 3, B x 3 x 3 -> B x num_expand_cell x 3

            expand_pos = torch.gather(pos, dim=1, index=outcell_index[:, :, None].repeat(1, 1, 3))

            expand_pos = expand_pos + offset

            expand_pos = expand_pos.masked_fill(expand_mask[:, :, None], 0.0)

            full_pos = torch.cat([pos, expand_pos], dim=1)

            full_dist = (pos[:, :, None, :] - full_pos[:, None, :, :])

           

            unit_mask = full_dist.norm(dim=-1) < 1e-6

            unit_mask = unit_mask.unsqueeze(-1).expand(-1, -1, -1, 3)

            full_dist[unit_mask] = torch.abs(full_dist[unit_mask]) + 20.0

           

            full_dist = full_dist.norm(p=2, dim=-1)

            if use_local_attention:

                local_attention_weight = self.polynomial(full_dist, self.pbc_multigraph_cutoff)

                padding_mask = mask_after_k_persample(batch_size, max_num_atoms, num_atoms)

                full_mask = torch.cat([padding_mask, expand_mask], dim=-1)

                local_attention_weight = local_attention_weight.masked_fill(full_mask[:, None, :], 0.0)

                local_attention_weight = local_attention_weight.masked_fill(padding_mask[:, :, None], 0.0)

            else:

                local_attention_weight = None

            expand_pair_token_type = torch.gather(

                pair_token_type,

                dim=2,

                index=outcell_index.unsqueeze(1)

                .unsqueeze(-1)

                .repeat(1, max_num_atoms, 1, pair_token_type.size()[-1]),

            )

            expand_node_type_edge = torch.cat(

                [pair_token_type, expand_pair_token_type], dim=2

            )



        pbc_expand_batched = {

            "pos": pos,

            "cell": cell,

            "expand_pos": expand_pos,

            "outcell_index": outcell_index,

            "expand_mask": expand_mask,

            "local_attention_weight": local_attention_weight,

            "expand_node_type_edge": expand_node_type_edge,

            "strain": strain,

        }

        return pbc_expand_batched

