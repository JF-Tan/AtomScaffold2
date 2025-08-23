# -*- coding: utf-8 -*-

from dataclasses import asdict, dataclass



@dataclass

class GraphormerConfig:

    model_type: str = "graphormer"

    num_classes: int = 1

    encoder_attention_heads: int = 32

    encoder_ffn_embed_dim: int = 768

    encoder_embed_dim: int = 768

    encoder_layers: int = 24

    num_pred_attn_layer: int = 4

    num_3d_bias_kernel: int = 128

    max_length: int = 1024

    pbc_expanded_token_cutoff: int = 512

    pbc_expanded_num_cell_per_direction: int = 10

    multi_hop_max_dist: int = 20



    droppath_prob: float = 0.0

    act_dropout: float = 0.0

    attn_dropout: float = 0.0

    dropout: float = 0.0

    sandwich_ln: bool = True

    noise_scale: float = 0.2

    mask_ratio: float = 0.5

    d_tilde: float = 1.0

    pbc_cutoff: float = 40.0

    pbc_multigraph_cutoff: float = 5.0



    data_path: str = ""

    dataset_names: str = ""

    loadcheck_path: str = ""



    add_3d: bool = False

    no_2d: bool = False

    ft: bool = False

    infer: bool = False

    use_pbc: bool = False



    # TM and ddpm params

    transformer_m_pretrain: bool = True

    mode_prob: str = "0.6,0.2,0.2"

    num_timesteps: int = 5000

    num_timesteps_stepsize: int = -1

    ddpm_beta_start: float = 1e-7

    ddpm_beta_end: float = 0.002

    ddpm_schedule: str = "sigmoid"

    noise_mode: str = "const"

    num_edges: int = 25600

    num_atom_features: int = 5120

    ###########################################################################################

    ####### THESE are from graphormer_base_architecture, is confict, follow upper default vaule.

    #

    #

    #     dropout: 0.0

    attention_dropout: float = 0.0

    #     act_dropout = 0.0

    #

    #     encoder_ffn_embed_dim = 4096

    #     encoder_layers = 6

    #     encoder_attention_heads = 8

    #

    #     encoder_embed_dim = 1024

    share_encoder_input_output_embed: bool = False

    encoder_learned_pos: bool = False

    no_token_positional_embeddings: bool = False

    num_segments: int = 2

    #

    sentence_class_num: int = 2

    sent_loss: bool = False

    #

    apply_bert_init: bool = False

    #

    activation_fn: str = "relu"

    pooler_activation_fn: str = "tanh"

    encoder_normalize_before: bool = False

    #

    #     sandwich_ln = False

    #     droppath_prob = 0.0

    #

    # add

    atom_loss_coeff: float = 1.0

    pos_loss_coeff: float = 1.0

    y_2d_loss_coeff: float = 1.0

    #

    max_positions: int = 512

    num_atoms: int = 512 * 9

    #     num_edges = 512 * 3

    num_in_degree: int = 512

    num_out_degree: int = 512

    num_spatial: int = 512

    num_edge_dis: int = 128

    #     multi_hop_max_dist:int = 5

    edge_type: str = "multi_hop"

    #

    layerdrop: float = 0.0

    apply_graphormer_init: bool = True

    pre_layernorm: bool = False

    #

    #

    #

    #     encoder_layers = 12

    #     encoder_attention_heads = 12

    #     encoder_ffn_embed_dim = 768

    #     encoder_embed_dim = 768

    #     dropout = 0.0

    #     attention_dropout = 0.0

    #     act_dropout = 0.0

    #     activation_fn ="gelu"

    #     encoder_normalize_before = True

    #     share_encoder_input_output_embed = False

    #     no_token_positional_embeddings = False

    #

    #

    #



    def __init__(

        self,

        args,

        **kwargs,

    ):

        graphormer_base_architecture(args)



        if not hasattr(args, "max_positions"):

            try:

                args.max_positions = args.tokens_per_sample

            except:

                args.max_positions = args.max_nodes



        # set attributes of args to self

        for k, v in asdict(self).items():

            if hasattr(args, k):

                setattr(self, k, getattr(args, k))



        self.num_atoms = args.num_atoms

        self.num_in_degree = args.num_in_degree

        self.num_out_degree = args.num_out_degree

        self.num_edges = args.num_edges

        self.num_spatial = args.num_spatial

        self.num_edge_dis = args.num_edge_dis

        self.edge_type = args.edge_type

        self.multi_hop_max_dist = args.multi_hop_max_dist

        self.num_encoder_layers = args.encoder_layers

        self.embedding_dim = args.encoder_embed_dim

        self.ffn_embedding_dim = args.encoder_ffn_embed_dim

        self.num_attention_heads = args.encoder_attention_heads

        self.dropout = args.dropout

        self.attention_dropout = args.attention_dropout

        self.activation_dropout = args.act_dropout

        self.layerdrop = args.layerdrop

        self.max_seq_len = args.max_positions

        self.num_segments = args.num_segments

        self.use_position_embeddings = not args.no_token_positional_embeddings

        self.encoder_normalize_before = args.encoder_normalize_before

        self.apply_bert_init = args.apply_bert_init

        self.activation_fn = args.activation_fn

        self.learned_pos_embedding = args.encoder_learned_pos

        self.sandwich_ln = args.sandwich_ln

        self.droppath_prob = args.droppath_prob

        self.add_3d = args.add_3d

        self.num_3d_bias_kernel = args.num_3d_bias_kernel

        self.no_2d = args.no_2d

        self.args = args



def base_architecture(args):

    args.dropout = getattr(args, "dropout", 0.0)

    args.attention_dropout = getattr(args, "attention_dropout", 0.0)

    args.act_dropout = getattr(args, "act_dropout", 0.0)



    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)

    args.encoder_layers = getattr(args, "encoder_layers", 6)

    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)



    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)

    args.share_encoder_input_output_embed = getattr(

        args, "share_encoder_input_output_embed", False

    )

    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)

    args.no_token_positional_embeddings = getattr(

        args, "no_token_positional_embeddings", False

    )

    args.num_segments = getattr(args, "num_segments", 2)



    args.sentence_class_num = getattr(args, "sentence_class_num", 2)

    args.sent_loss = getattr(args, "sent_loss", False)



    args.apply_bert_init = getattr(args, "apply_bert_init", False)



    args.activation_fn = getattr(args, "activation_fn", "relu")

    args.pooler_activation_fn = getattr(args, "pooler_activation_fn", "tanh")

    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)



    args.sandwich_ln = getattr(args, "sandwich_ln", False)

    args.droppath_prob = getattr(args, "droppath_prob", 0.0)



    # add

    args.atom_loss_coeff = getattr(args, "atom_loss_coeff", 1.0)

    args.pos_loss_coeff = getattr(args, "pos_loss_coeff", 1.0)

    args.y_2d_loss_coeff = getattr(args, "y_2d_loss_coeff", 1.0)



    args.max_positions = getattr(args, "max_positions", 512)

    args.num_atoms = getattr(args, "num_atoms", 512 * 9)

    args.num_edges = getattr(args, "num_edges", 512 * 3)

    args.num_in_degree = getattr(args, "num_in_degree", 512)

    args.num_out_degree = getattr(args, "num_out_degree", 512)

    args.num_spatial = getattr(args, "num_spatial", 512)

    args.num_edge_dis = getattr(args, "num_edge_dis", 128)

    args.multi_hop_max_dist = getattr(args, "multi_hop_max_dist", 5)

    args.edge_type = getattr(args, "edge_type", "multi_hop")



    args.layerdrop = getattr(args, "layerdrop", 0.0)



    # args.q_noise = getattr(args, "q_noise", 0.0)

    # args.qn_block_size = getattr(args, "qn_block_size", 8)

    # args.freeze_embeddings = getattr(args, "freeze_embeddings", False)

    # args.n_trans_layers_to_freeze = getattr(args, "n_trans_layers_to_freeze", 0)

    # args.export = getattr(args, "export", False)

    # args.export = getattr(args, "export", False)

    # args.traceable = getattr(args, "traceable", False)

    # args.embed_scale = getattr(args, "embed_scale", None)

    # args.apply_bert_init = getattr(args, "apply_bert_init", False)






def graphormer_base_architecture(args):

    args.encoder_layers = getattr(args, "encoder_layers", 12)

    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 12)

    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 768)

    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)

    args.dropout = getattr(args, "dropout", 0.0)

    args.attention_dropout = getattr(args, "attention_dropout", 0.0)

    args.act_dropout = getattr(args, "act_dropout", 0.0)

    args.activation_fn = getattr(args, "activation_fn", "gelu")

    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)

    args.apply_graphormer_init = getattr(args, "apply_graphormer_init", True)

    args.share_encoder_input_output_embed = getattr(

        args, "share_encoder_input_output_embed", False

    )

    args.no_token_positional_embeddings = getattr(

        args, "no_token_positional_embeddings", False

    )

    args.pre_layernorm = getattr(args, "pre_layernorm", False)

    base_architecture(args)



