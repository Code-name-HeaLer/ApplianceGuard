import torch
import torch.nn as nn
import math
from src.models.layers import PositionalEncoding

class GCNTransformerAutoencoder(nn.Module):
    """
    GCN-Transformer Autoencoder for time-series reconstruction (Conditional AE).
    """
    def __init__(self,
                 input_dim: int,
                 seq_len: int,
                 n_clusters: int,         
                 d_model: int,            
                 nhead: int,              
                 num_encoder_layers: int, 
                 num_decoder_layers: int, 
                 dim_feedforward: int,    
                 gcn_out_dim: int,        
                 dropout: float = 0.1,
                 activation: str = 'gelu', 
                 use_deconvolution: bool = False, 
                 deconv_intermediate_channels: int = 64, 
                 deconv_kernel_size: int = 7,
                 deconv_stride: int = 1
                 ):
        super().__init__()

        if d_model % nhead != 0:
             raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")

        self.seq_len = seq_len
        self.input_dim = input_dim
        self.d_model = d_model
        self.gcn_out_dim = gcn_out_dim
        self.n_clusters = n_clusters
        self.use_deconvolution = use_deconvolution

        # --- Shared Components ---
        self.input_embed = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=max(seq_len + 100, 5000))

        # --- Encoder Components ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout,
            activation=activation, batch_first=True, norm_first=False
        )
        encoder_norm = nn.LayerNorm(d_model)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_encoder_layers, encoder_norm
        )

        # --- Context Combination Components ---
        decoder_memory_input_dim = d_model + gcn_out_dim
        self.memory_projection = nn.Linear(decoder_memory_input_dim, d_model)

        # --- Decoder Components ---
        decoder_layer = nn.TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout,
            activation=activation, batch_first=True, norm_first=False
        )
        decoder_norm = nn.LayerNorm(d_model)
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_decoder_layers, decoder_norm
        )

        # --- Final Output Layers ---
        if self.use_deconvolution:
            # 1. Permute [B, L, D] -> [B, D, L]
            self.pre_deconv_permute = lambda x: x.permute(0, 2, 1)
            deconv_padding = deconv_kernel_size // 2
            self.deconv_layer = nn.ConvTranspose1d(
                in_channels=d_model,
                out_channels=deconv_intermediate_channels,
                kernel_size=deconv_kernel_size,
                stride=deconv_stride,
                padding=deconv_padding
            )
            self.deconv_activation = nn.GELU()
            self.final_conv = nn.Conv1d(
                in_channels=deconv_intermediate_channels,
                out_channels=input_dim, 
                kernel_size=1
            )
            # 4. Permute back: [B, 1, L] -> [B, L, 1]
            self.post_deconv_permute = lambda x: x.permute(0, 2, 1)
        else:
            self.output_linear = nn.Linear(d_model, input_dim)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.input_embed.weight.data.uniform_(-initrange, initrange)
        self.memory_projection.bias.data.zero_()
        self.memory_projection.weight.data.uniform_(-initrange, initrange)
        if hasattr(self, 'output_linear'):
            self.output_linear.bias.data.zero_()
            self.output_linear.weight.data.uniform_(-initrange, initrange)

    def _generate_square_subsequent_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.full((sz, sz), float('-inf'), device=device), diagonal=1)

    def forward(self,
                src: torch.Tensor,           
                state_indices: torch.Tensor, 
                all_state_embeddings: torch.Tensor, 
                tgt: torch.Tensor = None,    
                src_key_padding_mask: torch.Tensor = None,
                tgt_key_padding_mask: torch.Tensor = None,
                memory_key_padding_mask: torch.Tensor = None
                ) -> torch.Tensor:
        
        device = src.device
        batch_size = src.size(0)
        seq_len = src.size(1)

        # --- Select GCN State Embedding ---
        batch_state_embeddings = all_state_embeddings[state_indices]

        # --- Transformer Encoder ---
        src_embedded = self.input_embed(src) * math.sqrt(self.d_model)
        src_embedded = self.pos_encoder(src_embedded)
        
        sequence_memory = self.transformer_encoder(
            src_embedded,
            src_key_padding_mask=src_key_padding_mask
        )

        # --- Combine Sequence Memory and GCN State ---
        expanded_state_embeddings = batch_state_embeddings.unsqueeze(1).expand(-1, seq_len, -1)
        combined_memory_features = torch.cat((sequence_memory, expanded_state_embeddings), dim=-1)
        projected_memory = self.memory_projection(combined_memory_features) 

        # --- Transformer Decoder ---
        if tgt is None: tgt = src 

        tgt_embedded = self.input_embed(tgt) * math.sqrt(self.d_model)
        tgt_embedded = self.pos_encoder(tgt_embedded)
        tgt_mask = self._generate_square_subsequent_mask(seq_len, device=device)

        decoder_output = self.transformer_decoder(
            tgt=tgt_embedded,
            memory=projected_memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask 
        )

        # --- Final Output Stage ---
        if self.use_deconvolution:
            permuted_output = self.pre_deconv_permute(decoder_output)
            deconv_out = self.deconv_activation(self.deconv_layer(permuted_output))
            final_out_permuted = self.final_conv(deconv_out)
            output = self.post_deconv_permute(final_out_permuted)
        else:
            output = self.output_linear(decoder_output)

        return output