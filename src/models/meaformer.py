"""
MEAformer baseline re-implementation.
Reference: https://github.com/zjukg/MEAformer
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel


class RelationalGAT(nn.Module):
    """Simple RGAT encoder for relational structure."""

    def __init__(self, num_entities, num_relations, hidden_dim, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.entity_emb = nn.Embedding(num_entities, hidden_dim)
        self.relation_emb = nn.Embedding(num_relations, hidden_dim)
        self.layers = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])

    def forward(self, entity_ids, edge_index, edge_type):
        x = self.entity_emb(entity_ids)
        for attn, norm in zip(self.layers, self.norms):
            # Simplified: treat all neighbors as a sequence (full implementation uses sparse ops)
            out, _ = attn(x.unsqueeze(0), x.unsqueeze(0), x.unsqueeze(0))
            x = norm(x + out.squeeze(0))
        return x


class ModalityProjector(nn.Module):
    """Project raw modality features into the shared embedding space."""

    def __init__(self, in_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x):
        return self.proj(x)


class CrossModalTransformer(nn.Module):
    """
    Transformer that takes [V, T, R] as a 3-token sequence and models
    cross-modal interactions via self-attention.
    """

    def __init__(self, hidden_dim, num_heads=4, num_layers=1, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dropout=dropout,
            batch_first=True, dim_feedforward=hidden_dim * 4
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, v, t, r):
        # v, t, r: (B, D)
        seq = torch.stack([v, t, r], dim=1)   # (B, 3, D)
        out = self.transformer(seq)             # (B, 3, D)
        return out[:, 0], out[:, 1], out[:, 2] # v', t', r'


class MetaModalityPredictor(nn.Module):
    """
    Predict per-entity modality importance weights (the 'meta modality hybrid').
    Outputs a 3-dim softmax weight over [V, T, R].
    """

    def __init__(self, hidden_dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, v, t, r):
        combined = torch.cat([v, t, r], dim=-1)   # (B, 3D)
        weights = F.softmax(self.gate(combined), dim=-1)  # (B, 3)
        return weights


class MEAformer(nn.Module):
    """
    MEAformer: Multi-modal Entity Alignment Transformer.

    Args:
        num_entities_1/2: entity vocab size for each KG
        num_relations_1/2: relation vocab size for each KG
        img_dim: visual feature dimension (e.g. 2048 for ResNet, 768 for ViT-B)
        txt_dim: text feature dimension (768 for BERT-base)
        hidden_dim: shared embedding dimension
    """

    def __init__(
        self,
        num_entities_1: int,
        num_entities_2: int,
        num_relations_1: int,
        num_relations_2: int,
        img_dim: int = 2048,
        txt_dim: int = 768,
        hidden_dim: int = 256,
        gnn_layers: int = 2,
        tf_layers: int = 1,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Relational structure encoders
        self.rgat_1 = RelationalGAT(num_entities_1, num_relations_1, hidden_dim, num_heads, gnn_layers, dropout)
        self.rgat_2 = RelationalGAT(num_entities_2, num_relations_2, hidden_dim, num_heads, gnn_layers, dropout)

        # Modality projectors
        self.img_proj = ModalityProjector(img_dim, hidden_dim, dropout)
        self.txt_proj = ModalityProjector(txt_dim, hidden_dim, dropout)

        # Cross-modal transformer
        self.cross_modal_tf = CrossModalTransformer(hidden_dim, num_heads, tf_layers, dropout)

        # Meta modality weight predictor
        self.meta_predictor = MetaModalityPredictor(hidden_dim)

    def encode_entities(self, entity_ids, img_feats, txt_feats, rgat):
        """Produce fused entity representations."""
        r = rgat(entity_ids, edge_index=None, edge_type=None)  # (N, D)
        v = self.img_proj(img_feats)                            # (N, D)
        t = self.txt_proj(txt_feats)                            # (N, D)

        v_out, t_out, r_out = self.cross_modal_tf(v, t, r)

        weights = self.meta_predictor(v_out, t_out, r_out)     # (N, 3)
        w_v, w_t, w_r = weights[:, 0:1], weights[:, 1:2], weights[:, 2:3]

        fused = w_v * v_out + w_t * t_out + w_r * r_out        # (N, D)
        return F.normalize(fused, dim=-1)

    def forward(self, batch):
        """
        Returns normalized embeddings for KG1 and KG2 entities in the batch.
        Full graph encoding is expected to be done outside (see trainer).
        """
        raise NotImplementedError("Use encode_entities() with full-graph tensors via the trainer.")

    def compute_alignment_loss(self, emb1, emb2, pos_pairs, neg_pairs, margin=1.0):
        """Margin-based alignment loss."""
        pos_e1, pos_e2 = emb1[pos_pairs[:, 0]], emb2[pos_pairs[:, 1]]
        pos_dist = (pos_e1 - pos_e2).norm(dim=-1)

        neg_e1, neg_e2 = emb1[neg_pairs[:, 0]], emb2[neg_pairs[:, 1]]
        neg_dist = (neg_e1 - neg_e2).norm(dim=-1)

        loss = F.relu(margin + pos_dist - neg_dist).mean()
        return loss
