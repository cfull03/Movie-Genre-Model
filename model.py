import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttentionPooling(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.lin1 = nn.Linear(d_model, d_model)
        self.lin2 = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        m = attn_mask.bool()
        scores = self.lin2(torch.tanh(self.lin1(x))).squeeze(-1)
        scores = scores.masked_fill(~m, float('-inf'))
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        return (weights * x).sum(dim=1)

class TextClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_heads: int = 8,
        max_seq_length: int = 512,
        num_attention_layers: int = 4,
        feedforward_dim: int = 256,
        num_dropout_samples: int = 8,
        mc_dropout_enabled: bool = True,
    ) -> None:
        super().__init__()
        self.num_dropout_samples = num_dropout_samples
        self.mc_dropout_enabled = mc_dropout_enabled

        self.emb = nn.Embedding(vocab_size, emb_dim)
        nn.init.normal_(self.emb.weight, std=0.02)

        self.positional_encoding = nn.Parameter(torch.zeros(1, max_seq_length, emb_dim))
        nn.init.normal_(self.positional_encoding, std=0.02)
        self.input_drop = nn.Dropout(0.3)

        transformer_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=0.3,
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            transformer_layer,
            num_layers=num_attention_layers
        )

        self.attention_pooling = SelfAttentionPooling(emb_dim)

        self.fc1 = nn.Linear(emb_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.res_proj = nn.Identity() if (hidden_dim // 2) == emb_dim else nn.Linear(emb_dim, hidden_dim // 2)
        self.layernorm_final = nn.LayerNorm(hidden_dim // 2)
        self.text_to_label_space = nn.Linear(hidden_dim // 2, emb_dim)

        self.label_embeddings = nn.Embedding(num_classes, emb_dim)
        nn.init.normal_(self.label_embeddings.weight, std=0.02)
        self.logit_bias = nn.Parameter(torch.zeros(num_classes))
        self.logit_scale = nn.Parameter(torch.tensor(10.0))

        self.dropout = nn.Dropout(0.5)

    @torch.no_grad()
    def enable_mc_dropout(self, enabled: bool = True) -> None:
        self.mc_dropout_enabled = enabled
        if enabled:
            self.dropout.train()
        else:
            self.dropout.eval()

    def _encode(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask_bool = attention_mask.bool()
        x = self.emb(x) * math.sqrt(self.emb.embedding_dim)
        x = x + self.positional_encoding[:, :x.size(1), :]
        x = self.input_drop(x)
        x = self.transformer_encoder(x, src_key_padding_mask=~mask_bool)
        x = self.attention_pooling(x, mask_bool)
        return x

    def _trunk(self, pooled: torch.Tensor) -> torch.Tensor:
        residual = pooled
        x = F.relu(self.fc1(pooled))
        x = F.relu(self.fc2(x))
        residual = self.res_proj(residual)
        x = self.layernorm_final(x + residual)
        x_proj = self.text_to_label_space(x)
        return x_proj

    def _logits_from_proj(self, x_proj: torch.Tensor) -> torch.Tensor:
        label_w = F.normalize(self.label_embeddings.weight, dim=-1)
        text_z  = F.normalize(x_proj, dim=-1)
        logits  = self.logit_scale * (text_z @ label_w.T) + self.logit_bias
        return logits

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        debugging: bool = False
    ) -> torch.Tensor:
        if debugging:
            print(f"input_ids: {x.shape}, attention_mask: {attention_mask.shape}")

        pooled = self._encode(x, attention_mask)
        if debugging:
            print(f"after encoder+pool: {pooled.shape}")

        x_proj = self._trunk(pooled)
        if debugging:
            print(f"after trunk projection: {x_proj.shape}")

        use_mc = self.mc_dropout_enabled and (self.training or self.dropout.training)
        if use_mc and self.num_dropout_samples > 1:
            logits_list = []
            for _ in range(self.num_dropout_samples):
                dropped = self.dropout(x_proj)
                logits_list.append(self._logits_from_proj(dropped))
            logits = torch.stack(logits_list, dim=0).mean(dim=0)
        else:
            logits = self._logits_from_proj(x_proj)

        if debugging:
            print(f"logits: {logits.shape}")
        return logits
