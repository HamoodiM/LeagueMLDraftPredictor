import torch
import torch.nn as nn

class WinPredictor(nn.Module):
    def __init__(self, vocab_size, emb_dim=48, hidden_dim=128):
        super(WinPredictor, self).__init__()
        
        # Embeddings: We learn a vector for every champion
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        
        # Feature Extractor for each team
        # We process Blue and Red symmetrically
        self.team_encoder = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Global Context (Bans)
        self.ban_encoder = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        # Final Decision Layer
        # Input: Blue_Feats + Red_Feats + Ban_Feats
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2 + (hidden_dim // 2), hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid() # Output probability 0.0-1.0
        )

    def forward(self, blue_ids, red_ids, ban_ids):
        """
        blue_ids: [Batch, 5] (Champion IDs, padded with 0 if partial team)
        red_ids:  [Batch, 5]
        ban_ids:  [Batch, 10]
        """
        # 1. Embed Champions
        # [Batch, 5, Emb]
        b_emb = self.embedding(blue_ids)
        r_emb = self.embedding(red_ids)
        ban_emb = self.embedding(ban_ids)
        
        # 2. Aggregate Teams (Sum or Mean)
        # We sum the vectors. A partial team (Ahri + 0 + 0...) is just Ahri's vector.
        # This handles the "Flexibility" naturally.
        b_vec = torch.sum(b_emb, dim=1) 
        r_vec = torch.sum(r_emb, dim=1)
        ban_vec = torch.mean(ban_emb, dim=1) # Average the bans
        
        # 3. Encode
        b_feat = self.team_encoder(b_vec)
        r_feat = self.team_encoder(r_vec)
        ban_feat = self.ban_encoder(ban_vec)
        
        # 4. Predict
        combined = torch.cat([b_feat, r_feat, ban_feat], dim=1)
        return self.classifier(combined)
