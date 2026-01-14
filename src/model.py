import torch
import torch.nn as nn
import torch.nn.functional as F

class DraftRecommenderNet(nn.Module):
    def __init__(self, vocab_size, embedding_dim=32, hidden_dim=256, dropout=0.3):
        super(DraftRecommenderNet, self).__init__()
        
        # 1. Embeddings for Bans (Sequence data)
        # Bans have order, so we embed them. 10 bans * embedding_dim
        self.ban_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # 2. Dense Layers for Picks (Set data / Multi-hot)
        # Input size: vocab_size (Blue Picks) + vocab_size (Red Picks)
        self.pick_projection = nn.Linear(vocab_size * 2, hidden_dim * 2)
        
        # 3. Combined Feature Extractor
        # Input: Flattened Bans (10*emb) + Projected Picks (hidden*2)
        combined_input_dim = (10 * embedding_dim) + (hidden_dim * 2)
        
        self.fc1 = nn.Linear(combined_input_dim, hidden_dim * 2)
        self.bn1 = nn.BatchNorm1d(hidden_dim * 2)
        
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        
        # 4. Output Head
        self.output = nn.Linear(hidden_dim, vocab_size)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, blue_picks_hot, red_picks_hot, bans_indices):
        """
        blue_picks_hot: [Batch, Vocab_Size] (Multi-hot of teammates)
        red_picks_hot:  [Batch, Vocab_Size] (Multi-hot of enemies)
        bans_indices:   [Batch, 10] (Indices of bans)
        """
        # A. Process Bans
        # [Batch, 10] -> [Batch, 10, Emb] -> [Batch, 10*Emb]
        bans_embed = self.ban_embedding(bans_indices)
        bans_flat = bans_embed.view(bans_embed.size(0), -1)
        
        # B. Process Picks
        picks_concat = torch.cat([blue_picks_hot, red_picks_hot], dim=1)
        picks_feat = F.relu(self.pick_projection(picks_concat))
        
        # C. Combine
        x = torch.cat([bans_flat, picks_feat], dim=1)
        
        # D. Deep Network (ResNet blocks could go here, using simple MLP for now)
        x = self.dropout(F.relu(self.bn1(self.fc1(x))))
        x = self.dropout(F.relu(self.bn2(self.fc2(x))))
        x = self.dropout(F.relu(self.bn3(self.fc3(x))))
        
        # E. Output Logits
        logits = self.output(x)
        return logits
