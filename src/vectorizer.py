import torch
import json
import os
import numpy as np
from sqlalchemy.orm import sessionmaker
from .schema import Match, Ban, Pick, init_db
from .config import DB_URL

class ChampionIndexer:
    """
    Maintains a bidirectional mapping between Champion Names and Integer IDs.
    """
    def __init__(self, vocab_file="data/vocab.json"):
        self.vocab_file = vocab_file
        self.champ_to_idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx_to_champ = {0: "<PAD>", 1: "<UNK>"}
        
        if os.path.exists(vocab_file):
            self.load()

    def build_from_db(self, session):
        """Scans the database for all unique champion names."""
        print("Building vocabulary from database...")
        # Get all picks and bans to ensure we cover every champion
        picked_champs = [r[0] for r in session.query(Pick.champion).distinct().all()]
        banned_champs = [r[0] for r in session.query(Ban.champion).distinct().all()]
        
        unique_champs = sorted(list(set(picked_champs + banned_champs)))
        
        # Start IDs at 2 (0 and 1 are reserved)
        for i, champ in enumerate(unique_champs):
            if champ and champ != "None": # Filter empty bans
                self.add_champion(champ)
        
        self.save()
        print(f"Vocabulary built: {len(self.champ_to_idx)} champions.")

    def add_champion(self, name):
        if name not in self.champ_to_idx:
            idx = len(self.champ_to_idx)
            self.champ_to_idx[name] = idx
            self.idx_to_champ[idx] = name

    def get_id(self, name):
        return self.champ_to_idx.get(name, self.champ_to_idx["<UNK>"])

    def get_name(self, idx):
        return self.idx_to_champ.get(idx, "<UNK>")

    def save(self):
        with open(self.vocab_file, 'w') as f:
            json.dump(self.champ_to_idx, f)

    def load(self):
        with open(self.vocab_file, 'r') as f:
            self.champ_to_idx = json.load(f)
            self.idx_to_champ = {int(k): v for v, k in self.champ_to_idx.items()}

class DraftVectorizer:
    """
    Converts a Match DB Object into Tensor representations.
    """
    def __init__(self, indexer):
        self.indexer = indexer
        self.vocab_size = len(indexer.champ_to_idx)

    def encode_match(self, match):
        """
        Returns a dictionary of tensors representing the Full Match State.
        Suitable for training a 'Win Predictor' or 'Imputation' model.
        """
        # 1. Encode Bans (Blue and Red)
        blue_bans = [self.indexer.get_id(b.champion) for b in match.bans if b.team_side == 'Blue']
        red_bans = [self.indexer.get_id(b.champion) for b in match.bans if b.team_side == 'Red']
        
        # Store raw bans before padding
        blue_bans_raw = blue_bans.copy()
        red_bans_raw = red_bans.copy()
        all_bans_raw = blue_bans_raw + red_bans_raw
        
        # Pad bans to fixed size (usually 5)
        blue_bans += [0] * (5 - len(blue_bans))
        red_bans += [0] * (5 - len(red_bans))

        # 2. Encode Picks (Set of 5)
        blue_picks = [self.indexer.get_id(p.champion) for p in match.picks if p.team_side == 'Blue']
        red_picks = [self.indexer.get_id(p.champion) for p in match.picks if p.team_side == 'Red']
        
        # Store raw picks before conversion
        blue_picks_raw = blue_picks.copy()
        red_picks_raw = red_picks.copy()
        
        # 3. Create Multi-Hot Vectors (Bag of Champions)
        # Shape: [Vocab_Size] -> 1 if champion is present, 0 otherwise
        blue_picks_vec = torch.zeros(self.vocab_size)
        blue_picks_vec[blue_picks] = 1.0
        
        red_picks_vec = torch.zeros(self.vocab_size)
        red_picks_vec[red_picks] = 1.0

        # 4. Result (Label)
        win_label = 1.0 if match.winner_side == 'Blue' else 0.0

        return {
            "blue_bans": torch.tensor(blue_bans, dtype=torch.long),
            "red_bans": torch.tensor(red_bans, dtype=torch.long),
            "blue_bans_raw": blue_bans_raw,
            "red_bans_raw": red_bans_raw,
            "bans_raw": all_bans_raw,
            "blue_picks_vec": blue_picks_vec,
            "red_picks_vec": red_picks_vec,
            "blue_picks_raw": blue_picks_raw,
            "red_picks_raw": red_picks_raw,
            "outcome": torch.tensor(win_label, dtype=torch.float)
        }

    def get_invalid_mask(self, current_bans, current_picks):
        """
        Creates a boolean mask [Vocab_Size] where 1 = Unavailable.
        Use this during inference to prevent the model from suggesting taken champs.
        """
        mask = torch.zeros(self.vocab_size, dtype=torch.bool)
        
        # Mask out special tokens
        mask[0] = True # PAD
        mask[1] = True # UNK
        
        # Mask out bans
        for ban_id in current_bans:
            if ban_id > 1:
                mask[ban_id] = True
                
        # Mask out existing picks
        for pick_id in current_picks:
            if pick_id > 1:
                mask[pick_id] = True
                
        return mask

# --- Usage Example ---
if __name__ == "__main__":
    # 1. Initialize DB
    engine = init_db(DB_URL)
    Session = sessionmaker(bind=engine)
    session = Session()

    # 2. Build/Load Vocab
    indexer = ChampionIndexer()
    if not os.path.exists(indexer.vocab_file):
        indexer.build_from_db(session)
    
    # 3. Vectorize a Sample Match
    vectorizer = DraftVectorizer(indexer)
    
    # Get first match
    match = session.query(Match).first()
    if match:
        data = vectorizer.encode_match(match)
        print(f"Match ID: {match.game_id}")
        print(f"Blue Picks Tensor Shape: {data['blue_picks_vec'].shape}") # Should be [Vocab_Size]
        print(f"Blue Bans: {data['blue_bans']}") # List of IDs
        
        # Test Masking
        # Imagine we want to recommend a pick, but Aatrox (ID X) is banned
        mask = vectorizer.get_invalid_mask(
            current_bans=data['blue_bans'].tolist() + data['red_bans'].tolist(),
            current_picks=[]
        )
        print(f"Mask Size: {mask.shape}")
        print(f"Is index {data['blue_bans'][0]} masked? {mask[data['blue_bans'][0]]}") # Should be True
    else:
        print("No matches found in DB. Run ingest.py first!")
