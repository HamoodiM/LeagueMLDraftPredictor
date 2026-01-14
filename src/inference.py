import torch
import os
import json
from .model import DraftRecommenderNet
from .vectorizer import ChampionIndexer

# --- CONFIG ---
MODEL_PATH = "data/model_v1.pth"
VOCAB_PATH = "data/vocab.json"
EMBEDDING_DIM = 32
HIDDEN_DIM = 256

def load_system():
    # 1. Load Vocab
    indexer = ChampionIndexer(VOCAB_PATH)
    if not os.path.exists(VOCAB_PATH):
        raise FileNotFoundError("Vocab file not found. Did you run vectorizer.py?")
    indexer.load()
    vocab_size = len(indexer.champ_to_idx)

    # 2. Load Model Architecture
    model = DraftRecommenderNet(vocab_size, EMBEDDING_DIM, HIDDEN_DIM)
    
    # 3. Load Weights
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model file not found. Did you run train.py?")
        
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval() # Set to evaluation mode (turns off dropout)
    return model, indexer

def predict_pick(model, indexer, my_picks, enemy_picks, bans):
    """
    Returns the top recommended picks for the next slot.
    """
    # 1. Convert names to IDs
    my_ids = [indexer.get_id(name) for name in my_picks]
    enemy_ids = [indexer.get_id(name) for name in enemy_picks]
    ban_ids = [indexer.get_id(name) for name in bans]

    # 2. Create Tensors
    vocab_size = len(indexer.champ_to_idx)
    
    # Multi-hot encoding for picks
    my_picks_vec = torch.zeros(1, vocab_size)
    my_picks_vec[0, my_ids] = 1.0
    
    enemy_picks_vec = torch.zeros(1, vocab_size)
    enemy_picks_vec[0, enemy_ids] = 1.0
    
    # Pad bans to 10
    ban_ids += [0] * (10 - len(ban_ids))
    bans_tensor = torch.tensor([ban_ids], dtype=torch.long)

    # 3. Forward Pass
    with torch.no_grad():
        logits = model(my_picks_vec, enemy_picks_vec, bans_tensor)
    
    # 4. Masking (BEFORE softmax) - Set logits of banned/picked champs to negative infinity
    unavailable = set(my_ids + enemy_ids + ban_ids)
    for uid in unavailable:
        if uid > 1:  # Don't mask special tokens (PAD=0, UNK=1)
            logits[0, uid] = -float('inf')

    # 5. Get Top 5 (after masking, before softmax)
    top_logits, top_indices = torch.topk(logits, 5)
    
    # Apply softmax only to top logits for display
    probs = torch.softmax(top_logits, dim=1)
    
    recommendations = []
    for i in range(5):
        idx = top_indices[0, i].item()
        name = indexer.get_name(idx)
        prob = probs[0, i].item()
        recommendations.append((name, prob))
        
    return recommendations

if __name__ == "__main__":
    # --- INTERACTIVE SHELL ---
    try:
        model, indexer = load_system()
        print("Model Loaded! Type 'exit' to quit.\n")
        
        while True:
            print("-" * 30)
            print("Enter current draft state (Comma separated names). Case-sensitive!")
            print("Example: Aatrox, Ahri")
            
            my_team_str = input("My Team Picks (so far): ")
            if my_team_str.lower() == 'exit': break
            
            enemy_team_str = input("Enemy Team Picks: ")
            bans_str = input("Bans (Global): ")
            
            # Helper to parse lists
            def parse(s):
                return [x.strip() for x in s.split(',') if x.strip()]

            my_team = parse(my_team_str)
            enemy_team = parse(enemy_team_str)
            bans = parse(bans_str)
            
            print(f"\nAnalyzed Context: Allies={len(my_team)}, Enemies={len(enemy_team)}, Bans={len(bans)}")
            
            # Validate champion names
            invalid = []
            for champ in my_team + enemy_team + bans:
                if indexer.get_id(champ) == 1:  # UNK token
                    invalid.append(champ)
            
            if invalid:
                print(f"Unknown champions: {invalid}")
                print("Check spelling and capitalization!")
                continue
            
            try:
                recs = predict_pick(model, indexer, my_team, enemy_team, bans)
                print("\nRecommended Picks:")
                for name, p in recs:
                    print(f"  {name}: {p*100:.1f}%")
            except Exception as e:
                print(f"Error during inference: {e}")
                
    except FileNotFoundError as e:
        print(f"Setup Error: {e}")
