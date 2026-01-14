import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sqlalchemy.orm import sessionmaker
from tqdm import tqdm
import os

from .schema import init_db, Match
from .config import DB_URL
from .vectorizer import ChampionIndexer, DraftVectorizer
from .model import DraftRecommenderNet

# --- Hyperparameters ---
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 50
EMBEDDING_DIM = 32
HIDDEN_DIM = 256
MODEL_PATH = "data/model_v1.pth"

class DraftDataset(Dataset):
    def __init__(self, matches, vectorizer):
        self.samples = []
        print("Preparing dataset...")
        for match in tqdm(matches):
            data = vectorizer.encode_match(match)
            
            # We want to train the model to predict picks for the WINNING team.
            # (Learning from losers introduces bad habits).
            if data['outcome'].item() == 1.0:
                target_picks = data['blue_picks_vec'] # Multi-hot
                enemy_picks = data['red_picks_vec']
                my_bans = data['blue_bans']
                enemy_bans = data['red_bans']
            else:
                target_picks = data['red_picks_vec']
                enemy_picks = data['blue_picks_vec']
                my_bans = data['red_bans']
                enemy_bans = data['blue_bans']

            all_bans = torch.cat([my_bans, enemy_bans], dim=0)

            # --- MASKING STRATEGY ---
            # The 'target_picks' vector has 5 ones.
            # We will create 5 training samples from this 1 match.
            # In each sample, we set ONE of those ones to zero (hide it),
            # and set that index as the Label.
            
            indices = (target_picks == 1).nonzero(as_tuple=False).view(-1)
            
            for idx_to_hide in indices:
                # Clone inputs
                partial_picks = target_picks.clone()
                partial_picks[idx_to_hide] = 0 # Hide this champion
                
                self.samples.append({
                    'context_picks': partial_picks,
                    'enemy_picks': enemy_picks,
                    'bans': all_bans,
                    'target_label': idx_to_hide # The ID of the hidden champ
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def train():
    """
    Train the Draft Recommender Network.
    
    Raises:
        FileNotFoundError: If vocab or database not found
        RuntimeError: If no matches found in database
    """
    try:
        # 1. Database & Vocab
        print("Initializing database and vocabulary...")
        engine = init_db(DB_URL)
        session = sessionmaker(bind=engine)()
        
        indexer = ChampionIndexer()
        if not os.path.exists(indexer.vocab_file):
            raise FileNotFoundError(
                f"Vocabulary file not found at {indexer.vocab_file}. "
                "Please run: python -m src.ingest"
            )
        indexer.load()
        print(f"✓ Loaded {len(indexer.champ_to_idx)} champions")
        
    except Exception as e:
        print(f"❌ Initialization Error: {e}")
        return
    
    try:
        vectorizer = DraftVectorizer(indexer)
        vocab_size = len(indexer.champ_to_idx)
        
        # 2. Load Data
        print("Loading match data from database...")
        matches = session.query(Match).all()
        if not matches:
            raise RuntimeError(
                "No matches found in database. "
                "Please run: python -m src.ingest"
            )
        print(f"✓ Loaded {len(matches)} matches")
            
        dataset = DraftDataset(matches, vectorizer)
        if len(dataset) == 0:
            raise RuntimeError("No training samples generated from matches")
        print(f"✓ Generated {len(dataset)} training samples")
        
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        
        # 3. Model Setup
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"✓ Training on {device}")
        
        model = DraftRecommenderNet(vocab_size, EMBEDDING_DIM, HIDDEN_DIM).to(device)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.CrossEntropyLoss()
        
    except Exception as e:
        print(f"❌ Data Loading Error: {e}")
        return
    
    # 4. Training Loop
    try:
        model.train()
        for epoch in range(EPOCHS):
            total_loss = 0
            correct = 0
            total = 0
            
            progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
            for batch in progress:
                # Move to device
                context_picks = batch['context_picks'].to(device)
                enemy_picks = batch['enemy_picks'].to(device)
                bans = batch['bans'].to(device)
                target = batch['target_label'].to(device) # Shape [Batch]
                
                optimizer.zero_grad()
                
                # Forward
                logits = model(context_picks, enemy_picks, bans)
                
                # Loss
                loss = criterion(logits, target)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                # Accuracy Calc
                _, predicted = torch.max(logits, 1)
                correct += (predicted == target).sum().item()
                total += target.size(0)
                
                progress.set_postfix(loss=loss.item(), acc=correct/total)
                
            print(f"Epoch {epoch+1} finished. Avg Loss: {total_loss/len(dataloader):.4f}, Acc: {correct/total:.4f}")

        # 5. Save
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"✓ Model saved to {MODEL_PATH}")
        
    except KeyboardInterrupt:
        print("\n⚠ Training interrupted by user")
        torch.save(model.state_dict(), MODEL_PATH.replace('.pth', '_interrupted.pth'))
        print(f"  Partial model saved to {MODEL_PATH.replace('.pth', '_interrupted.pth')}")
    except Exception as e:
        print(f"❌ Training Error: {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        print(f"❌ Fatal Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
