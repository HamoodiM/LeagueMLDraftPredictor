import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sqlalchemy.orm import sessionmaker
from tqdm import tqdm
import os
import random

from .schema import init_db, Match
from .config import DB_URL
from .vectorizer import ChampionIndexer, DraftVectorizer
from .win_model import WinPredictor

BATCH_SIZE = 64
LR = 0.0005
EPOCHS = 50
MODEL_PATH = "data/win_model_v1.pth"

class PartialDraftDataset(Dataset):
    def __init__(self, matches, vectorizer):
        self.samples = []
        print("Generating partial draft states...")
        
        for match in tqdm(matches):
            data = vectorizer.encode_match(match)
            
            # Get raw IDs (lists) instead of Multi-hot for embedding lookups
            blue_team = data['blue_picks_raw']
            red_team = data['red_picks_raw']
            bans = data['bans_raw']
            outcome = data['outcome'].item()
            
            # STATE AUGMENTATION:
            # Generate random partial states from this finished game.
            # e.g., Blue has 2 champs, Red has 1.
            # This teaches the model to predict the final winner from early info.
            for _ in range(3): # Generate 3 random snapshots per game
                b_count = random.randint(0, 5)
                r_count = random.randint(0, 5)
                
                # Mask out the future picks (set to 0)
                partial_blue = blue_team[:b_count] + [0]*(5-b_count)
                partial_red = red_team[:r_count] + [0]*(5-r_count)
                
                # Pad bans to 10
                padded_bans = bans + [0]*(10 - len(bans))
                
                self.samples.append({
                    'blue': torch.tensor(partial_blue, dtype=torch.long),
                    'red': torch.tensor(partial_red, dtype=torch.long),
                    'bans': torch.tensor(padded_bans[:10], dtype=torch.long),
                    'label': torch.tensor(outcome, dtype=torch.float)
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def train():
    # Setup
    engine = init_db(DB_URL)
    session = sessionmaker(bind=engine)()
    indexer = ChampionIndexer()
    if not os.path.exists(indexer.vocab_file):
        print("Error: Vocabulary not found. Run vectorizer.py first.")
        return
    indexer.load()
    
    vectorizer = DraftVectorizer(indexer)
    
    matches = session.query(Match).all()
    if not matches:
        print("Error: No matches found in database. Run ingest.py first.")
        return
    
    dataset = PartialDraftDataset(matches, vectorizer)
    
    # Train/Val/Test Split (70/15/15)
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WinPredictor(len(indexer.champ_to_idx)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCELoss() # Binary Cross Entropy
    
    print(f"Training on {device}")
    print(f"Dataset sizes: Train={train_size}, Val={val_size}, Test={test_size}")
    
    best_val_loss = float('inf')
    best_model_path = MODEL_PATH.replace('.pth', '_best.pth')
    
    # Training Loop
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        correct = 0
        total = 0
        
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for batch in progress:
            b = batch['blue'].to(device)
            r = batch['red'].to(device)
            bans = batch['bans'].to(device)
            y = batch['label'].to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            pred = model(b, r, bans)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            predicted_class = (pred > 0.5).float()
            correct += (predicted_class == y).sum().item()
            total += y.size(0)
            
            progress.set_postfix(loss=loss.item(), acc=correct/total)
        
        avg_train_loss = total_loss / len(train_loader)
        train_acc = correct / total
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in val_loader:
                b = batch['blue'].to(device)
                r = batch['red'].to(device)
                bans = batch['bans'].to(device)
                y = batch['label'].to(device).unsqueeze(1)
                
                pred = model(b, r, bans)
                loss = criterion(pred, y)
                
                val_loss += loss.item()
                val_correct += (pred.round() == y).sum().item()
                val_total += y.size(0)
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total
        
        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Acc={train_acc:.4f} | Val Loss={avg_val_loss:.4f}, Acc={val_acc:.4f}")
        
        # Checkpoint best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> Saved best model (Val Loss: {best_val_loss:.4f})")
        
        model.train()
    
    # Test Set Evaluation
    print("\n" + "="*50)
    print("FINAL TEST SET EVALUATION")
    print("="*50)
    model.eval()
    test_loss = 0
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for batch in test_loader:
            b = batch['blue'].to(device)
            r = batch['red'].to(device)
            bans = batch['bans'].to(device)
            y = batch['label'].to(device).unsqueeze(1)
            
            pred = model(b, r, bans)
            loss = criterion(pred, y)
            
            test_loss += loss.item()
            test_correct += (pred.round() == y).sum().item()
            test_total += y.size(0)
    
    test_acc = test_correct / test_total
    avg_test_loss = test_loss / len(test_loader)
    print(f"Test Loss: {avg_test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f} ({test_correct}/{test_total})")
    
    # Save final model
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\nFinal model saved to {MODEL_PATH}")
    print(f"Best model saved to {best_model_path}")
    print("Training Complete.")
    total_loss = 0
    correct = 0
    total = 0
    
    progress = tqdm(dataloader, desc=f"Epoch {epoch+1}")
    for batch in progress:
        b = batch['blue'].to(device)
        r = batch['red'].to(device)
        bans = batch['bans'].to(device)
        y = batch['label'].to(device).unsqueeze(1)
        
        optimizer.zero_grad()
        pred = model(b, r, bans)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        predicted_class = (pred > 0.5).float()
        correct += (predicted_class == y).sum().item()
        total += y.size(0)
        
        progress.set_postfix(loss=loss.item(), acc=correct/total)
        
    torch.save(model.state_dict(), MODEL_PATH)
    print("Training Complete.")

if __name__ == "__main__":
    train()
