import torch
import os
from sqlalchemy.orm import sessionmaker
from collections import Counter

from .win_model import WinPredictor
from .vectorizer import ChampionIndexer, DraftVectorizer
from .schema import init_db, Match, Pick
from .config import DB_URL

MODEL_PATH = "data/win_model_v1.pth"

class DraftEngine:
    def __init__(self):
        self.engine = init_db(DB_URL)
        self.session = sessionmaker(bind=self.engine)()
        
        self.indexer = ChampionIndexer()
        self.indexer.load()
        self.vocab_size = len(self.indexer.champ_to_idx)
        
        self.model = WinPredictor(self.vocab_size)
        self.model.load_state_dict(torch.load(MODEL_PATH))
        self.model.eval()
        
        # Build Role Cache (Which champs go where?)
        self.role_map = self._build_role_map()

    def _build_role_map(self):
        """Scans DB to find most common roles for each champ."""
        print("Building Role Map...")
        role_counts = {} # {ChampName: {Top: 10, Mid: 50...}}
        
        picks = self.session.query(Pick).all()
        for p in picks:
            if p.champion not in role_counts:
                role_counts[p.champion] = Counter()
            role_counts[p.champion][p.role] += 1
            
        # Normalize
        champ_roles = {}
        for champ, counts in role_counts.items():
            primary = counts.most_common(1)[0][0]
            # Allow secondary roles if they are > 20% of picks
            total = sum(counts.values())
            valid_roles = [r for r, c in counts.items() if c/total > 0.2]
            champ_roles[champ] = valid_roles
        return champ_roles

    def get_recommendations(self, my_team, enemy_team, bans, target_role=None):
        """
        Simulates adding every possible champion to 'my_team' 
        and returns the ones with highest predicted Win %.
        """
        my_ids = [self.indexer.get_id(c) for c in my_team]
        en_ids = [self.indexer.get_id(c) for c in enemy_team]
        ban_ids = [self.indexer.get_id(c) for c in bans]
        
        candidates = []
        
        # Iterate over all champions
        for name, cid in self.indexer.champ_to_idx.items():
            # 1. Skip invalid
            if cid <= 1: continue # PAD/UNK
            if name in my_team or name in enemy_team or name in bans: continue
            
            # 2. Filter by Role (if requested)
            if target_role:
                valid_roles = self.role_map.get(name, [])
                if target_role not in valid_roles:
                    continue

            # 3. Simulate Draft
            # Add candidate to my team
            sim_my_team = my_ids + [cid]
            
            # Pad to 5
            sim_my_input = sim_my_team + [0]*(5 - len(sim_my_team))
            sim_en_input = en_ids + [0]*(5 - len(en_ids))
            sim_ban_input = ban_ids + [0]*(10 - len(ban_ids))
            
            # Tensorize
            b_t = torch.tensor([sim_my_input], dtype=torch.long)
            r_t = torch.tensor([sim_en_input], dtype=torch.long)
            ban_t = torch.tensor([sim_ban_input], dtype=torch.long)
            
            # Predict
            with torch.no_grad():
                win_prob = self.model(b_t, r_t, ban_t).item()
                
            candidates.append((name, win_prob))
            
        # Sort by Win %
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:10]

if __name__ == "__main__":
    try:
        engine = DraftEngine()
        indexer = engine.indexer  # Reference to indexer for validation
        print("\n" + "="*50)
        print("LEAGUE OF LEGENDS DRAFT ASSISTANT")
        print("="*50)
        print("Type 'exit' to quit, 'help' for commands\n")
        
        my_team = []
        enemy_team = []
        bans = []
        
        while True:
            print(f"\nCurrent Draft State:")
            print(f"  My Team ({len(my_team)}/5):   {', '.join(my_team) if my_team else '(empty)'}")
            print(f"  Enemy ({len(enemy_team)}/5):  {', '.join(enemy_team) if enemy_team else '(empty)'}")
            print(f"  Bans ({len(bans)}/10):      {', '.join(bans) if bans else '(empty)'}")
            
            action = input("\n(P)ick, (E)nemy, (B)an, (R)ecommend, (C)lear, or e(X)it? ").lower().strip()
            
            if action in ['x', 'exit']: 
                break
            elif action == 'help':
                print("\nCommands:")
                print("  P - Add your team's pick")
                print("  E - Add enemy team's pick")
                print("  B - Add a banned champion")
                print("  R - Get pick recommendations (with optional role filter)")
                print("  C - Clear all and start over")
                print("  X - Exit")
                continue
            elif action == 'c':
                my_team, enemy_team, bans = [], [], []
                print("Draft cleared!")
            elif action == 'p':
                if len(my_team) >= 5:
                    print("Your team is already full (5 champions)")
                    continue
                champ = input("Enter Champion Name: ").strip()
                if champ and indexer.get_id(champ) > 1:  # Valid champion
                    my_team.append(champ)
                    print(f"Added {champ} to your team")
                else:
                    print(f"Unknown champion: {champ}")
            elif action == 'e':
                if len(enemy_team) >= 5:
                    print("Enemy team is already full (5 champions)")
                    continue
                champ = input("Enter Enemy Champion: ").strip()
                if champ and engine.indexer.get_id(champ) > 1:
                    enemy_team.append(champ)
                    print(f"Added {champ} to enemy team")
                else:
                    print(f"Unknown champion: {champ}")
            elif action == 'b':
                if len(bans) >= 10:
                    print("Ban phase is complete (10 champions)")
                    continue
                champ = input("Enter Banned Champion: ").strip()
                if champ and engine.indexer.get_id(champ) > 1:
                    bans.append(champ)
                    print(f"Added {champ} to bans")
                else:
                    print(f"Unknown champion: {champ}")
            elif action == 'r':
                role_input = input("Filter by Role? (top/jng/mid/bot/sup or Enter for all): ").lower().strip()
                target_role = role_input if role_input else None
                
                try:
                    recs = engine.get_recommendations(my_team, enemy_team, bans, target_role=target_role)
                    if recs:
                        print("\n" + "="*40)
                        print("TOP 10 RECOMMENDATIONS")
                        print("="*40)
                        for i, (name, win_pct) in enumerate(recs, 1):
                            bar_length = int(win_pct * 20)
                            bar = "█" * bar_length + "░" * (20 - bar_length)
                            print(f"{i:2d}. {name:20s} {bar} {win_pct*100:5.1f}%")
                    else:
                        print("No valid recommendations available")
                except Exception as e:
                    print(f"Error generating recommendations: {e}")
            else:
                print("Invalid command. Type 'help' for options.")
                
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure to train the model first: python -m src.train_win")
    except Exception as e:
        print(f"Critical Error: {e}")
        import traceback
        traceback.print_exc()
