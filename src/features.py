"""
Advanced Feature Engineering for Draft Prediction.

Includes:
- Champion Synergy features (common pairs that win together)
- Counter-pick information (which champs counter which)
- Patch-specific variations
- Role-aware embeddings
"""

import json
import os
from collections import defaultdict, Counter
from sqlalchemy.orm import sessionmaker
import torch

from .schema import init_db, Match, Pick
from .config import DB_URL
from .vectorizer import ChampionIndexer


class SynergyCalculator:
    """Computes champion pair synergy scores from historical data."""
    
    def __init__(self, session, indexer):
        """
        Args:
            session: SQLAlchemy session
            indexer: ChampionIndexer instance
        """
        self.session = session
        self.indexer = indexer
        self.synergy_scores = self._compute_synergies()
        self.synergy_file = "data/synergies.json"
        
    def _compute_synergies(self):
        """
        For each pair of champions, compute:
        synergy = (wins together) / (total games together)
        
        Returns:
            dict: {(champ1, champ2): synergy_score}
        """
        print("Computing champion synergies...")
        pair_stats = defaultdict(lambda: {"wins": 0, "total": 0})
        
        matches = self.session.query(Match).all()
        for match in matches:
            # Get blue and red picks
            blue_picks = [p.champion for p in match.picks if p.team_side == 'Blue']
            red_picks = [p.champion for p in match.picks if p.team_side == 'Red']
            
            is_blue_win = (match.winner_side == 'Blue')
            
            # Compute all blue pairs
            for i, c1 in enumerate(blue_picks):
                for c2 in blue_picks[i+1:]:
                    key = tuple(sorted([c1, c2]))
                    pair_stats[key]["total"] += 1
                    if is_blue_win:
                        pair_stats[key]["wins"] += 1
            
            # Compute all red pairs
            for i, c1 in enumerate(red_picks):
                for c2 in red_picks[i+1:]:
                    key = tuple(sorted([c1, c2]))
                    pair_stats[key]["total"] += 1
                    if not is_blue_win:
                        pair_stats[key]["wins"] += 1
        
        # Convert to synergy scores (win rate)
        synergies = {}
        for pair, stats in pair_stats.items():
            if stats["total"] >= 5:  # Only consider pairs with 5+ games
                synergy = stats["wins"] / stats["total"]
                synergies[pair] = synergy
        
        print(f"Computed {len(synergies)} pair synergies")
        return synergies
    
    def get_synergy(self, champ1, champ2):
        """Returns synergy score for a champion pair (0-1)."""
        key = tuple(sorted([champ1, champ2]))
        return self.synergy_scores.get(key, 0.5)  # Default to neutral if not found
    
    def save(self):
        """Save synergies to JSON."""
        serializable = {str(k): v for k, v in self.synergy_scores.items()}
        with open(self.synergy_file, 'w') as f:
            json.dump(serializable, f)
        print(f"Synergies saved to {self.synergy_file}")
    
    def load(self):
        """Load synergies from JSON."""
        if os.path.exists(self.synergy_file):
            with open(self.synergy_file, 'r') as f:
                data = json.load(f)
                # Convert back to tuple keys
                self.synergy_scores = {eval(k): v for k, v in data.items()}
            print(f"Loaded synergies from {self.synergy_file}")


class CounterPickAnalyzer:
    """Analyzes which champions counter which."""
    
    def __init__(self, session, indexer):
        """
        Args:
            session: SQLAlchemy session
            indexer: ChampionIndexer instance
        """
        self.session = session
        self.indexer = indexer
        self.counters = self._compute_counters()
        self.counter_file = "data/counters.json"
    
    def _compute_counters(self):
        """
        For each champion pair (ally vs enemy):
        counter_score = (ally wins) / (total matchups)
        
        Returns:
            dict: {enemy_champ: {ally_champ: counter_score}}
        """
        print("Computing counter-pick statistics...")
        matchup_stats = defaultdict(lambda: defaultdict(lambda: {"wins": 0, "total": 0}))
        
        matches = self.session.query(Match).all()
        for match in matches:
            blue_picks = [p.champion for p in match.picks if p.team_side == 'Blue']
            red_picks = [p.champion for p in match.picks if p.team_side == 'Red']
            
            is_blue_win = (match.winner_side == 'Blue')
            
            # Check all 1v1 matchups
            for blue_champ in blue_picks:
                for red_champ in red_picks:
                    matchup_stats[red_champ][blue_champ]["total"] += 1
                    if is_blue_win:
                        matchup_stats[red_champ][blue_champ]["wins"] += 1
        
        # Convert to counter scores
        counters = {}
        for enemy, allies in matchup_stats.items():
            counters[enemy] = {}
            for ally, stats in allies.items():
                if stats["total"] >= 3:  # Only consider matchups with 3+ games
                    win_rate = stats["wins"] / stats["total"]
                    counters[enemy][ally] = win_rate
        
        print(f"Computed matchups for {len(counters)} champions")
        return counters
    
    def get_counter_score(self, my_champ, enemy_champ):
        """Returns how well my_champ counters enemy_champ (0-1)."""
        if enemy_champ not in self.counters:
            return 0.5
        return self.counters[enemy_champ].get(my_champ, 0.5)
    
    def save(self):
        """Save counter data to JSON."""
        with open(self.counter_file, 'w') as f:
            json.dump(self.counters, f)
        print(f"Counter data saved to {self.counter_file}")
    
    def load(self):
        """Load counter data from JSON."""
        if os.path.exists(self.counter_file):
            with open(self.counter_file, 'r') as f:
                self.counters = json.load(f)
            print(f"Loaded counter data from {self.counter_file}")


class RoleAwareEmbeddings:
    """Generates role-aware champion embeddings."""
    
    def __init__(self, session, indexer, embedding_dim=32):
        """
        Args:
            session: SQLAlchemy session
            indexer: ChampionIndexer instance
            embedding_dim: Dimension of role-aware embeddings
        """
        self.session = session
        self.indexer = indexer
        self.embedding_dim = embedding_dim
        self.role_embeddings = self._compute_role_embeddings()
        self.embed_file = "data/role_embeddings.pth"
    
    def _compute_role_embeddings(self):
        """
        For each champion, create role-specific features:
        - Primary role (one-hot)
        - Role flexibility (how many roles they can play)
        - Win rate by role
        
        Returns:
            torch.Tensor: [vocab_size, embedding_dim] embeddings
        """
        print("Computing role-aware embeddings...")
        
        roles = ['top', 'jng', 'mid', 'bot', 'sup']
        role_to_idx = {r: i for i, r in enumerate(roles)}
        
        # Gather stats per champion per role
        champ_role_stats = defaultdict(lambda: defaultdict(lambda: {"wins": 0, "total": 0}))
        
        picks = self.session.query(Pick).all()
        for pick in picks:
            role_lower = pick.role.lower()
            champ_role_stats[pick.champion][role_lower]["total"] += 1
            if pick.win:
                champ_role_stats[pick.champion][role_lower]["wins"] += 1
        
        # Create embeddings
        vocab_size = len(self.indexer.champ_to_idx)
        embeddings = torch.zeros(vocab_size, self.embedding_dim)
        
        for champ_name, role_dict in champ_role_stats.items():
            champ_id = self.indexer.get_id(champ_name)
            if champ_id <= 1:  # Skip special tokens
                continue
            
            # Compute features
            total_games = sum(stats["total"] for stats in role_dict.values())
            if total_games == 0:
                continue
            
            # Fill embedding
            for role, stats in role_dict.items():
                if role in role_to_idx:
                    role_idx = role_to_idx[role]
                    # Feature 1: Win rate in this role
                    win_rate = stats["wins"] / stats["total"] if stats["total"] > 0 else 0
                    embeddings[champ_id, role_idx] = win_rate
            
            # Feature 6: Overall win rate
            total_wins = sum(stats["wins"] for stats in role_dict.values())
            embeddings[champ_id, 5] = total_wins / total_games if total_games > 0 else 0
            
            # Features 7+: Flexibility (how many roles played)
            num_roles = len([s for s in role_dict.values() if s["total"] > 0])
            embeddings[champ_id, 6] = num_roles / len(roles)
        
        print(f"Created role-aware embeddings of shape {embeddings.shape}")
        return embeddings
    
    def get_embedding(self, champ_id):
        """Returns role-aware embedding for a champion."""
        if 0 <= champ_id < len(self.role_embeddings):
            return self.role_embeddings[champ_id]
        return torch.zeros(self.embedding_dim)
    
    def save(self):
        """Save embeddings to file."""
        torch.save(self.role_embeddings, self.embed_file)
        print(f"Role embeddings saved to {self.embed_file}")
    
    def load(self):
        """Load embeddings from file."""
        if os.path.exists(self.embed_file):
            self.role_embeddings = torch.load(self.embed_file)
            print(f"Loaded role embeddings from {self.embed_file}")


class PatchAnalyzer:
    """Analyzes patch-specific champion balance changes."""
    
    def __init__(self, session, indexer):
        """
        Args:
            session: SQLAlchemy session
            indexer: ChampionIndexer instance
        """
        self.session = session
        self.indexer = indexer
        self.patch_stats = self._compute_patch_stats()
        self.patch_file = "data/patch_stats.json"
    
    def _compute_patch_stats(self):
        """
        For each patch and champion:
        stats = {pick_rate, ban_rate, win_rate, trend}
        
        Returns:
            dict: {patch: {champ: stats}}
        """
        print("Computing patch-specific statistics...")
        patch_stats = defaultdict(lambda: defaultdict(lambda: {"picks": 0, "bans": 0, "wins": 0}))
        
        matches = self.session.query(Match).all()
        for match in matches:
            patch = match.patch
            
            # Count picks
            for pick in match.picks:
                patch_stats[patch][pick.champion]["picks"] += 1
                if pick.win:
                    patch_stats[patch][pick.champion]["wins"] += 1
            
            # Count bans
            for ban in match.bans:
                patch_stats[patch][ban.champion]["bans"] += 1
        
        # Convert to rates and add analysis
        stats_with_rates = {}
        for patch, champs in patch_stats.items():
            stats_with_rates[patch] = {}
            total_games = len([m for m in matches if m.patch == patch])
            
            for champ, data in champs.items():
                if data["picks"] > 0:
                    stats_with_rates[patch][champ] = {
                        "pick_rate": data["picks"] / max(total_games, 1),
                        "ban_rate": data["bans"] / max(total_games, 1),
                        "win_rate": data["wins"] / data["picks"],
                    }
        
        print(f"Computed stats for {len(stats_with_rates)} patches")
        return stats_with_rates
    
    def get_patch_stat(self, patch, champ, stat_type="win_rate"):
        """
        Returns patch-specific statistic for a champion.
        
        Args:
            patch: Patch string (e.g., "14.5")
            champ: Champion name
            stat_type: 'pick_rate', 'ban_rate', or 'win_rate'
        
        Returns:
            float: Stat value (0-1) or 0.5 if not found
        """
        if patch not in self.patch_stats:
            return 0.5
        if champ not in self.patch_stats[patch]:
            return 0.5
        return self.patch_stats[patch][champ].get(stat_type, 0.5)
    
    def save(self):
        """Save patch stats to JSON."""
        with open(self.patch_file, 'w') as f:
            json.dump(self.patch_stats, f)
        print(f"Patch stats saved to {self.patch_file}")
    
    def load(self):
        """Load patch stats from JSON."""
        if os.path.exists(self.patch_file):
            with open(self.patch_file, 'r') as f:
                self.patch_stats = json.load(f)
            print(f"Loaded patch stats from {self.patch_file}")


def build_all_features(session=None, indexer=None):
    """
    Build and save all feature engineering artifacts.
    
    Args:
        session: SQLAlchemy session (creates new if None)
        indexer: ChampionIndexer (creates new if None)
    """
    if session is None:
        engine = init_db(DB_URL)
        session = sessionmaker(bind=engine)()
    
    if indexer is None:
        indexer = ChampionIndexer()
        if not os.path.exists(indexer.vocab_file):
            print("Error: Vocabulary not found. Run vectorizer.py first.")
            return
        indexer.load()
    
    print("\n" + "="*50)
    print("BUILDING FEATURE ENGINEERING ARTIFACTS")
    print("="*50)
    
    # Build synergies
    synergy_calc = SynergyCalculator(session, indexer)
    synergy_calc.save()
    
    # Build counters
    counter_analyzer = CounterPickAnalyzer(session, indexer)
    counter_analyzer.save()
    
    # Build role embeddings
    role_embeddings = RoleAwareEmbeddings(session, indexer)
    role_embeddings.save()
    
    # Build patch stats
    patch_analyzer = PatchAnalyzer(session, indexer)
    patch_analyzer.save()
    
    print("="*50)
    print("All features built successfully!")
    print("="*50)


if __name__ == "__main__":
    build_all_features()
