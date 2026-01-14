"""
Command-Line Interface and API for SummonerDraft.

Provides:
- CLI for ingest, train, predict operations
- Model serving capabilities
- Performance benchmarking
"""

import argparse
import sys
import os
import time
import json
from datetime import datetime
import torch
from sqlalchemy.orm import sessionmaker

from .schema import init_db, Match
from .config import DB_URL
from .vectorizer import ChampionIndexer, DraftVectorizer
from .model import DraftRecommenderNet
from .inference import load_system, predict_pick
from .inference_win import DraftEngine
from .features import (
    build_all_features,
    SynergyCalculator,
    CounterPickAnalyzer,
    RoleAwareEmbeddings,
    PatchAnalyzer
)


class ModelServer:
    """Serves models and provides predictions via API."""
    
    def __init__(self):
        self.pick_model, self.indexer = None, None
        self.win_engine = None
        self.load_models()
    
    def load_models(self):
        """Load both pick recommendation and win prediction models."""
        try:
            print("Loading Pick Recommendation Model...")
            self.pick_model, self.indexer = load_system()
            print("✓ Pick model loaded")
            
            print("Loading Win Prediction Engine...")
            self.win_engine = DraftEngine()
            print("✓ Win prediction engine loaded")
        except FileNotFoundError as e:
            print(f"❌ Model Loading Error: {e}")
            raise
    
    def recommend_picks(self, my_team, enemy_team, bans, top_k=5):
        """
        Get pick recommendations.
        
        Args:
            my_team: List of champion names on my team
            enemy_team: List of champion names on enemy team
            bans: List of banned champion names
            top_k: Number of recommendations to return
        
        Returns:
            List of (champion_name, probability) tuples
        """
        if not self.pick_model:
            raise RuntimeError("Pick model not loaded")
        
        try:
            recs = predict_pick(self.pick_model, self.indexer, my_team, enemy_team, bans)
            return recs[:top_k]
        except Exception as e:
            print(f"❌ Recommendation Error: {e}")
            raise
    
    def recommend_with_winrate(self, my_team, enemy_team, bans, target_role=None, top_k=10):
        """
        Get win-rate-based recommendations.
        
        Args:
            my_team: List of champion names on my team
            enemy_team: List of champion names on enemy team
            bans: List of banned champion names
            target_role: Optional role filter (top/jng/mid/bot/sup)
            top_k: Number of recommendations to return
        
        Returns:
            List of (champion_name, win_probability) tuples
        """
        if not self.win_engine:
            raise RuntimeError("Win prediction engine not loaded")
        
        try:
            recs = self.win_engine.get_recommendations(my_team, enemy_team, bans, target_role)
            return recs[:top_k]
        except Exception as e:
            print(f"❌ Win Recommendation Error: {e}")
            raise


def cmd_ingest(args):
    """Command: Ingest data from CSV."""
    print("\n" + "="*50)
    print("DATA INGESTION")
    print("="*50)
    
    from .ingest import run_ingestion
    run_ingestion()


def cmd_train(args):
    """Command: Train pick recommendation model."""
    print("\n" + "="*50)
    print("TRAIN PICK RECOMMENDATION MODEL")
    print("="*50)
    
    from .train import train
    train()


def cmd_train_win(args):
    """Command: Train win prediction model."""
    print("\n" + "="*50)
    print("TRAIN WIN PREDICTION MODEL")
    print("="*50)
    
    from .train_win import train
    train()


def cmd_features(args):
    """Command: Build feature engineering artifacts."""
    print("\n" + "="*50)
    print("BUILD FEATURE ENGINEERING ARTIFACTS")
    print("="*50)
    
    build_all_features()


def cmd_predict(args):
    """Command: Interactive prediction shell."""
    print("\n" + "="*50)
    print("DRAFT ASSISTANT - PICK RECOMMENDATION")
    print("="*50)
    
    try:
        model, indexer = load_system()
        
        # Parse input
        def parse_list(s):
            return [x.strip() for x in s.split(',') if x.strip()]
        
        my_team = parse_list(args.my_team) if args.my_team else []
        enemy_team = parse_list(args.enemy_team) if args.enemy_team else []
        bans = parse_list(args.bans) if args.bans else []
        
        print(f"\nAnalysis:")
        print(f"  My Team: {my_team}")
        print(f"  Enemy: {enemy_team}")
        print(f"  Bans: {bans}")
        
        # Validate
        invalid = []
        for champ in my_team + enemy_team + bans:
            if indexer.get_id(champ) == 1:
                invalid.append(champ)
        
        if invalid:
            print(f"❌ Unknown champions: {invalid}")
            return
        
        # Predict
        recs = predict_pick(model, indexer, my_team, enemy_team, bans)
        
        print("\n" + "="*40)
        print("RECOMMENDATIONS")
        print("="*40)
        for i, (name, prob) in enumerate(recs, 1):
            bar_length = int(prob * 20)
            bar = "█" * bar_length + "░" * (20 - bar_length)
            print(f"{i}. {name:20s} {bar} {prob*100:5.1f}%")
            
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


def cmd_predict_win(args):
    """Command: Interactive win prediction shell."""
    print("\n" + "="*50)
    print("DRAFT ASSISTANT - WIN PREDICTION")
    print("="*50)
    
    try:
        engine = DraftEngine()
        
        # Parse input
        def parse_list(s):
            return [x.strip() for x in s.split(',') if x.strip()]
        
        my_team = parse_list(args.my_team) if args.my_team else []
        enemy_team = parse_list(args.enemy_team) if args.enemy_team else []
        bans = parse_list(args.bans) if args.bans else []
        
        print(f"\nAnalysis:")
        print(f"  My Team: {my_team}")
        print(f"  Enemy: {enemy_team}")
        print(f"  Bans: {bans}")
        
        # Predict
        recs = engine.get_recommendations(my_team, enemy_team, bans, args.role)
        
        print("\n" + "="*40)
        print("TOP WIN RECOMMENDATIONS")
        print("="*40)
        for i, (name, win_prob) in enumerate(recs, 1):
            bar_length = int(win_prob * 20)
            bar = "█" * bar_length + "░" * (20 - bar_length)
            print(f"{i:2d}. {name:20s} {bar} {win_prob*100:5.1f}%")
            
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


def cmd_benchmark(args):
    """Command: Benchmark model performance."""
    print("\n" + "="*50)
    print("MODEL PERFORMANCE BENCHMARK")
    print("="*50)
    
    try:
        engine = init_db(DB_URL)
        session = sessionmaker(bind=engine)()
        
        # Load models
        print("\nLoading models...")
        model_server = ModelServer()
        
        # Get test data
        print("Loading test matches...")
        matches = session.query(Match).limit(100).all()
        if not matches:
            print("❌ No matches found for benchmarking")
            return
        
        print(f"✓ Loaded {len(matches)} test matches\n")
        
        # Benchmark pick model
        print("PICK RECOMMENDATION MODEL")
        print("-" * 40)
        times = []
        for match in matches[:10]:
            blue_picks = [p.champion for p in match.picks if p.team_side == 'Blue']
            red_picks = [p.champion for p in match.picks if p.team_side == 'Red']
            bans = [b.champion for b in match.bans]
            
            start = time.time()
            model_server.recommend_picks(blue_picks[:2], red_picks[:2], bans[:3])
            elapsed = time.time() - start
            times.append(elapsed)
        
        avg_time = sum(times) / len(times) * 1000  # ms
        print(f"Avg inference time: {avg_time:.2f}ms")
        print(f"Throughput: {1000/avg_time:.1f} requests/sec")
        
        # Benchmark win model
        print("\nWIN PREDICTION MODEL")
        print("-" * 40)
        times = []
        for match in matches[:10]:
            blue_picks = [p.champion for p in match.picks if p.team_side == 'Blue']
            red_picks = [p.champion for p in match.picks if p.team_side == 'Red']
            bans = [b.champion for b in match.bans]
            
            start = time.time()
            model_server.recommend_with_winrate(blue_picks[:2], red_picks[:2], bans[:3])
            elapsed = time.time() - start
            times.append(elapsed)
        
        avg_time = sum(times) / len(times) * 1000  # ms
        print(f"Avg inference time: {avg_time:.2f}ms")
        print(f"Throughput: {1000/avg_time:.1f} requests/sec")
        
        print("\n✓ Benchmark Complete")
        
    except Exception as e:
        print(f"❌ Benchmark Error: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="SummonerDraft - League of Legends Draft AI Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.cli ingest
  python -m src.cli train
  python -m src.cli train-win
  python -m src.cli features
  python -m src.cli predict --my-team "Aatrox,Ahri" --enemy-team "Ashe" --bans "Jax,Kai'Sa"
  python -m src.cli predict-win --my-team "Aatrox" --enemy-team "Ashe,Skarner" --bans "Jax"
  python -m src.cli benchmark
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Ingest command
    subparsers.add_parser('ingest', help='Ingest data from CSV into database')
    
    # Train command
    subparsers.add_parser('train', help='Train pick recommendation model')
    
    # Train Win command
    subparsers.add_parser('train-win', help='Train win prediction model')
    
    # Features command
    subparsers.add_parser('features', help='Build feature engineering artifacts')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Get pick recommendations')
    predict_parser.add_argument('--my-team', type=str, help='Comma-separated list of your team picks')
    predict_parser.add_argument('--enemy-team', type=str, help='Comma-separated list of enemy picks')
    predict_parser.add_argument('--bans', type=str, help='Comma-separated list of banned champions')
    
    # Predict Win command
    predict_win_parser = subparsers.add_parser('predict-win', help='Get win-rate-based recommendations')
    predict_win_parser.add_argument('--my-team', type=str, help='Comma-separated list of your team picks')
    predict_win_parser.add_argument('--enemy-team', type=str, help='Comma-separated list of enemy picks')
    predict_win_parser.add_argument('--bans', type=str, help='Comma-separated list of banned champions')
    predict_win_parser.add_argument('--role', type=str, help='Filter by role (top/jng/mid/bot/sup)')
    
    # Benchmark command
    subparsers.add_parser('benchmark', help='Benchmark model performance')
    
    args = parser.parse_args()
    
    # Dispatch command
    commands = {
        'ingest': cmd_ingest,
        'train': cmd_train,
        'train-win': cmd_train_win,
        'features': cmd_features,
        'predict': cmd_predict,
        'predict-win': cmd_predict_win,
        'benchmark': cmd_benchmark,
    }
    
    if args.command not in commands:
        parser.print_help()
        return 1
    
    try:
        commands[args.command](args)
        return 0
    except KeyboardInterrupt:
        print("\n⚠ Interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
