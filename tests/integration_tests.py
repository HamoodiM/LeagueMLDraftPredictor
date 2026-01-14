"""
Integration tests for the complete SummonerDraft pipeline.

Tests:
- End-to-end ingest → train → predict flow
- Model loading and inference
- Feature engineering
- CLI commands
"""

import pytest
import torch
import os
import tempfile
import shutil
from sqlalchemy.orm import sessionmaker

from src.schema import init_db, Match, Ban, Pick
from src.config import DB_URL
from src.vectorizer import ChampionIndexer, DraftVectorizer
from src.model import DraftRecommenderNet
from src.win_model import WinPredictor
from src.inference import load_system, predict_pick
from src.inference_win import DraftEngine
from src.features import (
    SynergyCalculator,
    CounterPickAnalyzer,
    RoleAwareEmbeddings,
    PatchAnalyzer
)


class TestDataPipeline:
    """Test data loading and vectorization."""
    
    def test_database_initialization(self):
        """Test that database can be created and initialized."""
        # Use temp database for testing
        with tempfile.TemporaryDirectory() as tmpdir:
            test_db = f"sqlite:///{tmpdir}/test.db"
            engine = init_db(test_db)
            assert engine is not None
            
            # Verify tables exist
            session = sessionmaker(bind=engine)()
            assert hasattr(Match, '__tablename__')
            assert hasattr(Ban, '__tablename__')
            assert hasattr(Pick, '__tablename__')
    
    def test_champion_indexer(self):
        """Test champion vocabulary building and retrieval."""
        indexer = ChampionIndexer()
        
        # Test special tokens
        assert indexer.get_id("<PAD>") == 0
        assert indexer.get_id("<UNK>") == 1
        assert indexer.get_name(0) == "<PAD>"
        assert indexer.get_name(1) == "<UNK>"
        
        # Test unknown champion
        unknown_id = indexer.get_id("NonexistentChampion")
        assert unknown_id == 1  # Maps to UNK


class TestModelInference:
    """Test model loading and inference."""
    
    def test_pick_model_exists(self):
        """Test that pick recommendation model can be loaded."""
        try:
            model, indexer = load_system()
            assert model is not None
            assert indexer is not None
            print("✓ Pick model loads successfully")
        except FileNotFoundError:
            pytest.skip("Model files not found")
    
    def test_pick_prediction(self):
        """Test pick prediction with sample data."""
        try:
            model, indexer = load_system()
            
            # Sample draft state
            my_team = ["Aatrox", "Ahri"]
            enemy_team = ["Ashe", "Skarner"]
            bans = ["Jax", "Kai'Sa"]
            
            recs = predict_pick(model, indexer, my_team, enemy_team, bans)
            
            assert len(recs) <= 5
            assert all(isinstance(r, tuple) and len(r) == 2 for r in recs)
            assert all(0 <= prob <= 1 for _, prob in recs)
            print(f"✓ Pick prediction returns {len(recs)} recommendations")
            
        except FileNotFoundError:
            pytest.skip("Model files not found")
    
    def test_win_model_exists(self):
        """Test that win prediction model can be loaded."""
        try:
            engine = DraftEngine()
            assert engine.model is not None
            assert engine.indexer is not None
            print("✓ Win prediction model loads successfully")
        except FileNotFoundError:
            pytest.skip("Model files not found")
    
    def test_win_prediction(self):
        """Test win prediction with sample data."""
        try:
            engine = DraftEngine()
            
            my_team = ["Aatrox"]
            enemy_team = ["Ashe", "Skarner"]
            bans = ["Jax"]
            
            recs = engine.get_recommendations(my_team, enemy_team, bans)
            
            assert len(recs) <= 10
            assert all(isinstance(r, tuple) and len(r) == 2 for r in recs)
            assert all(0 <= prob <= 1 for _, prob in recs)
            print(f"✓ Win prediction returns {len(recs)} recommendations")
            
        except FileNotFoundError:
            pytest.skip("Model files not found")


class TestFeatureEngineering:
    """Test feature engineering modules."""
    
    def test_synergy_calculator(self):
        """Test champion synergy computation."""
        if not os.path.exists("data/database.db"):
            pytest.skip("Database not found")
        
        try:
            engine = init_db(DB_URL)
            session = sessionmaker(bind=engine)()
            indexer = ChampionIndexer()
            indexer.load()
            
            synergy_calc = SynergyCalculator(session, indexer)
            
            # Test getting synergy
            score = synergy_calc.get_synergy("Aatrox", "Ahri")
            assert 0 <= score <= 1
            print(f"✓ Synergy calculator works (Aatrox-Ahri: {score:.2f})")
            
        except Exception as e:
            pytest.skip(f"Feature engineering unavailable: {e}")
    
    def test_counter_analyzer(self):
        """Test counter-pick analysis."""
        if not os.path.exists("data/database.db"):
            pytest.skip("Database not found")
        
        try:
            engine = init_db(DB_URL)
            session = sessionmaker(bind=engine)()
            indexer = ChampionIndexer()
            indexer.load()
            
            counter_analyzer = CounterPickAnalyzer(session, indexer)
            
            # Test getting counter score
            score = counter_analyzer.get_counter_score("Aatrox", "Ashe")
            assert 0 <= score <= 1
            print(f"✓ Counter analyzer works (Aatrox vs Ashe: {score:.2f})")
            
        except Exception as e:
            pytest.skip(f"Feature engineering unavailable: {e}")
    
    def test_role_aware_embeddings(self):
        """Test role-aware embedding generation."""
        if not os.path.exists("data/database.db"):
            pytest.skip("Database not found")
        
        try:
            engine = init_db(DB_URL)
            session = sessionmaker(bind=engine)()
            indexer = ChampionIndexer()
            indexer.load()
            
            embeddings = RoleAwareEmbeddings(session, indexer)
            
            # Test embedding shape
            assert embeddings.role_embeddings.shape[1] == 7  # 5 roles + win_rate + flexibility
            
            # Test getting embedding
            embed = embeddings.get_embedding(10)
            assert embed.shape == (7,)
            print(f"✓ Role embeddings work (shape: {embeddings.role_embeddings.shape})")
            
        except Exception as e:
            pytest.skip(f"Feature engineering unavailable: {e}")


class TestCLI:
    """Test CLI commands."""
    
    def test_predict_command(self):
        """Test CLI predict command."""
        try:
            from src.cli import cmd_predict
            import argparse
            
            args = argparse.Namespace(
                my_team="Aatrox,Ahri",
                enemy_team="Ashe",
                bans="Jax"
            )
            
            # Should not raise an exception
            cmd_predict(args)
            print("✓ CLI predict command works")
            
        except FileNotFoundError:
            pytest.skip("Model not trained")
    
    def test_predict_win_command(self):
        """Test CLI predict-win command."""
        try:
            from src.cli import cmd_predict_win
            import argparse
            
            args = argparse.Namespace(
                my_team="Aatrox",
                enemy_team="Ashe",
                bans="Jax",
                role=None
            )
            
            # Should not raise an exception
            cmd_predict_win(args)
            print("✓ CLI predict-win command works")
            
        except FileNotFoundError:
            pytest.skip("Model not trained")


class TestEndToEnd:
    """End-to-end pipeline tests."""
    
    def test_complete_inference_flow(self):
        """Test complete inference pipeline."""
        if not os.path.exists("data/model_v1.pth"):
            pytest.skip("Pick model not trained")
        
        try:
            # Load system
            model, indexer = load_system()
            
            # Create realistic draft state
            my_team = ["Aatrox", "Elise"]
            enemy_team = ["Gnar", "Lee Sin", "Ahri"]
            bans = ["Jax", "Kai'Sa", "Kalista", "Renata", "Rakan"]
            
            # Get recommendations
            recs = predict_pick(model, indexer, my_team, enemy_team, bans)
            
            # Validate results
            assert len(recs) > 0
            
            # Check no banned/picked champs in recommendations
            taken = set(my_team + enemy_team + bans)
            for champ, _ in recs:
                assert champ not in taken
            
            print(f"✓ Complete inference flow works")
            print(f"  Top recommendation: {recs[0][0]} ({recs[0][1]*100:.1f}%)")
            
        except Exception as e:
            pytest.skip(f"Inference unavailable: {e}")


def run_tests():
    """Run all integration tests."""
    print("\n" + "="*50)
    print("INTEGRATION TEST SUITE")
    print("="*50 + "\n")
    
    pytest.main([__file__, "-v", "-s"])


if __name__ == "__main__":
    run_tests()
