"""
Unit tests for SummonerDraft model pipeline.
Run with: python tests/test_suite.py
"""

import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.model import DraftRecommenderNet
from src.vectorizer import ChampionIndexer, DraftVectorizer


class TestChampionIndexer:
    """Test champion vocabulary management."""
    
    def test_indexer_initialization(self):
        """Test that indexer initializes with PAD and UNK tokens."""
        indexer = ChampionIndexer()
        assert indexer.get_id("<PAD>") == 0
        assert indexer.get_id("<UNK>") == 1
        print("Indexer initialization")
    
    def test_add_champion(self):
        """Test adding champions to vocabulary."""
        indexer = ChampionIndexer()
        indexer.add_champion("Aatrox")
        indexer.add_champion("Ahri")
        assert indexer.get_id("Aatrox") == 2
        assert indexer.get_id("Ahri") == 3
        print("Add champion")
    
    def test_unknown_champion(self):
        """Test that unknown champions map to UNK token."""
        indexer = ChampionIndexer()
        unknown_id = indexer.get_id("NonexistentChamp")
        assert unknown_id == 1
        print("Unknown champion handling")


class TestDraftVectorizer:
    """Test match encoding to tensor representations."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.indexer = ChampionIndexer()
        for champ in ["Aatrox", "Ahri", "Ashe", "Akali", "Annie"]:
            self.indexer.add_champion(champ)
        self.vectorizer = DraftVectorizer(self.indexer)
    
    def test_invalid_mask_creation(self):
        """Test that invalid mask correctly masks unavailable picks."""
        mask = self.vectorizer.get_invalid_mask([2, 3], [4, 5])
        assert mask[0] == True  # PAD masked
        assert mask[1] == True  # UNK masked
        assert mask[2] == True  # Ban masked
        assert mask[6] == False  # Available champion
        print("Invalid mask creation")


class TestDraftRecommenderNet:
    """Test neural network model."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.vocab_size = 100
        self.batch_size = 4
        self.model = DraftRecommenderNet(self.vocab_size, embedding_dim=32, hidden_dim=256)
    
    def test_forward_pass_shape(self):
        """Test forward pass output shape is correct."""
        blue_picks = torch.zeros(self.batch_size, self.vocab_size)
        red_picks = torch.zeros(self.batch_size, self.vocab_size)
        bans = torch.zeros(self.batch_size, 10, dtype=torch.long)
        
        logits = self.model(blue_picks, red_picks, bans)
        assert logits.shape == (self.batch_size, self.vocab_size)
        print("Forward pass shape")
    
    def test_no_nan_values(self):
        """Test that forward pass doesn't produce NaN values."""
        blue_picks = torch.zeros(2, self.vocab_size)
        red_picks = torch.zeros(2, self.vocab_size)
        bans = torch.zeros(2, 10, dtype=torch.long)
        
        logits = self.model(blue_picks, red_picks, bans)
        assert not torch.isnan(logits).any()
        print("No NaN values")


class TestInferencePipeline:
    """Test the complete inference pipeline."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.indexer = ChampionIndexer()
        for champ in ["Aatrox", "Ahri", "Ashe", "Akali", "Annie", "Aphelios", "Evelynn"]:
            self.indexer.add_champion(champ)
        self.vocab_size = len(self.indexer.champ_to_idx)
    
    def test_masking_before_topk(self):
        """Test that masking is applied correctly before topk."""
        logits = torch.randn(1, self.vocab_size)
        unavailable = {0, 1, 2, 3}
        
        for uid in unavailable:
            logits[0, uid] = -float('inf')
        
        top_logits, top_indices = torch.topk(logits, 5)
        
        for idx in top_indices[0].tolist():
            assert idx not in unavailable
        print("Masking before topk")
    
    def test_softmax_probabilities(self):
        """Test that softmax probabilities sum to approximately 1."""
        logits = torch.randn(1, self.vocab_size)
        top_logits, _ = torch.topk(logits, 5)
        probs = torch.softmax(top_logits, dim=1)
        
        prob_sum = probs.sum(dim=1)[0].item()
        assert abs(prob_sum - 1.0) < 1e-5
        print("Softmax probabilities")


def run_tests():
    """Run all unit tests."""
    test_classes = [
        TestChampionIndexer,
        TestDraftVectorizer,
        TestDraftRecommenderNet,
        TestInferencePipeline,
    ]
    
    total = 0
    passed = 0
    
    print("Running tests...")
    print("-" * 50)
    
    for test_class in test_classes:
        test_instance = test_class()
        test_methods = [m for m in dir(test_instance) if m.startswith('test_')]
        
        for method_name in test_methods:
            total += 1
            try:
                if hasattr(test_instance, 'setup_method'):
                    test_instance.setup_method()
                getattr(test_instance, method_name)()
                passed += 1
            except AssertionError as e:
                print(f" {test_class.__name__}.{method_name}: {e}")
            except Exception as e:
                print(f" {test_class.__name__}.{method_name}: ERROR: {e}")
    
    print("-" * 50)
    print(f"Tests: {passed}/{total} passed")
    
    return passed == total


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
