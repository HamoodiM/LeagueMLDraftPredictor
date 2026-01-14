# SummonerDraft

Advanced machine learning system for League of Legends draft pick recommendations and win probability analysis.

## Quick Setup

### 1. Install Dependencies
```bash
python -m pip install -r requirements.txt
```

### 2. Prepare Data
- Download CSV from [Oracle's Elixir](https://oracleselixir.com/)
- Place in: `data/raw/2024_LoL_esports_match_data_from_OraclesElixir.csv`

### 3. Run Complete Pipeline
```bash
python -m src.cli ingest      # Load data into database
python -m src.cli train       # Train pick recommendation model
python -m src.cli train-win   # Train win prediction model
python -m src.cli features    # Build feature engineering artifacts
```

## Usage

### Command-Line Interface (Recommended)

**Pick Recommendations** (top 5 by model prediction):
```bash
python -m src.cli predict --my-team "Aatrox,Ahri" --enemy-team "Ashe" --bans "Jax,Kai'Sa"
```

**Win Predictions** (top 10 by win probability):
```bash
python -m src.cli predict-win --my-team "Aatrox" --enemy-team "Ashe,Skarner" --bans "Jax" --role mid
```

**Interactive Shells**:
```bash
python -m src.inference      # Pick recommendations interactive mode
python -m src.inference_win  # Win predictions interactive mode
```

**Benchmarking**:
```bash
python -m src.cli benchmark  # Measure model performance
```

**View Help**:
```bash
python -m src.cli --help     # All commands
python -m src.cli predict --help
python -m src.cli predict-win --help
```

## Models

### Pick Recommendation Model
- **Architecture**: Neural network with batch normalization (256→256→256 hidden dims)
- **Input**: Blue/Red team picks (multi-hot) + bans (embeddings)
- **Output**: Top 5 champion recommendations
- **Training**: Masking strategy, 70/15/15 train/val/test splits
- **Expected Accuracy**: 25-40% (baseline: 20% random)

### Win Prediction Model
- **Architecture**: Symmetric dual-team encoder
- **Input**: Partial team picks + global bans
- **Output**: Win probability (0-1)
- **Training**: Binary cross-entropy, random partial draft augmentation
- **Use Case**: Predict winner from early draft state

## Features

### Feature Engineering (`src/features.py`)
- **Synergy Calculator**: Champion pair win rates (4171+ pairs)
- **Counter Analyzer**: 1v1 matchup statistics (168+ champions)
- **Role Embeddings**: 7D role-aware champion vectors
- **Patch Analysis**: Meta statistics per patch version

Python API Example:
```python
from src.features import SynergyCalculator, CounterPickAnalyzer

synergy = SynergyCalculator(session, indexer)
score = synergy.get_synergy("Ahri", "Aatrox")  # 0-1

counter = CounterPickAnalyzer(session, indexer)
matchup = counter.get_counter_score("Aatrox", "Ashe")  # How well Aatrox counters Ashe
```

## Testing

### Unit Tests (8/8 passing)
```bash
python tests/test_suite.py
```
Tests champion indexer, vectorization, neural networks, masking, probabilities

### Integration Tests
```bash
pytest tests/integration_tests.py -v -s
```
End-to-end pipeline, model loading, feature engineering, CLI commands

### Validation
- Train/Validation/Test splits (70/15/15)
- Best model checkpointing
- Per-epoch accuracy tracking
- Final test set evaluation

## Architecture

### Data Pipeline
```
CSV → Ingest → SQLite DB
                  ↓
          Vectorizer (Encode)
                  ↓
         Feature Engineering
    (Synergies, Counters, Embeddings, Patch Stats)
                  ↓
            Train Dataset
         ├─ Pick Dataset → DraftRecommenderNet
         └─ Win Dataset → WinPredictor
                  ↓
            Checkpoint Best
                  ↓
            Evaluate on Test
                  ↓
            CLI/API Inference
```

### Database Schema
- **Match**: Game metadata (date, league, patch, winner)
- **Ban**: Banned champion (team_side, turn order)
- **Pick**: Selected champion (team_side, role, win status)

### Key Modules
| Module | Purpose |
|--------|---------|
| `config.py` | Database configuration |
| `schema.py` | SQLAlchemy database models |
| `ingest.py` | CSV → SQLite with error handling |
| `vectorizer.py` | Champion indexing & tensor encoding |
| `model.py` | Pick recommendation network |
| `win_model.py` | Win prediction network |
| `train.py` | Pick model training with validation/test |
| `train_win.py` | Win model training with checkpointing |
| `inference.py` | Pick prediction interactive shell |
| `inference_win.py` | Win prediction interactive shell |
| `features.py` | Advanced feature engineering |
| `cli.py` | Unified command-line interface |

## Examples

### Complete Workflow
```bash
# Setup
python -m src.cli ingest
python -m src.cli train
python -m src.cli train-win
python -m src.cli features

# Quick prediction
python -m src.cli predict --my-team "Aatrox" --enemy-team "Ashe" --bans "Jax"
```

### Interactive Mode
```bash
python -m src.inference_win
# Commands: P(ick), E(nemy), B(an), R(ecommend), C(lear), e(X)it, Help
```

### Feature Analysis
```bash
python -m src.cli features  # Compute all feature artifacts
# Generates: synergies.json, counters.json, role_embeddings.pth, patch_stats.json
```

### Performance Testing
```bash
python -m src.cli benchmark  # Latency and throughput for both models
python tests/test_suite.py   # Unit tests
pytest tests/integration_tests.py -v  # Integration tests
```

## Requirements

- Python 3.8+
- PyTorch 2.1.0+
- SQLAlchemy 2.0+
- pandas 2.3+
- numpy 1.26+
- tqdm 4.66+
- pytest (for integration tests)

See `requirements.txt` for exact versions.

## Error Handling

All modules include comprehensive error handling:
- ✓ Graceful fallbacks for missing models
- ✓ Input validation for champion names
- ✓ Database transaction rollbacks
- ✓ Informative error messages with next steps
- ✓ Keyboard interrupt handling

## Development

### Adding New Features
1. Create feature in `src/features.py`
2. Update models if needed
3. Retrain: `python -m src.cli train`

### Extending CLI
Edit `src/cli.py` to add new commands

### Contributing
- Run tests before committing: `python tests/test_suite.py`
- Keep error messages user-friendly
- Document new functions with docstrings

## Performance

- **Inference Speed**: ~5-10ms per pick prediction, ~50-100ms per win prediction
- **Training Time**: ~5-10 minutes per model for 50 epochs
- **Feature Generation**: ~2-5 seconds (one-time)
- **Database Size**: ~10-100MB depending on data

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Vocab not found" | Run `python -m src.cli ingest` |
| "Model not found" | Run `python -m src.cli train` and `python -m src.cli train-win` |
| "Unknown champion" | Check spelling and capitalization (case-sensitive) |
| No recommendations | Ensure not all champions are banned/picked |

## License

MIT
