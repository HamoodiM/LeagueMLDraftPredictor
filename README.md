# SummonerDraft

A machine learning system for League of Legends draft pick recommendations.

## Setup

1. Install dependencies:
```bash
python -m pip install -r requirements.txt
```

2. Prepare data:
   - Download CSV from Oracle's Elixir: https://oracleselixir.com/
   - Place in: `data/raw/2024_LoL_esports_match_data_from_OraclesElixir.csv`

## Usage

### Ingest Data
```bash
python -m src.ingest
```
Loads CSV data into SQLite database and builds champion vocabulary.

### Train Model
```bash
python -m src.train
```
Trains the neural network on winning team picks (50 epochs).

### Make Predictions
```bash
python -m src.inference
```
Interactive shell for draft recommendations. Enter comma-separated champion names:
```
My Team Picks (so far): Yone, Gnar, Sejuani, Caitlyn
Enemy Team Picks: Ashe, Skarner, Sylas, Rumble, Seraphine
Bans (Global): Jax, Kai'Sa, Kalista, Renata Glasc, Rakan, Jhin, Ziggs, Varus, Vi, Aurora
```

### Run Tests
```bash
python tests/test_suite.py
```

## Model Accuracy

Current stage: Early training (10 epochs baseline, 50+ recommended)
- Expected accuracy: 25-40% after full training
- Baseline: 20% (random from 5 champions)
- Improves with more training epochs and data

## Architecture

- **Data**: CSV → SQLite (Match, Ban, Pick tables)
- **Vectorization**: Champion embeddings + multi-hot pick encoding
- **Model**: Neural network with 3 hidden layers (256→256→256)
- **Training**: CrossEntropyLoss, Adam optimizer, masking strategy for pick generation
- **Inference**: Logit masking, top-5 recommendations with softmax probabilities

## Files

- `src/config.py` - Database configuration
- `src/schema.py` - Database models
- `src/ingest.py` - Data loading from CSV
- `src/vectorizer.py` - Champion indexing and tensor encoding
- `src/model.py` - Neural network architecture
- `src/train.py` - Training loop
- `src/inference.py` - Interactive prediction shell
- `tests/test_suite.py` - Unit tests (17 tests, all passing)

## Requirements

- Python 3.8+
- PyTorch
- SQLAlchemy
- pandas
- tqdm

See `requirements.txt` for full list.
