import pandas as pd
import os
from sqlalchemy.orm import sessionmaker
from .schema import init_db, Match, Ban, Pick
from .config import DB_URL
from tqdm import tqdm

# --- CONFIGURATION ---
CSV_PATH = "data/raw/2024_LoL_esports_match_data_from_OraclesElixir.csv"

def process_bans(session, row, game_id, side):
    """Extracts bans from a team-summary row."""
    # Oracle's Elixir columns: 'ban1', 'ban2', 'ban3', 'ban4', 'ban5'
    for i in range(1, 6):
        col_name = f"ban{i}"
        champ = row.get(col_name)
        
        # Check if ban exists and isn't empty
        if pd.notna(champ) and champ != "":
            ban = Ban(
                game_id=game_id,
                team_side=side,
                turn=i,
                champion=champ
            )
            session.add(ban)

def process_picks(session, player_rows, game_id):
    """Extracts picks from the 10 player rows."""
    for _, row in player_rows.iterrows():
        pick = Pick(
            game_id=game_id,
            team_side=row['side'],    # 'Blue' or 'Red'
            role=row['position'],     # 'top', 'jng', 'mid', 'bot', 'sup'
            champion=row['champion'],
            win=(row['result'] == 1)
        )
        session.add(pick)

def run_ingestion():
    print(f"Connecting to DB at {DB_URL}...")
    engine = init_db(DB_URL)
    Session = sessionmaker(bind=engine)
    session = Session()

    print(f"Reading CSV from {CSV_PATH}...")
    # Read only necessary columns to save memory
    needed_cols = [
        'gameid', 'date', 'league', 'split', 'patch', 'side', 'position', 
        'teamname', 'result', 'champion', 'ban1', 'ban2', 'ban3', 'ban4', 'ban5'
    ]
    try:
        df = pd.read_csv(CSV_PATH, usecols=lambda c: c in needed_cols)
    except FileNotFoundError:
        print("Error: CSV file not found. Please download from Oracle's Elixir and place in data/raw/")
        return

    # Filter for complete games (Oracle's sometimes has partial data)
    # We group by gameid to process one match at a time
    unique_games = df['gameid'].unique()
    print(f"Found {len(unique_games)} matches. Starting ingestion...")

    for game_id in tqdm(unique_games):
        # Slice the dataframe for this specific game
        game_rows = df[df['gameid'] == game_id]
        
        # Metadata comes from any row (taking the first one)
        meta = game_rows.iloc[0]
        
        # Determine winner
        winner = 'Blue' if game_rows[(game_rows['side'] == 'Blue') & (game_rows['result'] == 1)].shape[0] > 0 else 'Red'

        # Create Match Object
        match = Match(
            game_id=str(game_id),
            date=pd.to_datetime(meta['date']).date(),
            league=meta['league'],
            split=meta['split'],
            patch=str(meta['patch']),
            blue_team=game_rows[game_rows['side'] == 'Blue']['teamname'].iloc[0],
            red_team=game_rows[game_rows['side'] == 'Red']['teamname'].iloc[0],
            winner_side=winner
        )
        session.add(match)

        # Process Bans (Use rows where position == 'team')
        team_rows = game_rows[game_rows['position'] == 'team']
        for _, row in team_rows.iterrows():
            process_bans(session, row, str(game_id), row['side'])

        # Process Picks (Use rows where position != 'team')
        player_rows = game_rows[game_rows['position'] != 'team']
        process_picks(session, player_rows, str(game_id))

    session.commit()
    print("Ingestion Complete!")

if __name__ == "__main__":
    run_ingestion()
