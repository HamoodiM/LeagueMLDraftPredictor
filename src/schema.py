from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, Date, ForeignKey
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

class Match(Base):
    __tablename__ = 'matches'
    
    game_id = Column(String, primary_key=True)
    date = Column(Date)
    league = Column(String)
    split = Column(String)
    patch = Column(String)  # Crucial for ML filtering (e.g. "14.1")
    blue_team = Column(String)
    red_team = Column(String)
    winner_side = Column(String) # 'Blue' or 'Red'
    
    # Relationships
    bans = relationship("Ban", back_populates="match", cascade="all, delete-orphan")
    picks = relationship("Pick", back_populates="match", cascade="all, delete-orphan")

class Ban(Base):
    """Represents a single ban in the draft phase."""
    __tablename__ = 'bans'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    game_id = Column(String, ForeignKey('matches.game_id'))
    team_side = Column(String) # 'Blue' or 'Red'
    turn = Column(Integer)     # 1, 2, 3, 4, 5 (The order matters for ML)
    champion = Column(String)
    
    match = relationship("Match", back_populates="bans")

class Pick(Base):
    """Represents a final pick (composition)."""
    __tablename__ = 'picks'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    game_id = Column(String, ForeignKey('matches.game_id'))
    team_side = Column(String)
    role = Column(String)      # 'Top', 'Jungle', 'Mid', 'Bot', 'Sup'
    champion = Column(String)
    win = Column(Boolean)
    
    match = relationship("Match", back_populates="picks")

def init_db(db_url):
    engine = create_engine(db_url)
    Base.metadata.create_all(engine)
    return engine
