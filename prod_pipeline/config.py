from pydantic import BaseModel
from typing import Dict, Optional

class MongoConfig(BaseModel):
    backup_folder: str
    db: str
    url: str
    collection: Dict[str, str]

class SeasonConfig(BaseModel):
    year: str
    league: str
    name: str
    country: str

class EmailConfig(BaseModel):
    smtp_host: str
    smtp_port: int
    username: Optional[str] = None
    from_email: Optional[str] = None
    to_email: Optional[str] = None
    use_tls: bool = True
    password: Optional[str] = None

class ScheduleGamesConfig(BaseModel):
    finished_status_code: int
    required_columns: list[str]

class GameStatsConfig(BaseModel):
    team_match_features: list[str]
    player_match_features: list[str]
    match_stats: list[str]
    pass_complettion_stats: list[str]
    perc_stats: Dict[str, Dict[str, str]]
    match_stats_rename: Dict[str, str] = {}

class ScrapeDataConfig(BaseModel):
    mongo: MongoConfig
    season: SeasonConfig
    email: EmailConfig
    schedule_games: ScheduleGamesConfig
    game_stats: GameStatsConfig
