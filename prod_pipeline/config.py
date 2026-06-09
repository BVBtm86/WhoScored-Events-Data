from pydantic import BaseModel
from typing import Dict, Optional

class MongoConfig(BaseModel):
    backup_folder: str
    db: str
    url: str
    collections: Dict[str, str]

    @property
    def collection(self) -> Dict[str, str]:
        return self.collections

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
    sequence_game_features: list[str] = []

class SequenceDataConfig(BaseModel):
    base_features: list[str]
    sequence_action_types: list[str]
    shot_types: list[str]
    ignore_types: list[str]
    break_types: list[str]
    opponent_pressure_types: list[str]
    shot_feature_keywords: list[str]
    pass_feature_keywords: list[str]
    shot_metadata_features: list[str]
    pass_metadata_features: list[str]

class ScrapeDataConfig(BaseModel):
    mongo: MongoConfig
    season: SeasonConfig
    email: EmailConfig
    schedule_games: ScheduleGamesConfig
    game_stats: GameStatsConfig
    sequence_data: SequenceDataConfig
