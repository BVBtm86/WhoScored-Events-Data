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

class ScrapeDataConfig(BaseModel):
    mongo: MongoConfig
    season: SeasonConfig
    email: EmailConfig
    schedule_games: ScheduleGamesConfig
