import json
import yaml
import logging
import traceback
import smtplib
import os
from pathlib import Path
from datetime import datetime, timezone
from email.message import EmailMessage
from typing import Dict, Any

import undetected_chromedriver as uc
import soccerdata._common as common

from prod_pipeline.config import ScrapeDataConfig

logger = logging.getLogger(__name__)

CHROME_MAJOR = 149

def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

def patch_soccerdata_chromedriver() -> None:
    """Patch SoccerData Selenium reader to use undetected_chromedriver."""

    def patched_init_webdriver(self):
        opts = uc.ChromeOptions()
        opts.add_argument("--no-sandbox")
        opts.add_argument("--disable-dev-shm-usage")
        opts.add_argument("--start-maximized")
        
        return uc.Chrome(options=opts, version_main=CHROME_MAJOR)

    common.BaseSeleniumReader._init_webdriver = patched_init_webdriver

def patch_soccerdata_json_loader() -> None:
    """Handle WhoScored responses where JSON is wrapped inside minimal HTML."""

    def tolerant_json_load(fp, *args, **kwargs):
        content = fp.read()
        if isinstance(content, bytes):
            content = content.decode("utf-8", errors="ignore")

        content = content.strip()
        if content.startswith("<html"):
            start = content.find("{")
            end = content.rfind("}") + 1
            if start != -1 and end > start:
                content = content[start:end]

        return json.loads(content)

    json.load = tolerant_json_load

def patch_soccerdata() -> None:
    """Apply all SoccerData runtime patches."""
    patch_soccerdata_chromedriver()
    patch_soccerdata_json_loader()

def load_yaml_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _find_env_file(start: str) -> str | None:
    current = os.path.abspath(start)
    if os.path.isfile(current):
        current = os.path.dirname(current)

    while True:
        candidate = os.path.join(current, ".env")
        if os.path.exists(candidate):
            return candidate

        parent = os.path.dirname(current)
        if parent == current:
            return None
        current = parent

def load_env_file(path: str) -> None:
    env_path = _find_env_file(path) or _find_env_file(os.getcwd())
    if not env_path:
        return

    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)

def load_app_config(path: str) -> ScrapeDataConfig:
    load_env_file(path)
    raw = load_yaml_config(path)
    config_path = Path(path).resolve()
    project_root = config_path.parent.parent

    backup_folder = raw.get("mongo", {}).get("backup_folder")
    if backup_folder:
        backup_path = Path(backup_folder).expanduser()
        if not backup_path.is_absolute():
            raw["mongo"]["backup_folder"] = str(project_root / backup_path)

    email_raw = raw.get("email", {})
    email_cfg = {
        **email_raw,
        "username": os.getenv("SMTP_EMAIL", email_raw.get("username")),
        "from_email": os.getenv("SMTP_EMAIL", email_raw.get("from_email")),
        "to_email": os.getenv("SMTP_EMAIL", email_raw.get("to_email")),
        "password": os.getenv("SMTP_PASSWORD", email_raw.get("password")),
    }

    raw["email"] = email_cfg

    return ScrapeDataConfig.parse_obj(raw)

def send_email(
    smtp_host: str,
    smtp_port: int,
    username: str,
    password: str,
    from_email: str,
    to_email: str,
    subject: str,
    body: str,
    use_tls: bool = True,
    timeout: int = 20,
) -> None:
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = from_email
    msg["To"] = to_email
    msg.set_content(body)

    try:
        if use_tls:
            with smtplib.SMTP(smtp_host, smtp_port, timeout=timeout) as server:
                server.starttls()
                server.login(username, password)
                server.send_message(msg)
        else:
            with smtplib.SMTP_SSL(smtp_host, smtp_port, timeout=timeout) as server:
                server.login(username, password)
                server.send_message(msg)

        logger.info("Sent email to %s", to_email)
    except Exception:
        logger.exception("Failed to send email to %s", to_email)
        raise

def format_exception(exc: Exception) -> str:
    return (
        f"{type(exc).__name__}: {exc}\n"
        f"Traceback:\n{traceback.format_exc()}"
    )
