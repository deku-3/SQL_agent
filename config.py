from pathlib import Path

SPIDER_DIR = Path(r"C:\Users\AdityaKumar\Downloads\spider_data\spider_data") 
SPIDER_DB_DIR = SPIDER_DIR / "database"
TABLES_JSON = SPIDER_DIR / "tables.json"
TRAIN_JSON = SPIDER_DIR / "train_spider.json"
DEV_JSON = SPIDER_DIR / "dev.json"
def spider_db_uri(db_id: str) -> str:
    """URI for one Spider database, e.g. spider_db_uri('flight_2')"""
    return f"sqlite:///{SPIDER_DB_DIR / db_id / f'{db_id}.sqlite'}"