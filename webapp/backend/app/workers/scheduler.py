from __future__ import annotations

import argparse
from datetime import datetime

from apscheduler.schedulers.blocking import BlockingScheduler

from ..database import Base, SessionLocal, engine
from ..models import IngestionRun
from ..services.data_service import bootstrap_database, load_master, load_registry


def refresh_once() -> None:
    load_master.cache_clear()
    load_registry.cache_clear()
    Base.metadata.create_all(bind=engine)
    with SessionLocal() as db:
        rows = bootstrap_database(db)
        db.add(
            IngestionRun(
                source="scheduled_file_refresh",
                status="ok",
                rows_loaded=rows,
                detail="Refreshed CSV and registry caches",
                finished_at=datetime.utcnow(),
            )
        )
        db.commit()
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] refreshed {rows} rows")


def main() -> None:
    parser = argparse.ArgumentParser(description="CofPred daily refresh scheduler")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--bootstrap", action="store_true")
    args = parser.parse_args()
    if args.once or args.bootstrap:
        refresh_once()
        return
    scheduler = BlockingScheduler(timezone="Asia/Bangkok")
    scheduler.add_job(refresh_once, "cron", hour=18, minute=30, id="cofpred_daily", replace_existing=True)
    print("CofPred scheduler runs daily at 18:30 Asia/Bangkok")
    scheduler.start()


if __name__ == "__main__":
    main()

