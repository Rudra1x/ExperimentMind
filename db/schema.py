import os
from sqlalchemy import (
    create_engine, Column, String, Float,
    Integer, DateTime, Text, Boolean
)
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.sql import func
from dotenv import load_dotenv

load_dotenv()

Base = declarative_base()
DB_PATH = os.getenv("DB_PATH", "db/experimentmind.db")


class Experiment(Base):
    __tablename__ = "experiments"

    run_id          = Column(String, primary_key=True)
    experiment_name = Column(String, nullable=False)
    config_path     = Column(String)
    config_hash     = Column(String)
    status          = Column(String, default="queued")   # queued | running | completed | failed
    executor        = Column(String, default="local")    # local | kaggle
    primary_metric  = Column(String)
    created_at      = Column(DateTime, default=func.now())
    started_at      = Column(DateTime, nullable=True)
    completed_at    = Column(DateTime, nullable=True)
    failure_reason  = Column(Text, nullable=True)


class Metric(Base):
    __tablename__ = "metrics"

    id          = Column(Integer, primary_key=True, autoincrement=True)
    run_id      = Column(String, nullable=False)
    metric_name = Column(String, nullable=False)
    value       = Column(Float)
    step        = Column(Integer, default=-1)   # -1 = final value
    logged_at   = Column(DateTime, default=func.now())


class AttentionItem(Base):
    __tablename__ = "attention_queue"

    id          = Column(Integer, primary_key=True, autoincrement=True)
    item_type   = Column(String)   # new_best_model | config_error | job_failure
    run_id      = Column(String, nullable=True)
    title       = Column(String)
    description = Column(Text)
    resolved    = Column(Boolean, default=False)
    created_at  = Column(DateTime, default=func.now())


class DigestLog(Base):
    __tablename__ = "digests"

    id             = Column(Integer, primary_key=True, autoincrement=True)
    digest_date    = Column(String)
    content_md     = Column(Text)
    runs_completed = Column(Integer, default=0)
    runs_failed    = Column(Integer, default=0)
    new_best       = Column(Boolean, default=False)
    created_at     = Column(DateTime, default=func.now())


def get_engine():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    return create_engine(f"sqlite:///{DB_PATH}", echo=False)

def init_db():
    """Call once on startup — creates all tables if they don't exist."""
    engine = get_engine()
    Base.metadata.create_all(engine)
    print(f"[DB] Initialized at {DB_PATH}")
    return engine

def get_session():
    engine = get_engine()
    Session = sessionmaker(bind=engine)
    return Session()