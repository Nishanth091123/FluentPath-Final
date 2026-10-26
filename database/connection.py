import os
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "").strip()

if DATABASE_URL:
    # Always use the psycopg2 driver for PostgreSQL
    if DATABASE_URL.startswith("postgresql+pg8000://"):
        DATABASE_URL = DATABASE_URL.replace(
            "postgresql+pg8000://", "postgresql+psycopg2://", 1
        )
    elif DATABASE_URL.startswith("postgresql://"):
        DATABASE_URL = DATABASE_URL.replace(
            "postgresql://", "postgresql+psycopg2://", 1
        )
    elif DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace(
            "postgres://", "postgresql+psycopg2://", 1
        )

    parts = urlsplit(DATABASE_URL)
    query_items = parse_qsl(parts.query, keep_blank_values=True)

    sslmode = None
    cleaned_query = []

    for key, value in query_items:
        key_lower = key.lower()

        # sslmode is passed through connect_args below
        if key_lower == "sslmode":
            sslmode = value.lower()
            continue

        # channel_binding is not needed, so remove it
        if key_lower == "channel_binding":
            continue

        cleaned_query.append((key, value))

    DATABASE_URL = urlunsplit(
        (
            parts.scheme,
            parts.netloc,
            parts.path,
            urlencode(cleaned_query),
            parts.fragment,
        )
    )

    connect_args = {}

    if sslmode:
        # psycopg2 understands sslmode natively
        connect_args["sslmode"] = sslmode

    engine = create_engine(
        DATABASE_URL,
        connect_args=connect_args,
        pool_pre_ping=True,
        pool_recycle=300,
    )

else:
    DATABASE_URL = "sqlite:///ai_english_coach.db"

    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},
    )

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)

Base = declarative_base()
