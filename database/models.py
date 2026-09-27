from datetime import datetime, timezone

from sqlalchemy import Column, Integer, String, DateTime

from database.connection import Base


def utc_now_naive():
    """Naive UTC timestamp for compatibility with the existing DateTime schema."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(20), unique=True, nullable=False, index=True)
    name = Column(String(100), nullable=False)
    created_at = Column(DateTime, default=utc_now_naive)


class AdminUser(Base):
    __tablename__ = "admin_users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(100), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=utc_now_naive)


class ActivityLog(Base):
    __tablename__ = "activity_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(20), nullable=False, index=True)
    activity_type = Column(String(50), nullable=False, index=True)
    activity = Column(String(255), nullable=False)
    score = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=utc_now_naive)
