import os
from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool

from alembic import context

# Wire in both SQLAlchemy Base objects so autogenerate sees all tables
from backend.database.models import Base as AppBase
from backend.memory.models import MemoryBase

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Combine metadata from both Base classes for autogenerate support
target_metadata = [AppBase.metadata, MemoryBase.metadata]

# Override sqlalchemy.url from environment so DATABASE_URL is the single source of truth
db_url = os.environ.get("DATABASE_URL", "sqlite:///./ai_cfo.db")
config.set_main_option("sqlalchemy.url", db_url)


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=True,  # required for SQLite ALTER TABLE support
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    from sqlalchemy import text as _text

    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        is_pg = "postgresql" in str(connectable.url)
        if is_pg:
            connection.execute(_text("SELECT pg_advisory_lock(12345678)"))
        try:
            context.configure(
                connection=connection,
                target_metadata=target_metadata,
                render_as_batch=True,
            )
            with context.begin_transaction():
                context.run_migrations()
        finally:
            if is_pg:
                connection.execute(_text("SELECT pg_advisory_unlock(12345678)"))


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
