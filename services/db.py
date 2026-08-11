from functools import lru_cache

import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from services.config import get_settings


def _configure(conn: psycopg.Connection) -> None:
    """Teaches psycopg how to adapt Python lists to/from pgvector's `vector`
    type — without this, `embedding <=> %(query_vec)s` fails with
    UndefinedFunction because a plain list is sent as a double precision[]."""
    register_vector(conn)


@lru_cache
def get_pool() -> ConnectionPool:
    settings = get_settings()
    return ConnectionPool(
        settings.database_url,
        min_size=1,
        max_size=10,
        kwargs={"row_factory": dict_row},
        configure=_configure,
        open=True,
    )


def get_connection() -> psycopg.Connection:
    return get_pool().connection()
