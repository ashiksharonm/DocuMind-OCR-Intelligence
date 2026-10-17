"""Job dispatch. SQLite polling by default; Redis (LIST) when REDIS_URL is set.

The SQLite store stays the source of truth in both modes: Redis only carries job ids, so a lost
Redis list is repaired by ``Store.recover_stale`` and the worker's periodic queued-job sweep.
"""

from typing import Any, Optional, Protocol

from app.review.store import Store

REDIS_KEY = "documind:jobs"


class Dispatcher(Protocol):
    def publish(self, job_id: str) -> None: ...

    def claim_next(self, timeout: float) -> Optional[dict[str, Any]]: ...


class SqliteDispatcher:
    """No broker: workers claim the oldest queued row."""

    def __init__(self, store: Store) -> None:
        self.store = store

    def publish(self, job_id: str) -> None:  # the queued row is the message
        return None

    def claim_next(self, timeout: float = 0.0) -> Optional[dict[str, Any]]:
        return self.store.claim()


class RedisDispatcher:
    def __init__(self, store: Store, client: Any) -> None:
        self.store, self.r = store, client

    def publish(self, job_id: str) -> None:
        self.r.lpush(REDIS_KEY, job_id)

    def claim_next(self, timeout: float = 1.0) -> Optional[dict[str, Any]]:
        item = self.r.brpop(REDIS_KEY, timeout=max(1, int(timeout)))
        if item:
            jid = item[1].decode() if isinstance(item[1], bytes) else item[1]
            job = self.store.claim(jid)
            if job:
                return job
        return self.store.claim()  # sweep: picks up rows whose message was lost


def make_dispatcher(store: Store, redis_url: Optional[str] = None) -> Dispatcher:
    if not redis_url:
        return SqliteDispatcher(store)
    import redis  # optional dependency

    return RedisDispatcher(store, redis.Redis.from_url(redis_url))
