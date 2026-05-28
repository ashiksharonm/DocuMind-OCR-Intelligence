"""
perf.py — Performance Monitoring Middleware
===========================================

ASGI middleware that measures end-to-end request latency for every HTTP
request processed by the DocuMind OCR API.

Features
--------
* Measures wall-clock request duration in milliseconds
* Logs a WARNING when a response exceeds the 2 000 ms SLA target
* Maintains an in-memory circular buffer of the last 1 000 request durations
* Exposes ``get_metrics()`` for the ``/metrics`` endpoint
* Adds ``X-Process-Time-Ms`` response header for client-side visibility

Usage
-----
Register in ``app/main.py``::

    from app.middleware.perf import LatencyMiddleware
    app.add_middleware(LatencyMiddleware)
"""

from __future__ import annotations

import time
import logging
from collections import deque
from typing import Deque, Dict, Any

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger("documind.perf")

# SLA target (food-tech operational reviews require sub-2-second responses)
SLA_THRESHOLD_MS: float = 2_000.0

# Circular buffer for the last N request durations
_HISTORY_SIZE: int = 1_000
_latency_buffer: Deque[float] = deque(maxlen=_HISTORY_SIZE)


def _percentile(data: list, pct: float) -> float:
    """Compute the p-th percentile of a sorted list."""
    if not data:
        return 0.0
    k = max(0, min(len(data) - 1, int(round(pct / 100 * len(data)))))
    return data[k]


def get_metrics() -> Dict[str, Any]:
    """
    Return latency metrics computed from the in-memory buffer.

    Returns
    -------
    dict
        ``total_requests``, ``p50_ms``, ``p95_ms``, ``p99_ms``,
        ``avg_ms``, ``max_ms``, ``sla_violation_pct``.
    """
    samples = sorted(_latency_buffer)
    n = len(samples)
    if n == 0:
        return {
            "total_requests": 0,
            "p50_ms": 0.0,
            "p95_ms": 0.0,
            "p99_ms": 0.0,
            "avg_ms": 0.0,
            "max_ms": 0.0,
            "sla_violation_pct": 0.0,
            "sla_threshold_ms": SLA_THRESHOLD_MS,
        }

    violations = sum(1 for ms in samples if ms > SLA_THRESHOLD_MS)

    return {
        "total_requests": n,
        "p50_ms": round(_percentile(samples, 50), 2),
        "p95_ms": round(_percentile(samples, 95), 2),
        "p99_ms": round(_percentile(samples, 99), 2),
        "avg_ms": round(sum(samples) / n, 2),
        "max_ms": round(max(samples), 2),
        "sla_violation_pct": round(violations / n * 100, 2),
        "sla_threshold_ms": SLA_THRESHOLD_MS,
    }


class LatencyMiddleware(BaseHTTPMiddleware):
    """
    ASGI middleware that tracks per-request latency.

    Adds ``X-Process-Time-Ms`` to every response header and warns when the
    food-tech SLA target of 2 000 ms is breached.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        t_start = time.perf_counter()

        response = await call_next(request)

        elapsed_ms = (time.perf_counter() - t_start) * 1000
        _latency_buffer.append(elapsed_ms)

        # Add header so clients and load-balancers can observe latency
        response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"

        if elapsed_ms > SLA_THRESHOLD_MS:
            logger.warning(
                f"SLA BREACH — {request.method} {request.url.path} "
                f"took {elapsed_ms:.1f} ms (threshold: {SLA_THRESHOLD_MS} ms)"
            )
        else:
            logger.info(
                f"{request.method} {request.url.path} — {elapsed_ms:.1f} ms"
            )

        return response
