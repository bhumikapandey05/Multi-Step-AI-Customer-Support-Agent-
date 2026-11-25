"""Bounded retries for transient failures inside workflow steps.

Only failures that are safe to repeat are retried: database lock/connection
errors (``OperationalError``) and timeouts. Every write the workflow retries
carries a fixed idempotency key, so a retry can at most *replay* a CRM record,
never create a second one.
"""

import logging
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

from sqlalchemy.exc import OperationalError

logger = logging.getLogger(__name__)

T = TypeVar("T")

TRANSIENT_ERRORS: tuple[type[BaseException], ...] = (OperationalError, TimeoutError)


class TransientFailure(Exception):
    """A transient error persisted after every allowed attempt."""

    def __init__(self, operation: str, attempts: int, cause: BaseException) -> None:
        super().__init__(f"{operation} failed after {attempts} attempts: {type(cause).__name__}")
        self.operation = operation
        self.attempts = attempts
        self.cause = cause


@dataclass(frozen=True)
class RetryPolicy:
    max_retries: int = 3
    base_delay_seconds: float = 0.2
    max_delay_seconds: float = 5.0
    sleep: Callable[[float], None] = time.sleep

    def run(self, operation: str, fn: Callable[[], T]) -> T:
        attempt = 0
        while True:
            try:
                return fn()
            except TRANSIENT_ERRORS as exc:
                if attempt >= self.max_retries:
                    logger.error(
                        "Transient failure persisted",
                        extra={"operation": operation, "attempts": attempt + 1},
                    )
                    raise TransientFailure(operation, attempt + 1, exc) from exc
                delay = min(self.base_delay_seconds * 2**attempt, self.max_delay_seconds)
                delay += random.uniform(0, delay / 2) if delay else 0
                logger.warning(
                    "Retrying after transient failure",
                    extra={
                        "operation": operation,
                        "attempt": attempt + 1,
                        "delay_s": round(delay, 3),
                        "error": type(exc).__name__,
                    },
                )
                self.sleep(delay)
                attempt += 1
