"""PIN hashing and signed session contexts."""

import dataclasses
from datetime import UTC, datetime, timedelta

import pytest

from support_agent.crm.errors import AuthenticationError
from support_agent.crm.identity import SessionSigner
from support_agent.crm.security import hash_pin, verify_pin

SECRET = "unit-test-secret-key-with-enough-length"


class Clock:
    def __init__(self) -> None:
        self.now = datetime(2026, 6, 15, 12, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def signer(clock) -> SessionSigner:
    return SessionSigner(SECRET, ttl=timedelta(minutes=30), clock=clock)


def test_pin_hash_round_trip_and_salting():
    first, second = hash_pin("123456", iterations=1_000), hash_pin("123456", iterations=1_000)
    assert first != second  # random salt
    assert verify_pin("123456", first)
    assert not verify_pin("654321", first)
    assert "123456" not in first


@pytest.mark.parametrize("encoded", ["", "garbage", "md5$1$00$00", "pbkdf2_sha256$x$zz$00"])
def test_malformed_hash_fails_closed(encoded):
    assert verify_pin("123456", encoded) is False


@pytest.mark.parametrize("pin", ["12345", "abcdef", "1234567"])
def test_hash_pin_rejects_invalid_pins(pin):
    with pytest.raises(ValueError):
        hash_pin(pin, iterations=1_000)


def test_issued_context_verifies(signer):
    ctx = signer.issue(42)
    assert signer.verify(ctx) == 42
    assert ctx.expires_at - ctx.issued_at == timedelta(minutes=30)


@pytest.mark.parametrize(
    "change",
    [
        {"customer_id": 43},
        {"session_id": "other"},
        {"signature": "forged"},
        {"expires_at": datetime(2030, 1, 1, tzinfo=UTC)},
    ],
)
def test_any_tampering_invalidates_context(signer, change):
    forged = dataclasses.replace(signer.issue(42), **change)
    with pytest.raises(AuthenticationError):
        signer.verify(forged)


def test_context_from_other_secret_is_rejected(signer, clock):
    other = SessionSigner(
        "a-completely-different-secret-key-value", ttl=timedelta(minutes=30), clock=clock
    )
    with pytest.raises(AuthenticationError):
        signer.verify(other.issue(42))


def test_expired_context_is_rejected(signer, clock):
    ctx = signer.issue(42)
    clock.now += timedelta(minutes=30)
    with pytest.raises(AuthenticationError, match="expired"):
        signer.verify(ctx)


@pytest.mark.parametrize("not_a_context", [None, 1, {"customer_id": 1}, "token"])
def test_non_context_objects_are_rejected(signer, not_a_context):
    with pytest.raises(AuthenticationError):
        signer.verify(not_a_context)


def test_token_round_trip(signer):
    ctx = signer.issue(7)
    assert signer.decode_token(signer.encode_token(ctx)) == ctx


@pytest.mark.parametrize("token", ["", "v1.abc", "v2.e30.sig", "v1.!!!.sig", "v1.e30.sig"])
def test_malformed_tokens_are_rejected(signer, token):
    with pytest.raises(AuthenticationError):
        signer.decode_token(token)


def test_short_secret_is_refused():
    with pytest.raises(ValueError):
        SessionSigner("too-short", ttl=timedelta(minutes=5))
