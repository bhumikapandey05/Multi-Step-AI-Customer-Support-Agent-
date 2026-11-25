"""PIN hashing (PBKDF2-HMAC-SHA256) using only the standard library."""

import hashlib
import hmac
import re
import secrets
from functools import cache

DEFAULT_PIN_ITERATIONS = 600_000
PIN_PATTERN = re.compile(r"^\d{6}$")
_ALGORITHM = "pbkdf2_sha256"


def hash_pin(pin: str, *, iterations: int = DEFAULT_PIN_ITERATIONS) -> str:
    """Return ``pbkdf2_sha256$<iterations>$<salt hex>$<hash hex>``."""
    if not PIN_PATTERN.fullmatch(pin):
        raise ValueError("PIN must be exactly 6 digits")
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", pin.encode(), salt, iterations)
    return f"{_ALGORITHM}${iterations}${salt.hex()}${digest.hex()}"


def verify_pin(pin: str, encoded: str) -> bool:
    """Constant-time check of ``pin`` against an encoded hash. Malformed hashes fail closed."""
    try:
        algorithm, iterations_str, salt_hex, hash_hex = encoded.split("$")
        if algorithm != _ALGORITHM:
            return False
        iterations = int(iterations_str)
        salt = bytes.fromhex(salt_hex)
    except ValueError:
        return False
    candidate = hashlib.pbkdf2_hmac("sha256", pin.encode(), salt, iterations)
    return hmac.compare_digest(candidate.hex(), hash_hex)


@cache
def dummy_pin_hash() -> str:
    """Hash used for unknown emails so response time does not reveal whether an account exists."""
    return hash_pin("000000")
