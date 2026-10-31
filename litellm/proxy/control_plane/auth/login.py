"""Username and password login for the admin configured through UI_USERNAME / UI_PASSWORD."""

import secrets
from collections.abc import Callable
from dataclasses import dataclass
from math import ceil
from typing import Final, TypeAlias

from litellm.proxy.control_plane.auth.jwt import SessionTokenCodec
from litellm.proxy.control_plane.auth.rbac import ADMIN_ROLE, RolePolicy, build_principal
from litellm.types.proxy.control_plane.auth import SessionTokenResponse

LOCKOUT_FAILURES: Final = 5
LOCKOUT_SECONDS: Final = 300.0


@dataclass(frozen=True, slots=True)
class InvalidCredentials:
    pass


@dataclass(frozen=True, slots=True)
class Throttled:
    retry_after_seconds: int


LoginResult: TypeAlias = SessionTokenResponse | InvalidCredentials | Throttled


class LoginThrottle:
    """Locks an account after ``failures`` misses inside ``lockout_seconds``. Source address limits belong to the load
    balancer in front of the service, which sees the real peer. Expired misses for a key are dropped when that key is
    next recorded."""

    def __init__(
        self, now: Callable[[], float], failures: int = LOCKOUT_FAILURES, lockout_seconds: float = LOCKOUT_SECONDS
    ) -> None:
        self._now = now
        self._failures = failures
        self._lockout_seconds = lockout_seconds
        self._misses: dict[str, tuple[float, ...]] = {}  # mutable-ok: per-account failure timestamps

    def _recent(self, key: str) -> tuple[float, ...]:
        cutoff: Final = self._now() - self._lockout_seconds
        return tuple(at for at in self._misses.get(key, ()) if at > cutoff)

    def locked_for(self, key: str) -> float:
        recent: Final = self._recent(key)
        if len(recent) < self._failures:
            return 0.0
        return max(0.0, recent[-1] + self._lockout_seconds - self._now())

    def record_failure(self, key: str) -> None:
        recent: Final = self._recent(key)
        self._misses[key] = (*recent, self._now())

    def tracked(self) -> int:
        return len(self._misses)

    def reset(self, key: str) -> None:
        self._misses.pop(key, None)


class LoginService:
    def __init__(
        self,
        sessions: SessionTokenCodec,
        policy: RolePolicy,
        ttl_seconds: int,
        admin_username: str | None,
        admin_password: str | None,
        throttle: LoginThrottle,
    ) -> None:
        self._sessions = sessions
        self._policy = policy
        self._ttl_seconds = ttl_seconds
        self._admin_username = admin_username
        self._admin_password = admin_password
        self._throttle = throttle

    def login(self, username: str, password: str) -> LoginResult:
        if self._admin_username is None or self._admin_password is None:
            return InvalidCredentials()
        if not secrets.compare_digest(username.encode("utf-8"), self._admin_username.encode("utf-8")):
            return InvalidCredentials()
        locked: Final = self._throttle.locked_for(self._admin_username)
        if locked > 0:
            return Throttled(retry_after_seconds=max(1, ceil(locked)))
        if not self._credentials_match(username, password):
            self._throttle.record_failure(self._admin_username)
            return InvalidCredentials()
        self._throttle.reset(self._admin_username)
        principal: Final = build_principal(
            policy=self._policy,
            subject=username,
            email=None,
            roles=frozenset({ADMIN_ROLE}),
            claimed_scopes=None,
            dataplane_ids=frozenset(),
            token_kind="session",
        )
        return SessionTokenResponse(access_token=self._sessions.mint(principal), expires_in=self._ttl_seconds)

    def _credentials_match(self, username: str, password: str) -> bool:
        if self._admin_username is None or self._admin_password is None:
            return False
        username_ok: Final = secrets.compare_digest(username.encode("utf-8"), self._admin_username.encode("utf-8"))
        password_ok: Final = secrets.compare_digest(password.encode("utf-8"), self._admin_password.encode("utf-8"))
        return username_ok and password_ok
