from __future__ import annotations

import os
import secrets

from fastapi import Header, HTTPException


ROLE_ENV_VARS = {
    "internal": "BRAIN_INTERNAL_API_KEY",
    "doctor": "BRAIN_DOCTOR_API_KEY",
    "customer": "BRAIN_CUSTOMER_API_KEY",
}


def authenticate_request(
    x_brain_role: str | None = Header(
        default=None,
        alias="X-Brain-Role",
    ),
    x_brain_key: str | None = Header(
        default=None,
        alias="X-Brain-Key",
    ),
):
    """
    Authenticate an API caller and return its authorized audience.

    The caller cannot choose its audience in the request body.
    Audience is derived from the authenticated role.
    """
    if not x_brain_role or not x_brain_key:
        raise HTTPException(
            status_code=401,
            detail="Missing Brain authentication credentials.",
        )

    role = x_brain_role.strip().lower()

    env_variable = ROLE_ENV_VARS.get(role)

    if env_variable is None:
        raise HTTPException(
            status_code=403,
            detail="This Brain role is not authorized.",
        )

    expected_key = os.getenv(
        env_variable,
        "",
    )

    if not expected_key:
        raise HTTPException(
            status_code=503,
            detail=(
                "Authentication is not configured "
                "for this Brain role."
            ),
        )

    if not secrets.compare_digest(
        x_brain_key,
        expected_key,
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid Brain authentication credentials.",
        )

    return role
