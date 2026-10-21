from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "crawl_wikipedia_blocked_users.py"
SPEC = importlib.util.spec_from_file_location("crawl_wikipedia_blocked_users", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_entity_hash_is_normalized_and_deterministic():
    assert MODULE.entity_hash("Example_User") == MODULE.entity_hash(" example user ")
    assert len(MODULE.entity_hash("Example User")) == 64


def test_target_name_is_redacted_from_public_text():
    text = "User:Example_User was mentioned by example user"
    redacted = MODULE.redact_target(text, "Example User")
    assert "example user" not in redacted.casefold()
    assert "example_user" not in redacted.casefold()
    assert redacted.count("[REDACTED_ENTITY]") == 2


def test_ip_and_temporary_accounts_are_excluded():
    assert MODULE.is_registered_account("Example User")
    assert not MODULE.is_registered_account("192.0.2.1")
    assert not MODULE.is_registered_account("2001:db8::1")
    assert not MODULE.is_registered_account("~2026-12345")
    assert not MODULE.is_registered_account("192.0.2.0/24")


def test_reason_classification_is_only_coarse_triage():
    assert MODULE.classify_reason("Persistent vandalism") == "vandalism"
    assert MODULE.classify_reason("Promotional spam") == "spam_or_promotion"
    assert MODULE.classify_reason("Needs human reading") == "other_manual_review"
