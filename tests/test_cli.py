"""Tests voor CLI-hulpfuncties."""

from datetime import timezone

from sim.cli import _parse_utc_datetime


def test_parse_utc_datetime_converts_aware_offset_to_utc() -> None:
    """Een expliciete offset mag niet stilzwijgend als UTC worden geïnterpreteerd."""
    parsed = _parse_utc_datetime("2025-01-01T01:30:00+01:00")

    assert parsed.isoformat() == "2025-01-01T00:30:00+00:00"
    assert parsed.tzinfo == timezone.utc


def test_parse_utc_datetime_treats_naive_value_as_utc() -> None:
    parsed = _parse_utc_datetime("2025-01-01")

    assert parsed.isoformat() == "2025-01-01T00:00:00+00:00"
    assert parsed.tzinfo == timezone.utc