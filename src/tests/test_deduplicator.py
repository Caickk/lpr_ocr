from datetime import datetime, timedelta

import pytest

from lpr_engine.deduplicator import PlateDeduplicator


class TestShouldEmit:
    def test_first_reading_of_a_plate_is_emitted(self):
        dedup = PlateDeduplicator(cooldown_seconds=30)
        assert dedup.should_emit("ABC1D23") is True

    def test_repeated_reading_within_cooldown_is_blocked(self):
        dedup = PlateDeduplicator(cooldown_seconds=30)
        t0 = datetime(2026, 1, 1, 12, 0, 0)

        assert dedup.should_emit("ABC1D23", now=t0) is True
        assert dedup.should_emit("ABC1D23", now=t0 + timedelta(seconds=5)) is False

    def test_reading_after_cooldown_expires_is_emitted_again(self):
        dedup = PlateDeduplicator(cooldown_seconds=30)
        t0 = datetime(2026, 1, 1, 12, 0, 0)

        assert dedup.should_emit("ABC1D23", now=t0) is True
        after_cooldown = t0 + timedelta(seconds=31)
        assert dedup.should_emit("ABC1D23", now=after_cooldown) is True

    def test_different_plates_do_not_interfere_with_each_other(self):
        dedup = PlateDeduplicator(cooldown_seconds=30)
        t0 = datetime(2026, 1, 1, 12, 0, 0)

        assert dedup.should_emit("ABC1D23", now=t0) is True
        assert dedup.should_emit("XYZ9K87", now=t0) is True

    def test_reading_exactly_at_cooldown_boundary_is_blocked(self):
        dedup = PlateDeduplicator(cooldown_seconds=30)
        t0 = datetime(2026, 1, 1, 12, 0, 0)

        assert dedup.should_emit("ABC1D23", now=t0) is True
        # exatamente no limite: (now - last) > cooldown é False para ==
        assert dedup.should_emit("ABC1D23", now=t0 + timedelta(seconds=30)) is False


class TestCleanup:
    def test_expired_entries_are_removed_after_cleanup_interval(self):
        dedup = PlateDeduplicator(cooldown_seconds=10, cleanup_interval=60)
        t0 = datetime(2026, 1, 1, 12, 0, 0)

        dedup.should_emit("ABC1D23", now=t0)
        assert dedup.stats()["placas_ativas"] == 1

        # avança além do cooldown E do intervalo de limpeza
        later = t0 + timedelta(seconds=120)
        dedup.should_emit("XYZ9K87", now=later)  # dispara o cleanup internamente

        assert "ABC1D23" not in dedup._seen
        assert dedup.stats()["placas_ativas"] == 1  # só a nova placa restou

    def test_cleanup_does_not_run_before_interval_elapses(self):
        dedup = PlateDeduplicator(cooldown_seconds=5, cleanup_interval=300)
        t0 = datetime(2026, 1, 1, 12, 0, 0)

        dedup.should_emit("ABC1D23", now=t0)
        # placa já expirou (cooldown 5s) mas o intervalo de limpeza (300s) não
        soon_after = t0 + timedelta(seconds=10)
        dedup.should_emit("XYZ9K87", now=soon_after)

        # entrada expirada ainda não foi varrida, mas não bloqueia novas placas
        assert "ABC1D23" in dedup._seen


class TestStats:
    def test_stats_reports_cooldown_in_seconds(self):
        dedup = PlateDeduplicator(cooldown_seconds=45)
        assert dedup.stats()["cooldown_segundos"] == 45
