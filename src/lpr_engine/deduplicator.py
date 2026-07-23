"""
C4 — Deduplicação de leituras por janela de tempo (TTL), thread-safe,
com limpeza periódica de entradas expiradas.
"""
import threading
from datetime import datetime, timedelta


class PlateDeduplicator:
    """Evita emitir múltiplos eventos para a mesma placa em janela de tempo."""

    def __init__(self, cooldown_seconds: int = 30, cleanup_interval: int = 300):
        self._seen: dict[str, datetime] = {}
        self._cooldown = timedelta(seconds=cooldown_seconds)
        self._cleanup_interval = cleanup_interval
        self._lock = threading.Lock()
        self._last_cleanup: datetime | None = None

    def should_emit(self, plate: str, now: datetime | None = None) -> bool:
        """
        Retorna True se a placa deve gerar um novo evento.
        O parâmetro `now` existe para permitir testes determinísticos
        sem depender de time.sleep().
        """
        now = now or datetime.now()

        with self._lock:
            self._maybe_cleanup(now)

            last = self._seen.get(plate)
            if last is None or (now - last) > self._cooldown:
                self._seen[plate] = now
                return True
            return False

    def _maybe_cleanup(self, now: datetime) -> None:
        if self._last_cleanup is None:
            self._last_cleanup = now
            return

        if (now - self._last_cleanup).total_seconds() < self._cleanup_interval:
            return

        expired = [
            plate for plate, ts in self._seen.items()
            if (now - ts) > self._cooldown
        ]
        for plate in expired:
            del self._seen[plate]

        self._last_cleanup = now

    def stats(self) -> dict:
        return {
            'placas_ativas': len(self._seen),
            'cooldown_segundos': self._cooldown.total_seconds(),
        }
