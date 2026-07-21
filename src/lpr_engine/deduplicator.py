from datetime import datetime, timedelta

class PlateDeduplicator:
    """Evita emitir múltiplos eventos para a mesma placa em janela de tempo."""

    def __init__(self, cooldown_seconds: int = 30):
        self._seen: dict[str, datetime] = {}
        self._cooldown = timedelta(seconds=cooldown_seconds)

    def should_emit(self, plate: str) -> bool:
        now = datetime.now()
        last = self._seen.get(plate)
        if last is None or (now - last) > self._cooldown:
            self._seen[plate] = now
            return True
        return False