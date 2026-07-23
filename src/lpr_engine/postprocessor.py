"""
Bloco C — Pós-processamento: limpeza, correção de caracteres e validação
da leitura bruta retornada pelo OCR contra os padrões de placa brasileiros.
"""
import re

PATTERN_MERCOSUL = re.compile(r'^[A-Z]{3}[0-9][A-Z][0-9]{2}$')
PATTERN_ANTIGO = re.compile(r'^[A-Z]{3}[0-9]{4}$')

# Correção por posição: dígito <-> letra, para os caracteres mais
# frequentemente confundidos pelo OCR.
_DIGIT_TO_CHAR = str.maketrans('01589', 'OISBB')
_CHAR_TO_DIGIT = str.maketrans('OISBQ', '01589')

_LETTER_POSITIONS = (0, 1, 2)   # sempre letras nos dois padrões
_DIGIT_POSITIONS = (3, 5, 6)    # sempre dígitos nos dois padrões
# posição 4 é ambígua (letra no Mercosul, dígito no padrão antigo) —
# não forçamos correção nela, deixamos a regex decidir.


def clean(raw: str) -> str:
    """C1 — Remove espaços, hífens e normaliza para maiúsculas."""
    return raw.upper().strip().replace(' ', '').replace('-', '')


def fix_chars(text: str) -> str:
    """C2 — Corrige confusão O/0, I/1, B/8, S/5 com base na posição esperada."""
    if len(text) != 7:
        return text

    chars = list(text)
    for i in _LETTER_POSITIONS:
        chars[i] = chars[i].translate(_DIGIT_TO_CHAR)
    for i in _DIGIT_POSITIONS:
        chars[i] = chars[i].translate(_CHAR_TO_DIGIT)
    return ''.join(chars)


def validate(raw: str) -> tuple[str, str] | None:
    """
    C3 — Pipeline completo de validação.
    Retorna (placa_normalizada, padrao) ou None se inválida.
    padrao: 'mercosul' | 'antigo'
    """
    text = clean(raw)

    if len(text) != 7:
        return None

    text = fix_chars(text)

    if PATTERN_MERCOSUL.match(text):
        return text, 'mercosul'
    if PATTERN_ANTIGO.match(text):
        return text, 'antigo'

    return None
