import re

# Mapa de substituição para chars frequentemente confundidos
# Aplicado por posição: letras esperadas nas pos. 0,1,2 e 4 (Mercosul)
_DIGIT_TO_CHAR = str.maketrans('01589', 'OISBB')  # para posições de letra
_CHAR_TO_DIGIT = str.maketrans('OISBQ', '01589')  # para posições de dígito

PATTERN_MERCOSUL = re.compile(r'^[A-Z]{3}[0-9][A-Z][0-9]{2}$')
PATTERN_ANTIGO   = re.compile(r'^[A-Z]{3}[0-9]{4}$')

def _fix_chars(text: str) -> str:
    """Corrige confusão O/0, I/1, B/8 baseando-se na posição esperada."""
    if len(text) != 7:
        return text
    chars = list(text)
    # posições 0,1,2 devem ser letras
    for i in range(3):
        chars[i] = chars[i].translate(_DIGIT_TO_CHAR)
    # posição 3 deve ser dígito
    chars[3] = chars[3].translate(_CHAR_TO_DIGIT)
    # posição 4: letra (Mercosul) ou dígito (antigo) — tenta Mercosul primeiro
    # posições 5,6 devem ser dígitos
    for i in range(5, 7):
        chars[i] = chars[i].translate(_CHAR_TO_DIGIT)
    return ''.join(chars)

def validate(raw: str) -> tuple[str, str] | None:
    """
    Retorna (placa_normalizada, padrão) ou None se inválida.
    padrão: 'mercosul' | 'antigo'
    """
    # C1 — Limpeza
    text = raw.upper().strip().replace(' ', '').replace('-', '')

    if len(text) != 7:
        return None

    # C2 — Correção de caracteres por posição
    text = _fix_chars(text)

    # C3 — Validação por regex
    if PATTERN_MERCOSUL.match(text):
        return text, 'mercosul'
    if PATTERN_ANTIGO.match(text):
        return text, 'antigo'

    return None