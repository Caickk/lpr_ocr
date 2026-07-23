import pytest

from lpr_engine.postprocessor import clean, fix_chars, validate


class TestClean:
    def test_uppercases_text(self):
        assert clean("abc1d23") == "ABC1D23"

    def test_removes_spaces(self):
        assert clean("ABC 1D23") == "ABC1D23"

    def test_removes_hyphens(self):
        assert clean("ABC-1234") == "ABC1234"

    def test_strips_leading_trailing_whitespace(self):
        assert clean("  ABC1D23  ") == "ABC1D23"


class TestFixChars:
    def test_corrects_digit_in_letter_position(self):
        # posição 0 deveria ser letra: '0' -> 'O'
        assert fix_chars("0BC1D23")[0] == "O"

    def test_corrects_letter_in_digit_position(self):
        # posição 3 deveria ser dígito: 'O' -> '0'
        assert fix_chars("ABCOD23")[3] == "0"

    def test_leaves_ambiguous_position_untouched(self):
        # posição 4 é ambígua (letra no Mercosul, dígito no antigo)
        result = fix_chars("ABC1023")
        assert result[4] == "0"  # não deve ser forçado a virar letra

    def test_returns_unchanged_if_length_not_seven(self):
        assert fix_chars("ABC123") == "ABC123"

    def test_already_valid_plate_is_unchanged(self):
        assert fix_chars("ABC1D23") == "ABC1D23"


class TestValidateMercosul:
    def test_accepts_valid_mercosul_plate(self):
        result = validate("ABC1D23")
        assert result == ("ABC1D23", "mercosul")

    def test_accepts_mercosul_plate_with_noise(self):
        result = validate("abc-1d23")
        assert result == ("ABC1D23", "mercosul")

    def test_corrects_common_ocr_confusion_in_mercosul_plate(self):
        # OCR leu '0' no lugar de 'O' na primeira posição (letra)
        result = validate("0BC1D23")
        assert result == ("OBC1D23", "mercosul")


class TestValidateAntigo:
    def test_accepts_valid_old_pattern_plate(self):
        result = validate("ABC1234")
        assert result == ("ABC1234", "antigo")

    def test_accepts_old_pattern_with_hyphen(self):
        result = validate("ABC-1234")
        assert result == ("ABC1234", "antigo")


class TestValidateRejection:
    @pytest.mark.parametrize("raw", [
        "",
        "ABC12",        # curto demais
        "ABC12345",     # longo demais
        "1234567",      # sem nenhuma letra
        "ABCDEFG",      # sem nenhum dígito
    ])
    def test_rejects_invalid_length_or_shape(self, raw):
        assert validate(raw) is None

    def test_rejects_text_that_matches_neither_pattern_after_fix(self):
        # 7 caracteres, mas a posição 3 (deve ser dígito em ambos os
        # padrões) tem 'Z', que não está no mapa de correção -> nenhum
        # padrão fecha mesmo após fix_chars
        assert validate("ABCZD23") is None
