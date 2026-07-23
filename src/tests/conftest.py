"""
Garante que os testes rodem mesmo em ambientes sem o PaddleOCR instalado
(ex.: CI leve, máquina de desenvolvimento sem os pesos baixados).

Os testes de ocr.py e engine.py usam mocks/injeção de dependência e não
dependem do modelo real — este stub só evita ImportError na importação
do módulo lpr_engine.ocr.
"""
import sys
import types

try:
    import paddleocr  # noqa: F401
except ImportError:
    fake_module = types.ModuleType("paddleocr")

    class _FakePaddleOCR:
        def __init__(self, *args, **kwargs):
            pass

        def ocr(self, *args, **kwargs):
            return None

    fake_module.PaddleOCR = _FakePaddleOCR
    sys.modules["paddleocr"] = fake_module
