# LPR Engine — Reconhecimento de Placas (LAIAI)

Módulo responsável por identificar o conteúdo de placas veiculares a
partir de um recorte de imagem já localizado por um detector (YOLO).
Pipeline: pré-processamento → OCR (PaddleOCR) → validação/correção →
deduplicação por TTL.

## Estrutura do projeto

```
lpr_engine_project/
├── .gitignore
├── README.md
├── requirements-dev.txt
├── pytest.ini
├── conftest.py              # vazio — fixa o rootdir do pytest
├── lpr_engine/
│   ├── __init__.py
│   ├── preprocessor.py      # Bloco A — resize, cinza, CLAHE, threshold
│   ├── ocr.py                # Bloco B — wrapper do PaddleOCR
│   ├── postprocessor.py      # Bloco C — limpeza, correção de chars, regex
│   ├── deduplicator.py        # C4 — deduplicação de leituras por TTL
│   └── engine.py              # orquestrador do pipeline completo
└── tests/
    ├── conftest.py               # stub do paddleocr se não estiver instalado
    ├── test_preprocessor.py
    ├── test_ocr.py
    ├── test_postprocessor.py
    ├── test_deduplicator.py
    └── test_engine.py
```

---

## Instalação

### Pré-requisitos

- Python 3.10+
- pip

### 1. Criar e ativar um ambiente virtual (recomendado)

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate
```

### 2. Instalar as dependências

```bash
cd lpr_engine_project
pip install -r requirements-dev.txt
```

Isso instala `pytest`, `numpy`, `opencv-python-headless`, `paddleocr` e
`paddlepaddle`. Na **primeira execução real** do OCR (não dos testes,
que usam mock), o PaddleOCR baixa automaticamente os pesos dos modelos
(`PP-OCRv5_mobile_det`, `latin_PP-OCRv5_mobile_rec`) para
`~/.paddleocr/` — é necessário ter internet nessa primeira vez, depois
tudo roda offline.

Se a máquina tiver GPU NVIDIA com CUDA configurado, troque
`paddlepaddle` por `paddlepaddle-gpu` e ajuste `device="gpu"` em
`lpr_engine/ocr.py`.

---

## Como rodar os testes

```bash
cd lpr_engine_project
pytest -v
```

Os testes **não** dependem de baixar os modelos do PaddleOCR — `ocr.py`
e `engine.py` são testados via `unittest.mock`, e `tests/conftest.py`
insere um stub do módulo `paddleocr` caso ele não esteja instalado.
Isso permite rodar a suíte inteira em segundos, mesmo sem GPU ou
conexão de internet.

### Rodando uma suíte específica

```bash
pytest tests/test_postprocessor.py -v
```

### Rodando com relatório de cobertura (opcional)

```bash
pip install pytest-cov
pytest --cov=lpr_engine --cov-report=term-missing
```

### Problema comum: `ModuleNotFoundError: No module named 'lpr_engine'`

Confirme que está rodando o `pytest` a partir da raiz do projeto
(`lpr_engine_project/`, onde estão o `conftest.py` e o `pytest.ini`
deste nível — não o de dentro de `tests/`). Esses dois arquivos
garantem que o pacote `lpr_engine/` seja encontrado independentemente
de como o comando é chamado.

---

## Uso básico da engine (fora dos testes)

```python
from lpr_engine.engine import LPREngine
import cv2

engine = LPREngine()  # carrega o PaddleOCR real na primeira instância

crop = cv2.imread("placa_recortada.jpg")  # recorte vindo do YOLO
evento = engine.process(crop)

if evento:
    print(evento)
    # {'placa': 'ABC1D23', 'confianca_ocr': 0.91, 'qualidade': 'alta',
    #  'padrao': 'mercosul', 'timestamp': '...'}
else:
    print("Leitura descartada (OCR abaixo do limiar, regex inválida ou duplicata).")
```

---

## Notas de configuração

- `enable_mkldnn` em `ocr.py` é controlado pela variável de ambiente
  `LPR_ENV`: fica desativado por padrão (`dev`) e só liga em
  `LPR_ENV=production`, já que o ganho de performance do MKL-DNN só se
  realiza em CPU física, não em ambientes virtualizados como Colab.
- `MIN_SCORE_HARD`/`MIN_SCORE_SOFT`/`MIN_SCORE_CLEAN` (em `engine.py`)
  e `cooldown_seconds` (deduplicador) são parâmetros de calibração —
  os valores atuais são um ponto de partida razoável, mas devem ser
  ajustados com dados reais de placa do ambiente da UEFS antes da
  entrega final.
