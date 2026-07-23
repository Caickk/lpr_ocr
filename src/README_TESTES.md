# LPR Engine — Testes Unitários

## Estrutura

```
lpr_engine_project/
├── conftest.py          # vazio — fixa o rootdir do pytest
├── pytest.ini            # garante que lpr_engine/ seja importável
├── lpr_engine/
│   ├── __init__.py
│   ├── preprocessor.py    # Bloco A — resize, cinza, CLAHE, threshold
│   ├── ocr.py              # Bloco B — wrapper do PaddleOCR
│   ├── postprocessor.py    # Bloco C — limpeza, correção de chars, regex
│   ├── deduplicator.py      # C4 — deduplicação de leituras por TTL
│   └── engine.py            # orquestrador do pipeline completo
└── tests/
    ├── conftest.py             # stub do paddleocr se não estiver instalado
    ├── test_preprocessor.py
    ├── test_ocr.py
    ├── test_postprocessor.py
    ├── test_deduplicator.py
    └── test_engine.py
```

## Como rodar

```bash
cd lpr_engine_project
pip install -r requirements-dev.txt
pytest -v
```

Se aparecer `ModuleNotFoundError: No module named 'lpr_engine'`, o comando
foi executado de um diretório onde o pytest não conseguiu localizar o
pacote — confirme que está rodando a partir de `lpr_engine_project/`
(onde estão o `conftest.py` e o `pytest.ini` da raiz).

Se o PaddleOCR/PaddlePaddle não estiverem instalados, os testes rodam
mesmo assim — `conftest.py` insere um stub do módulo `paddleocr` para
que `ocr.py` seja importável. Os testes de `test_ocr.py` e `test_engine.py`
usam `unittest.mock` e não dependem do modelo real em nenhum caso.

## O que cada suíte cobre

| Arquivo | Cobertura |
|---|---|
| `test_preprocessor.py` | resize preserva aspect ratio e não reduz imagem já grande; grayscale é idempotente; CLAHE preserva shape/dtype; threshold produz saída binária (0/255); pipeline completo não quebra em shapes extremos |
| `test_postprocessor.py` | limpeza de string (espaços, hífens, case); correção de caracteres por posição (dígito↔letra); validação Mercosul e padrão antigo; rejeição de tamanhos inválidos e de texto irrecuperável |
| `test_ocr.py` | filtragem por score de confiança; seleção do candidato de maior confiança; arredondamento do score; respeito ao `min_confidence` customizado — tudo via mock do PaddleOCR |
| `test_deduplicator.py` | primeira leitura sempre emite; leitura repetida dentro do cooldown é bloqueada; leitura após expiração é emitida de novo; placas diferentes não interferem entre si; limpeza automática de entradas expiradas |
| `test_engine.py` | integração ponta a ponta: OCR vazio → None; texto inválido → None; placa válida → evento completo (`placa`, `padrao`, `confianca_ocr`, `qualidade`, `timestamp`); deduplicação de leituras consecutivas da mesma placa |

## Nota sobre `conftest.py`

O stub de `paddleocr` existe só para permitir importar `lpr_engine.ocr`
em ambientes sem os pesos do modelo baixados (ex.: CI leve). Isso não
substitui testar com o PaddleOCR real — antes de integrar ao pipeline
com o YOLO, vale rodar um teste manual com imagens reais de placa para
validar que a configuração (`latin_PP-OCRv5_mobile_rec`,
`use_doc_unwarping=True` etc.) se comporta como esperado em produção.
