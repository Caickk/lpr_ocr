# Este arquivo existe propositalmente vazio.
#
# A presença de um conftest.py na raiz do projeto sinaliza ao pytest
# que este diretório é o rootdir — isso garante que `lpr_engine/`
# (pacote irmão de `tests/`) seja adicionado ao sys.path automaticamente,
# independentemente de onde o comando `pytest` for executado.
#
# Sem este arquivo, rodar `pytest` de dentro de tests/ (ou em alguns
# terminais Windows) pode resultar em:
#   ModuleNotFoundError: No module named 'lpr_engine'
