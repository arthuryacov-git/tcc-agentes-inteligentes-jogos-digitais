@echo off
cd /d "%~dp0"
python -m pip install -r requirements.txt || goto erro
ollama list || goto erro
ollama pull qwen2.5:3b || goto erro
ollama pull llama3.2:3b || goto erro
python testar_infra.py || goto erro
python experimento.py --n 2 --seed 20261001 --out resultados/v5_verificacao || goto erro
python verificar_coleta.py resultados/v5_verificacao --gate || goto erro
python diagnostico_60hz.py || goto erro
echo Verificacao tecnica concluida. Envie a pasta resultados inteira para auditoria.
pause
exit /b 0
:erro
echo Verificacao interrompida. Consulte o erro. Registros concluidos sao preservados.
pause
exit /b 1
