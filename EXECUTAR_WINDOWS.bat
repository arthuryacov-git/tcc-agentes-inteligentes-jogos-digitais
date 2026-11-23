@echo off
cd /d "%~dp0"
python verificar_coleta.py resultados/v5_verificacao --gate || goto erro
python diagnostico_60hz.py --gate || goto erro
python experimento.py --verified-config resultados/v5_verificacao/config.json --out resultados/v5_principal || goto erro
python verificar_coleta.py resultados/v5_principal || goto erro
python relatorio_final.py resultados/v5_principal || goto erro
python experimento.py --verified-config resultados/v5_verificacao/config.json --realtime --n 10 --out resultados/v5_tempo_real || goto erro
python verificar_coleta.py resultados/v5_tempo_real || goto erro
python relatorio_final.py resultados/v5_tempo_real || goto erro
python resumir_frames.py resultados/v5_tempo_real || goto erro
echo Resultados nas pastas resultados\v5_principal e resultados\v5_tempo_real.
pause
exit /b 0
:erro
echo Execucao interrompida. Verifique o erro. Se ainda nao fez, execute VALIDAR_WINDOWS.bat.
pause
exit /b 1
