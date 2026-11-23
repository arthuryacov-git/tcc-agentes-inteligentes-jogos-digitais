# Agentes inteligentes em jogos digitais

Materiais para reproduzir a análise do TCC. A coleta principal contém 3.600 episódios, com seis configurações de agente, três cenários e 200 sementes pareadas por cenário. A coleta complementar tem 180 episódios em tempo real.

## Materiais

- [experimento.py](experimento.py) contém os controladores e os prompts. A função `system_prompt` gera as instruções de sistema. A requisição ao Ollama acrescenta a observação, as escolhas habilitadas e a intenção anterior.
- [simulador.py](simulador.py) implementa os cenários e as regras do mundo.
- [analisar.py](analisar.py) e [relatorio_final.py](relatorio_final.py) reproduzem as análises. [conferir_contrastes.py](conferir_contrastes.py) recalcula McNemar exato e Holm a partir dos episódios e confere os CSVs originais.
- [Coleta principal](resultados/v5_principal) e [coleta complementar](resultados/v5_tempo_real) incluem os CSVs por episódio e as tabelas de resultados. [dados-completos.rar](resultados/dados-completos.rar) preserva os registros originais, inclusive as requisições com os prompts efetivamente usados, as decisões, as inferências, os frames e as coletas técnicas.
- [LEIA-ME.txt](LEIA-ME.txt) detalha o protocolo e a execução de uma nova coleta.

## Reproduzir a análise

Os comandos abaixo usam os episódios já coletados e dispensam Ollama. Execute a partir da raiz em uma cópia de trabalho, pois o relatório regrava as tabelas e os gráficos.

```powershell
python conferir_contrastes.py
python -m pip install -r requirements-auditoria.txt
python relatorio_final.py resultados/v5_principal
python relatorio_final.py resultados/v5_tempo_real
python conferir_contrastes.py
```

Para verificar os registros brutos, extraia o RAR com a árvore de pastas preservada e execute, a partir da raiz:

```powershell
tar -xf resultados/dados-completos.rar
python verificar_coleta.py resultados/v5_principal
python verificar_coleta.py resultados/v5_tempo_real
```

O autor confirmou as versões diretamente no ambiente original do experimento.

| Componente | Versão original |
|---|---|
| Python | 3.12.6 |
| NumPy | 2.5.4 |
| SciPy | 1.18.1 |
| Matplotlib | 3.11.2 |
| Ollama | 0.40.16 |

O ambiente da auditoria usou Python **3.12.14** e as versões em [requirements-auditoria.txt](requirements-auditoria.txt).

## Contrastes de sucesso

O script exploratório usa McNemar exato e aplica Holm aos 45 pares de agentes da coleta principal, incluindo os seis pares confirmatórios. Ao nível de 5%, **38 de 45** foram significativos. Foram 12 de 15 em pressão, 14 de 15 em obstáculos e 12 de 15 em persona.

A análise confirmatória compara puro e híbrido em cada combinação de modelo e cenário. Ela aplica Holm à família de seis testes e encontra **4 de 6** diferenças significativas. Para Qwen, a diferença foi significativa nos três cenários. Para Llama, foi significativa em obstáculos. Entre os outros 39 pares, 34 foram significativos segundo a correção dos 45. O Wilcoxon da fração do orçamento de turnos é um desfecho secundário, com Holm próprio.
