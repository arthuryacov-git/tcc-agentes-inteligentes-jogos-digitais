"""Recalcula McNemar exato e Holm a partir dos episódios pareados por semente."""

import argparse
import csv
import json
from collections import defaultdict
from itertools import combinations
from math import comb, isclose
from pathlib import Path


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def exact_mcnemar(a_only, b_only):
    discordant = a_only + b_only
    tail = sum(comb(discordant, k) for k in range(min(a_only, b_only) + 1))
    return min(1.0, 2 * tail / (2 ** discordant))


def adjust_holm(p_values):
    adjusted = {}
    previous = 0.0
    for rank, (key, p_value) in enumerate(sorted(p_values.items(), key=lambda row: row[1])):
        previous = max(previous, min(1.0, (len(p_values) - rank) * p_value))
        adjusted[key] = previous
    return adjusted


def verify_table(rows, counts, adjusted, primary=False):
    seen = set()
    for row in rows:
        if primary:
            model = row["model"]
            key = (row["scenario"], f"llm_hibrido:{model}", f"llm_puro:{model}")
            columns = ("hibrido_only", "puro_only", "p_holm")
        else:
            key = (row["scenario"], row["A"], row["B"])
            columns = ("A_only_success", "B_only_success", "p_mcnemar_holm")
        a_only, b_only = counts[key]
        if (
            key in seen
            or (a_only, b_only) != tuple(int(row[column]) for column in columns[:2])
            or not isclose(exact_mcnemar(a_only, b_only), float(row["p_mcnemar"]), rel_tol=1e-10)
            or not isclose(adjusted[key], float(row[columns[2]]), rel_tol=1e-10)
        ):
            raise ValueError(f"CSV divergente: {key}")
        seen.add(key)
    if seen != adjusted.keys():
        raise ValueError("Família de comparações incompleta")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pasta", nargs="?", type=Path, default=Path("resultados/v5_principal"))
    folder = parser.parse_args().pasta
    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    seeds = set(range(config["seed"], config["seed"] + config["n"]))
    groups = defaultdict(dict)
    for row in read_csv(folder / "episodios.csv"):
        group = groups[row["scenario"], row["mode"]]
        seed, success = int(row["seed"]), int(row["success"])
        if seed in group or success not in (0, 1):
            raise ValueError(f"Episódio inválido ou duplicado: {row}")
        group[seed] = success
    scenarios = sorted({scenario for scenario, _ in groups})
    models = [model["model"] for model in config["models"]]
    modes = ["fsm_bt", "planejador"] + [
        f"{mode}:{model}" for model in models for mode in ("llm_puro", "llm_hibrido")
    ]
    if (
        len(scenarios) != 3 or len(set(modes)) != 6 or len(seeds) != 200
        or set(groups) != {(scenario, mode) for scenario in scenarios for mode in modes}
        or any(group.keys() != seeds for group in groups.values())
    ):
        raise ValueError("Desenho principal incompleto")

    counts = {}
    for scenario in scenarios:
        for a, b in combinations(sorted(modes), 2):
            differences = [groups[scenario, a][s] - groups[scenario, b][s] for s in seeds]
            counts[scenario, a, b] = differences.count(1), differences.count(-1)
    p_values = {key: exact_mcnemar(*pair) for key, pair in counts.items()}
    primary_keys = [
        (scenario, f"llm_hibrido:{model}", f"llm_puro:{model}")
        for scenario in scenarios
        for model in models
    ]
    primary = {key: p_values[key] for key in primary_keys}
    full_adjusted = adjust_holm(p_values)
    primary_adjusted = adjust_holm(primary)
    verify_table(read_csv(folder / "comparacoes_exploratorias.csv"), counts, full_adjusted)
    verify_table(read_csv(folder / "comparacoes_primarias.csv"), counts, primary_adjusted, primary=True)

    for scenario in scenarios:
        values = [p for key, p in full_adjusted.items() if key[0] == scenario]
        print(f"{scenario}: {sum(p < 0.05 for p in values)}/{len(values)}")
    print(f"Todos os pares: {sum(p < 0.05 for p in full_adjusted.values())}/{len(full_adjusted)}")
    print(f"Confirmatórios: {sum(p < 0.05 for p in primary_adjusted.values())}/{len(primary_adjusted)}")
    remaining = [p for key, p in full_adjusted.items() if key not in primary]
    print(f"Pares fora dos confirmatórios, com Holm dos 45: {sum(p < 0.05 for p in remaining)}/{len(remaining)}")
    remaining_adjusted = adjust_holm({key: p for key, p in p_values.items() if key not in primary})
    print(f"Análise alternativa, Holm só dos 39: {sum(p < 0.05 for p in remaining_adjusted.values())}/39")


if __name__ == "__main__":
    main()
