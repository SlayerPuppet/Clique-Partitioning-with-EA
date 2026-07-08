"""
Component ablation and hyperparameter study for the memetic clique
partitioning pipeline.

Usage (one command runs everything):
    python ablation_study.py                    # verify + full study (~8 min)
    python ablation_study.py --quick            # tiny EA budget smoke run (~1 min)
    python ablation_study.py --phases a         # only the component study
    python ablation_study.py --phases verify b  # only verification + lambda sweep
    python ablation_study.py --seeds 3          # more EA seeds in phase C
    python ablation_study.py --instances corr40-10 cpn50-4

Phases:
    verify  Evaluate the shipped optimal partitions (CP-Lib Optimal/ folders)
            in our loaded graphs and check they match the literature optima.
            Catches loader/sign-convention bugs before any experiment runs.
    a       Component study: build the pipeline up stage by stage
            (GAEC -> reweighted GAEC -> +KLj -> +EA) and knock single
            components out of the full pipeline (partial optimality,
            reweighting, crossover, LB pruning).
    b       Sweep the reweighting blend lambda at the GAEC+KLj level.
    c       EA hyperparameter grid: crossover rate x max kick.

Results go to CSV files and charts in the working directory.
"""
import argparse
import os
import re
import sys
import csv
import time
import copy
import random

import matplotlib.pyplot as plt

from src.data_loader import load_benchmark_graph
from src.reductions import apply_partial_optimality
from src.bounds import iterative_cycle_packing
from src.heuristics import gaec, reweight_and_gaec, klj_local_search, calculate_original_cost
from src.evolutionary import MemeticAlgorithm

DATA = os.path.join("data", "CP-Lib-main")

# (family, short name, path, literature optimum in min-cost convention)
INSTANCES = [
    ("ABR",         "cars",      os.path.join(DATA, "ABR", "cars.txt"),               -1501.0),
    ("Random",      "cpn35-3",   os.path.join(DATA, "Random", "cpn35-3.txt"),         -7633.0),
    ("Correlation", "corr40-10", os.path.join(DATA, "Correlation", "corr40-10.txt"),  -2301.0),
    ("MCF",         "boc_5",     os.path.join(DATA, "MCF", "boc_5.txt"),              -72.0),
    ("Artificial",  "am-25-10",  os.path.join(DATA, "Artificial", "am-25-10.txt"),    -800.0),
    ("ClusEdit",    "ce50-50",   os.path.join(DATA, "ClusEdit", "ce50-50.txt"),       -163.0),
    ("Equicut",     "neg-c-50",  os.path.join(DATA, "Equicut", "neg-c-50.txt"),       -549.0),
    ("Random",      "cpn50-4",   os.path.join(DATA, "Random", "cpn50-4.txt"),         -13728.0),  # best known
]

# Reduced budget for study runs (single restart) so the full matrix stays tractable.
# --quick shrinks this further for a fast smoke run.
EA_BUDGET = dict(pop_size=20, generations=50)

# Subset used for the EA hyperparameter grid (Phase C)
PHASE_C_INSTANCES = ["corr40-10", "cpn50-4", "ce50-50", "boc_5"]

LAMBDAS = [0.0, 0.25, 0.5, 0.75, 1.0]
CROSSOVER_RATES = [0.0, 0.3, 0.6]
MAX_KICKS = [0.20, 0.40]


# ----------------------------------------------------------------------------
# Pipeline building blocks
# ----------------------------------------------------------------------------

def with_c_paper(G):
    H = copy.deepcopy(G)
    for u, v, d in H.edges(data=True):
        d['c_paper'] = -d['cost']
    return H


def reweighted_seed(G, use_po=True, lam=0.5):
    """PO reductions (optional) + ICP + reweighted GAEC. Returns partition."""
    H = with_c_paper(G)
    reduced = apply_partial_optimality(H) if use_po else H
    _, residuals = iterative_cycle_packing(reduced)
    return reweight_and_gaec(reduced, residuals, lam=lam)


def run_ea(G, seed_partition, rng_seed, **ea_kwargs):
    random.seed(rng_seed)
    kwargs = dict(EA_BUDGET)
    kwargs.update(ea_kwargs)
    ea = MemeticAlgorithm(G, verbose=False, **kwargs)
    partition, cost, _ = ea.optimize(seed_partition=seed_partition)
    return partition, cost


def run_config(config, G, rng_seed=7):
    """Runs one pipeline configuration; returns (cost, seconds)."""
    t0 = time.perf_counter()

    if config == "gaec_raw":
        part = gaec(G)
    elif config == "gaec_rw":
        part = reweighted_seed(G)
    elif config == "klj":
        part, _ = klj_local_search(G, reweighted_seed(G))
    elif config == "full":
        seed, _ = klj_local_search(G, reweighted_seed(G))
        part, _ = run_ea(G, seed, rng_seed)
    elif config == "no_po":
        seed, _ = klj_local_search(G, reweighted_seed(G, use_po=False))
        part, _ = run_ea(G, seed, rng_seed)
    elif config == "no_rw":
        seed, _ = klj_local_search(G, gaec(G))
        part, _ = run_ea(G, seed, rng_seed)
    elif config == "no_xover":
        seed, _ = klj_local_search(G, reweighted_seed(G))
        part, _ = run_ea(G, seed, rng_seed, crossover_rate=0.0)
    elif config == "no_prune":
        seed, _ = klj_local_search(G, reweighted_seed(G))
        part, _ = run_ea(G, seed, rng_seed, use_lb_pruning=False)
    else:
        raise ValueError(config)

    cost = calculate_original_cost(G, part)
    return cost, time.perf_counter() - t0


def gap_pct(cost, optimum):
    return 100.0 * (cost - optimum) / abs(optimum)


# ----------------------------------------------------------------------------
# Chart styling (light surface)
# ----------------------------------------------------------------------------

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
SERIES = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948"]


def styled_axes(ax, title, ylabel):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_title(title, color=INK, fontsize=12, loc="left", pad=12)
    ax.set_ylabel(ylabel, color=INK2, fontsize=10)


def grouped_bars(ax, labels, series, colors):
    """series: list of (name, [values per label])."""
    n_groups, n_series = len(labels), len(series)
    group_w = 0.8
    bar_w = group_w / n_series
    for si, (name, values) in enumerate(series):
        xs = [g - group_w / 2 + bar_w * (si + 0.5) for g in range(n_groups)]
        ax.bar(xs, values, width=bar_w, label=name, color=colors[si],
               edgecolor=SURFACE, linewidth=1.0)
    ax.set_xticks(range(n_groups))
    ax.set_xticklabels(labels, rotation=20, ha="right", color=INK2)
    leg = ax.legend(frameon=False, fontsize=9, loc="upper left", ncols=min(n_series, 3))
    for t in leg.get_texts():
        t.set_color(INK2)


def save_fig(fig, path):
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"   chart -> {path}")


# ----------------------------------------------------------------------------
# Phases
# ----------------------------------------------------------------------------

def load_all(instances):
    graphs = {}
    for family, name, path, opt in instances:
        G = load_benchmark_graph(path)
        graphs[name] = (family, G, opt)
    return graphs


def phase_verify(graphs, instances):
    """Evaluate shipped optimal partitions in our loaded graphs.

    A mismatch means the loader (parsing, sign convention, indexing) is
    broken — abort the study rather than produce meaningless numbers.
    """
    print("\n===== Phase VERIFY: shipped optimal solutions =====")
    failures = 0
    for family, name, path, table_opt in instances:
        opt_path = os.path.join(os.path.dirname(path), "Optimal", f"{name}_opt.txt")
        if not os.path.exists(opt_path):
            print(f"   {name:<10} no Optimal/ file (best-known only) - skipped")
            continue

        _, G, _ = graphs[name]
        text = open(opt_path).read()
        m = re.search(r"Optimal value:\s*(-?\d+)", text)
        published = -float(m.group(1))  # CP-Lib maximization -> our min cost

        # Shipped clusters are 1-indexed; loader nodes are 0-indexed.
        clusters = [set(int(x) - 1 for x in body.split())
                    for body in re.findall(r"\{([\d\s]+)\}", text)]
        covered = set().union(*clusters) if clusters else set()
        for n in G.nodes():
            if n not in covered:
                clusters.append({n})

        cost = calculate_original_cost(G, clusters)
        ok = abs(cost - published) < 1e-6 and abs(published - table_opt) < 1e-6
        status = "PASS" if ok else "FAIL"
        if not ok:
            failures += 1
        print(f"   {name:<10} shipped partition cost={cost:>10.0f}  "
              f"published={published:>10.0f}  table={table_opt:>10.0f}  {status}")

    if failures:
        print(f"\n   {failures} verification failure(s) - loader or optima table is broken. Aborting.")
        sys.exit(1)
    print("   All verifications passed.")


def phase_a(graphs, instances):
    print("\n===== Phase A: component study =====")
    configs = ["gaec_raw", "gaec_rw", "klj", "full", "no_po", "no_rw", "no_xover", "no_prune"]
    rows = []
    for _, name, _, _ in instances:
        family, G, opt = graphs[name]
        for config in configs:
            cost, secs = run_config(config, G)
            rows.append(dict(instance=name, family=family, config=config,
                             cost=cost, optimum=opt,
                             gap_pct=round(gap_pct(cost, opt), 2),
                             time_s=round(secs, 2)))
            print(f"   {name:<10} {config:<9} cost={cost:>10.0f}  "
                  f"gap={rows[-1]['gap_pct']:>6.2f}%  t={secs:5.1f}s")

    with open("ablation_components.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print("   csv -> ablation_components.csv")

    by = {(r["instance"], r["config"]): r["gap_pct"] for r in rows}
    labels = [name for _, name, _, _ in instances]

    # Chart 1: pipeline build-up
    fig, ax = plt.subplots(figsize=(10, 5))
    styled_axes(ax, "Pipeline build-up: gap to literature optimum", "Gap (%)")
    grouped_bars(ax, labels, [
        ("GAEC (raw)",        [by[(l, "gaec_raw")] for l in labels]),
        ("GAEC (reweighted)", [by[(l, "gaec_rw")] for l in labels]),
        ("+ KLj",             [by[(l, "klj")] for l in labels]),
        ("+ Memetic EA",      [by[(l, "full")] for l in labels]),
    ], SERIES)
    save_fig(fig, "ablation_buildup.png")

    # Chart 2: knockouts from the full pipeline
    fig, ax = plt.subplots(figsize=(10, 5))
    styled_axes(ax, "Component knockouts: gap when one part is removed", "Gap (%)")
    grouped_bars(ax, labels, [
        ("Full pipeline",     [by[(l, "full")] for l in labels]),
        ("- partial opt.",    [by[(l, "no_po")] for l in labels]),
        ("- reweighting",     [by[(l, "no_rw")] for l in labels]),
        ("- crossover",       [by[(l, "no_xover")] for l in labels]),
        ("- LB pruning",      [by[(l, "no_prune")] for l in labels]),
    ], SERIES)
    save_fig(fig, "ablation_knockouts.png")
    return rows


def phase_b(graphs, instances):
    print("\n===== Phase B: reweighting lambda sweep (GAEC+KLj level) =====")
    rows = []
    for _, name, _, _ in instances:
        family, G, opt = graphs[name]
        for lam in LAMBDAS:
            t0 = time.perf_counter()
            part, _ = klj_local_search(G, reweighted_seed(G, lam=lam))
            cost = calculate_original_cost(G, part)
            rows.append(dict(instance=name, family=family, lam=lam, cost=cost,
                             gap_pct=round(gap_pct(cost, opt), 2),
                             time_s=round(time.perf_counter() - t0, 2)))
        print(f"   {name:<10} " + "  ".join(
            f"lam={r['lam']:.2f}:{r['gap_pct']:.1f}%" for r in rows[-len(LAMBDAS):]))

    with open("ablation_lambda.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print("   csv -> ablation_lambda.csv")

    fig, ax = plt.subplots(figsize=(8, 5))
    styled_axes(ax, "Reweighting blend λ: gap after GAEC+KLj", "Gap (%)")
    per_inst = {}
    for r in rows:
        per_inst.setdefault(r["instance"], []).append(r["gap_pct"])
    for gaps in per_inst.values():
        ax.plot(LAMBDAS, gaps, color=BASELINE, linewidth=1.0, zorder=1)
    means = [sum(per_inst[i][k] for i in per_inst) / len(per_inst)
             for k in range(len(LAMBDAS))]
    ax.plot(LAMBDAS, means, color=SERIES[0], linewidth=2.0, marker="o",
            markersize=5, zorder=2)
    ax.annotate(f"mean of {len(per_inst)} instances", xy=(LAMBDAS[-1], means[-1]),
                xytext=(-8, 10), textcoords="offset points",
                ha="right", fontsize=9, color=SERIES[0])
    ax.set_xlabel("λ   (1.0 = raw |c| only, 0.0 = ICP residual only)",
                  color=INK2, fontsize=10)
    ax.set_xticks(LAMBDAS)
    save_fig(fig, "ablation_lambda.png")
    return rows


def phase_c(graphs, instances, seeds):
    print("\n===== Phase C: EA hyperparameters (crossover x max kick) =====")
    selected = [n for n in PHASE_C_INSTANCES if n in graphs] or [i[1] for i in instances]
    rows = []
    for name in selected:
        family, G, opt = graphs[name]
        seed, _ = klj_local_search(G, reweighted_seed(G))
        for xr in CROSSOVER_RATES:
            for mk in MAX_KICKS:
                gaps, secs = [], []
                for s in range(1, seeds + 1):
                    t0 = time.perf_counter()
                    _, cost = run_ea(G, seed, rng_seed=s,
                                     crossover_rate=xr, max_kick=mk)
                    secs.append(time.perf_counter() - t0)
                    gaps.append(gap_pct(cost, opt))
                rows.append(dict(instance=name, family=family,
                                 crossover_rate=xr, max_kick=mk,
                                 mean_gap_pct=round(sum(gaps) / len(gaps), 2),
                                 best_gap_pct=round(min(gaps), 2),
                                 mean_time_s=round(sum(secs) / len(secs), 2)))
                print(f"   {name:<10} xover={xr:.1f} kick={mk:.1f} "
                      f"mean gap={rows[-1]['mean_gap_pct']:.2f}%")

    with open("ablation_ea_params.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print("   csv -> ablation_ea_params.csv")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    styled_axes(ax, f"EA hyperparameters: mean gap across {len(selected)} instances",
                "Mean gap (%)")
    labels = [f"{xr:.1f}" for xr in CROSSOVER_RATES]
    series = []
    for mk in MAX_KICKS:
        vals = []
        for xr in CROSSOVER_RATES:
            sel = [r["mean_gap_pct"] for r in rows
                   if r["crossover_rate"] == xr and r["max_kick"] == mk]
            vals.append(sum(sel) / len(sel))
        series.append((f"max kick {int(mk*100)}%", vals))
    grouped_bars(ax, labels, series, SERIES)
    ax.set_xlabel("crossover rate", color=INK2, fontsize=10)
    for tick in ax.get_xticklabels():
        tick.set_rotation(0)
        tick.set_ha("center")
    save_fig(fig, "ablation_ea_params.png")
    return rows


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Ablation & hyperparameter study for the memetic CP pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--phases", nargs="+", choices=["verify", "a", "b", "c"],
                   default=["verify", "a", "b", "c"],
                   help="which phases to run")
    p.add_argument("--quick", action="store_true",
                   help="tiny EA budget (pop 8, gen 10) for a fast smoke run")
    p.add_argument("--seeds", type=int, default=2,
                   help="number of EA seeds per combination in phase C")
    p.add_argument("--instances", nargs="+", metavar="NAME",
                   help="restrict to these instance short names "
                        "(e.g. corr40-10 cpn50-4)")
    return p.parse_args()


def main():
    global EA_BUDGET
    args = parse_args()

    if args.quick:
        EA_BUDGET = dict(pop_size=8, generations=10)

    instances = INSTANCES
    if args.instances:
        wanted = {n.lower() for n in args.instances}
        instances = [i for i in INSTANCES if i[1].lower() in wanted]
        missing = wanted - {i[1].lower() for i in instances}
        if missing:
            print(f"Unknown instance name(s): {', '.join(sorted(missing))}")
            print("Available: " + ", ".join(i[1] for i in INSTANCES))
            sys.exit(1)

    t0 = time.perf_counter()
    print("=" * 60)
    print("   Ablation & Hyperparameter Study")
    print(f"   phases={','.join(args.phases)}  quick={args.quick}  "
          f"seeds={args.seeds}  instances={len(instances)}")
    print("=" * 60)

    graphs = load_all(instances)
    if "verify" in args.phases:
        phase_verify(graphs, instances)
    if "a" in args.phases:
        phase_a(graphs, instances)
    if "b" in args.phases:
        phase_b(graphs, instances)
    if "c" in args.phases:
        phase_c(graphs, instances, args.seeds)

    print(f"\nDone in {time.perf_counter() - t0:.0f}s.")


if __name__ == "__main__":
    main()
