"""
Statistical significance tests for TopoAdamW vs AdamW baselines (CIFAR-10, 5 seeds).

Comparisons:
  - TopoAdamW vs AdamW (plain)
  - TopoAdamW vs AdamW+Cosine (cosine-annealed schedule)

Outputs:
  - Paired t-tests (two-sided)
  - Cohen's d effect size
  - 95% CI on mean difference
  - LaTeX snippets for paper
"""

import json
import numpy as np
from scipy import stats

CHECKPOINT = "multiseed_results.json"
SEEDS = list(range(5))
DATASET = "cifar10"


def load_final_accs(ckpt, opt_label):
    accs = []
    for s in SEEDS:
        key = f"{DATASET}/{opt_label}/seed={s}"
        accs.append(ckpt[key]["val_acc"][-1])
    return np.array(accs)


def cohens_d(a, b):
    """Paired Cohen's d = mean(diff) / std(diff)."""
    diff = a - b
    return diff.mean() / (diff.std(ddof=1) + 1e-12)


def compare(label_a, a, label_b, b):
    diff = a - b
    t_stat, p_two = stats.ttest_rel(a, b)
    d = cohens_d(a, b)
    ci = stats.t.interval(0.95, df=len(diff)-1,
                          loc=diff.mean(), scale=stats.sem(diff))

    sig = ('***' if p_two < 0.001 else '**' if p_two < 0.01
           else '*' if p_two < 0.05 else 'n.s.')
    print(f"\n--- {label_a} vs {label_b} (paired two-sided t-test) ---")
    print(f"  Mean diff  = {diff.mean()*100:+.3f} pp  ({label_a} − {label_b})")
    print(f"  t = {t_stat:.4f}")
    print(f"  p (two-sided) = {p_two:.4f}  {sig}")
    print(f"  Cohen's d = {d:.4f}  ({'large' if abs(d)>0.8 else 'medium' if abs(d)>0.5 else 'small'})")
    print(f"  95% CI on mean diff: [{ci[0]*100:.3f}, {ci[1]*100:.3f}] pp")

    sig_str = ("$p < 0.001$" if p_two < 0.001 else
               "$p < 0.01$"  if p_two < 0.01  else
               f"$p = {p_two:.3f}$")
    latex = (
        f"(paired $t$-test, $t={t_stat:.2f}$, {sig_str}; "
        f"Cohen's $d={d:.2f}$; "
        f"95\\% CI: $[{ci[0]*100:.2f},\\,{ci[1]*100:.2f}]$ pp)"
    )
    print(f"  LaTeX: {latex}")
    return t_stat, p_two, d, ci


def main():
    with open(CHECKPOINT) as f:
        ckpt = json.load(f)

    adamw   = load_final_accs(ckpt, "AdamW")
    cosine  = load_final_accs(ckpt, "AdamW+Cosine")
    topo    = load_final_accs(ckpt, "TopoAdamW")

    print("=" * 60)
    print(f"CIFAR-10 final val_acc over {len(SEEDS)} seeds")
    print("=" * 60)
    for label, arr in [("AdamW", adamw), ("AdamW+Cosine", cosine), ("TopoAdamW", topo)]:
        print(f"  {label:15s}: {arr}  mean={arr.mean():.4f}  std={arr.std(ddof=1):.4f}")

    compare("TopoAdamW", topo, "AdamW",        adamw)
    compare("TopoAdamW", topo, "AdamW+Cosine", cosine)


if __name__ == "__main__":
    main()
