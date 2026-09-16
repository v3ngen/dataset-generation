"""Cost matrices for the two stakeholders in the Coffee Health v3 coursework.

Both stakeholders want to identify people who will have a year of high health needs
(``HighHealthNeeds == 1``) so they can invite them onto a preventive programme.
They pay very different prices for getting it wrong, so they will not necessarily
agree about which model is best.

See STAKEHOLDERS_AND_COSTS.md for the scenario the numbers come from.

Typical use::

    from costs import COST_MATRICES, total_cost, cost_per_person

    for name, matrix in COST_MATRICES.items():
        print(name, cost_per_person(y_true, y_pred, matrix))

All costs are per person scored, in euros. Negative values are net *benefits*.
"""

from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------------------
# The matrices
# --------------------------------------------------------------------------------------

#: Stakeholder A. Publicly funded, judged against downstream treatment costs.
#: Missing someone is roughly eight times as expensive as an unnecessary invitation,
#: so this stakeholder wants high recall.
COST_PUBLIC_HEALTH = {
    "TN": 0,       # correctly not invited: no cost
    "FP": 180,     # invited unnecessarily: wasted health check and programme place
    "FN": 1450,    # missed: avoidable primary and secondary care
    "TP": -720,    # correctly invited: 180 programme cost less ~900 of avoided care
}

#: Stakeholder B. Fixed annual block of programme places, signed off by finance.
#: A wasted place is nearly as expensive as a miss, so this stakeholder wants high
#: precision.
COST_EMPLOYER = {
    "TN": 0,       # correctly not invited: no cost
    "FP": 520,     # invited unnecessarily: a scarce place consumed for no return
    "FN": 780,     # missed: absence cover, temporary staffing, lost productivity
    "TP": -80,     # correctly invited: 520 programme cost less ~600 of avoided absence
}

#: Both matrices, keyed by a readable stakeholder name.
COST_MATRICES = {
    "Public health agency": COST_PUBLIC_HEALTH,
    "Employer occupational health": COST_EMPLOYER,
}


# --------------------------------------------------------------------------------------
# Applying them
# --------------------------------------------------------------------------------------

def confusion_counts(y_true, y_pred) -> dict:
    """Return ``{'TN':.., 'FP':.., 'FN':.., 'TP':..}`` for binary 0/1 labels.

    The positive class (1) means "will have a year of high health needs".
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")
    return {
        "TN": int(np.sum((y_true == 0) & (y_pred == 0))),
        "FP": int(np.sum((y_true == 0) & (y_pred == 1))),
        "FN": int(np.sum((y_true == 1) & (y_pred == 0))),
        "TP": int(np.sum((y_true == 1) & (y_pred == 1))),
    }


def total_cost(y_true, y_pred, matrix: dict) -> float:
    """Total cost of a set of predictions under one stakeholder's cost matrix."""
    counts = confusion_counts(y_true, y_pred)
    return float(sum(counts[cell] * matrix[cell] for cell in ("TN", "FP", "FN", "TP")))


def cost_per_person(y_true, y_pred, matrix: dict) -> float:
    """Total cost divided by the number of people scored.

    Prefer this to :func:`total_cost` when comparing across datasets of different
    sizes -- for example the development set against the smaller held-out test set.
    """
    n = len(np.asarray(y_true))
    if n == 0:
        raise ValueError("no rows to score")
    return total_cost(y_true, y_pred, matrix) / n


def cost_summary(y_true, y_pred) -> dict:
    """Per-person cost under every stakeholder, keyed by stakeholder name."""
    return {
        name: cost_per_person(y_true, y_pred, matrix)
        for name, matrix in COST_MATRICES.items()
    }


if __name__ == "__main__":
    # A worked example: 10 people, of whom 5 actually had a year of high health
    # needs, and a model that correctly identified 2 of them while wrongly
    # inviting nobody.
    example_actual    = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    example_predicted = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]

    print(f"counts: {confusion_counts(example_actual, example_predicted)}\n")
    for stakeholder, cost_matrix in COST_MATRICES.items():
        print(f"{stakeholder:<32} "
              f"total EUR {total_cost(example_actual, example_predicted, cost_matrix):>9,.0f}   "
              f"per person EUR {cost_per_person(example_actual, example_predicted, cost_matrix):>8,.2f}")
