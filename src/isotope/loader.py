"""
Load chemical formulas and compute isotopic fine structures using IsoSpec.
"""

import numpy as np
import IsoSpecPy as iso


def load_compounds(
        filename: str,
        coverage: float = 0.99
):
    """
    Read chemical formulas from a text file (one per line) and compute:
        - isotope masses (m/z)    [g/mol]
        - absolute probabilities
        - relative abundances (probability / max(probability))

    Args:
        filename: Path to text file with one formula per line
        coverage: Probability coverage for IsoSpec (default: 0.99)

    Returns:
        formulas: List of formula strings
        masses: List of 1D float32 arrays (m/z of each isotope)
        probs: List of 1D float32 arrays (probabilities)
        rel_abundances: List of 1D float32 arrays (normalised to [0,1])
    """
    with open(filename, "r") as f:
        formulas = [line.strip() for line in f.readlines()]

    masses = []
    probs = []
    rel_abund = []

    for form in formulas:
        spec = iso.IsoTotalProb(formula=form, prob_to_cover=coverage)
        mz = np.array([j for j, _ in spec], dtype=np.float32)
        prob = np.array([k for _, k in spec], dtype=np.float32)
        masses.append(mz)
        probs.append(prob)
        rel_abund.append(prob / prob.max())

    print(f"Loaded {len(formulas)} compounds from '{filename}'")

    # Show random sample
    n_show = min(50, len(formulas))
    idx = np.sort(np.random.randint(0, len(formulas), size=n_show))
    print("Sample compounds (index, formula):")
    for i in idx:
        print(f"{i:4d}  {formulas[i]}")

    return formulas, masses, probs, rel_abund
