"""
Convert isotope masses to cyclotron frequencies.
"""
import numpy as np
from typing import List


def masses_to_kg(mz_array: np.ndarray, avogadro: float) -> np.ndarray:
    """
    Convert m/z values from g/mol to kg/ion.

    Args:
        mz_array: Mass-to-charge ratios [g/mol]
        avogadro: Avogadro's number [mol⁻¹]

    Returns:
        Mass in kg per ion
    """
    return (mz_array * 1e-3) / avogadro


def compute_frequencies(
        masses_kg: np.ndarray,
        magnetic_field: float,
        charge: int,
        electron_charge: float
) -> np.ndarray:
    """
    Cyclotron frequency f = (B * q) / (2π * m_kg)

    Args:
        masses_kg: Ion masses [kg]
        magnetic_field: Magnetic field strength [T]
        charge: Ion charge in elementary charge units
        electron_charge: Electron charge [C]

    Returns:
        Cyclotron frequencies [Hz]
    """
    q = charge * electron_charge
    return magnetic_field * q / (2 * np.pi * masses_kg)


def process_all_compounds(
        masses_list: List[np.ndarray],
        avogadro: float,
        magnetic_field: float,
        charge: int,
        electron_charge: float
) -> List[np.ndarray]:
    """
    Convenience function to convert all compounds from m/z to frequency.

    Returns:
        List of frequency arrays for each compound
    """
    frequencies = []
    for masses in masses_list:
        masses_kg = masses_to_kg(masses, avogadro)
        freqs = compute_frequencies(masses_kg,
                                    magnetic_field,
                                    charge,
                                    electron_charge)
        frequencies.append(freqs)
    return frequencies