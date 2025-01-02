"""
Compute MAE and MSE metrics using trained models via the ASE calculator.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from ase import Atoms
from camp.ase.calculator import CAMPCalculator


def load_data(path: Path) -> tuple[list[Atoms], list[np.ndarray], list[np.ndarray]]:
    """
    Load data from a file.

    Args:
        path: path to the json file containing the data.

    Returns:
        config: list of ASE atoms objects
        energy: total target energies of the configurations
        forces: forces on the atoms
    """

    def create_atoms(row):
        return Atoms(
            cell=row["cell"],
            positions=row["coords"],
            numbers=row["atomic_number"],
            pbc=row["pbc"],
        )

    df = pd.read_json(path)
    df["config"] = df.apply(create_atoms, axis=1)

    return df["config"].tolist(), df["energy"].tolist(), df["forces"].tolist()


def get_energy_and_forces(calculator, atoms):
    atoms.set_calculator(calculator)
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    return energy, forces


def compute_metrics(
    config: list[Atoms],
    energy: list[np.ndarray],
    forces: list[np.ndarray],
    calculator,
):
    """
    Compute MAE and MSE metrics.

    Args:
        config: ASE atoms objects
        energy: total target energies of the configurations
        forces: forces on the atoms
    """
    all_e = []
    all_f = []
    n_atoms = []
    for conf in config:
        e, f = get_energy_and_forces(calculator, conf)
        all_e.append(e)
        all_f.append(f)
        n_atoms.append(len(conf))

    mae_e = np.mean(np.abs(np.array(all_e) / n_atoms - np.array(energy) / n_atoms))
    mse_e = np.mean((np.array(all_e) / n_atoms - np.array(energy) / n_atoms) ** 2)

    mae_f = np.mean(np.abs(np.array(all_f) - np.array(forces)))
    mse_f = np.mean((np.array(all_f) - np.array(forces)) ** 2)

    print(f"Energy MAE: {mae_e} eV/atom")
    print(f"Energy MSE: {mse_e} eV^2/atom^2")
    print(f"Forces MAE: {mae_f} eV/Angstrom")
    print(f"Forces MSE: {mse_f} eV^2/Angstrom^2")


if __name__ == "__main__":
    data_path = "/Users/mjwen.admin/Packages/camp_analysis/dataset/water/json_data/val_water.json"
    config, energy, forces = load_data(data_path)

    model_path = "/Users/mjwen.admin/Packages/camp_analysis/tests/water_stability/2-water_ase_md/240510_water/epoch=4649-step=1664700.ckpt"
    calc = CAMPCalculator(model_path, use_ema_params=False)

    compute_metrics(config, energy, forces, calc)
