from pathlib import Path

from ase import Atoms
from ase.build import bulk
from camp.ase.calculator import CAMPCalculator


def get_SiC() -> Atoms:
    """Create an ASE SiC crystal."""

    return bulk("SiC", "zincblende", a=4.3596)


def test_calculator():
    path = Path(__file__).parents[2] / "scripts/last_epoch.ckpt"
    calc = CAMPCalculator(
        path, device="cpu", override_atomic_numbers=[6, 14], need_stress=True
    )

    atoms = get_SiC()
    atoms.set_calculator(calc)

    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    stress = atoms.get_stress()

    assert isinstance(energy, float)
    assert forces.shape == (2, 3)
    assert stress.shape == (6,)
