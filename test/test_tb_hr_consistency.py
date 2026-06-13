from pathlib import Path
import unittest

import numpy as np

from ssg4wann.parsergen.hr_parser import hr
from ssg4wann.parsergen.tb_parser import tb


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TB_PATH = REPOSITORY_ROOT / "examples/Fe_tb/wannier90.up_symmed_tb.dat"
HR_PATH = REPOSITORY_ROOT / "examples/Fe/wannier90.up_symmed_hr.dat"
MAX_HAMILTONIAN_DIFFERENCE_NORM = 1.0e-6


def tb_hamiltonians(path: Path) -> tuple[dict[tuple[int, int, int], np.ndarray], int]:
    parsed = tb.raw_read(str(path))
    num_wann = parsed["num_wann"]
    matrices = {}
    for r_vector, block in parsed["H"].items():
        matrix = np.zeros((num_wann, num_wann), dtype=complex)
        for (i, j), value in block.items():
            matrix[i - 1, j - 1] = value
        matrices[r_vector] = matrix
    return matrices, num_wann


def hr_hamiltonians(path: Path) -> tuple[dict[tuple[int, int, int], np.ndarray], int]:
    dataframe, num_wann, _ = hr.raw_read(str(path))
    matrices = {}
    for row in dataframe.itertuples(index=False):
        r_vector = (int(row.R1), int(row.R2), int(row.R3))
        matrix = matrices.setdefault(
            r_vector,
            np.zeros((num_wann, num_wann), dtype=complex),
        )
        matrix[int(row.i) - 1, int(row.j) - 1] = row.H
    return matrices, num_wann


class FeTbHamiltonianConsistencyTests(unittest.TestCase):
    def test_tb_hamiltonian_matches_reference_hr(self):
        tb_matrices, tb_num_wann = tb_hamiltonians(TB_PATH)
        hr_matrices, hr_num_wann = hr_hamiltonians(HR_PATH)

        self.assertEqual(tb_num_wann, hr_num_wann)
        self.assertEqual(set(tb_matrices), set(hr_matrices))

        difference_norms = {
            r_vector: np.linalg.norm(tb_matrices[r_vector] - hr_matrices[r_vector])
            for r_vector in tb_matrices
        }
        max_r = max(difference_norms, key=difference_norms.get)
        max_norm = difference_norms[max_r]
        self.assertLessEqual(
            max_norm,
            MAX_HAMILTONIAN_DIFFERENCE_NORM,
            f"Maximum H(R) difference norm is {max_norm:.6e} at R={max_r}.",
        )


if __name__ == "__main__":
    unittest.main()
