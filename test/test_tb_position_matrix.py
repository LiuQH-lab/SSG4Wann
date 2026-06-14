from pathlib import Path
import unittest

import numpy as np

from ssg4wann.parsergen.tb_parser import tb


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FE_TB_DIRECTORY = REPOSITORY_ROOT / "examples/Fe_tb"
CHANNELS = ("up", "dn")
CENTER_TOLERANCE = 1.0e-10
HERMITICITY_TOLERANCE = 1.0e-10


def position_matrices(block, num_wann: int) -> np.ndarray:
    matrices = np.zeros((3, num_wann, num_wann), dtype=complex)
    for (i, j), value in block.items():
        matrices[:, i - 1, j - 1] = value
    return matrices


class FeTbPositionMatrixTests(unittest.TestCase):
    @staticmethod
    def read_channel(channel: str) -> dict:
        path = FE_TB_DIRECTORY / f"wannier90.{channel}_symmed_tb.dat"
        return tb.raw_read(str(path))

    def test_home_cell_diagonal_centers_are_lattice_vectors(self):
        for channel in CHANNELS:
            with self.subTest(channel=channel):
                parsed = self.read_channel(channel)
                lattice = parsed["lattice"]
                zero_block = parsed["r"][(0, 0, 0)]

                for orbital in range(1, parsed["num_wann"] + 1):
                    center = zero_block[(orbital, orbital)]
                    fractional_center = np.linalg.solve(lattice, center.real)
                    nearest_lattice_point = np.rint(fractional_center)
                    residual = center.real - lattice @ nearest_lattice_point

                    self.assertLessEqual(
                        np.linalg.norm(center.imag),
                        CENTER_TOLERANCE,
                        f"{channel} orbital {orbital} has a complex Wannier center.",
                    )
                    self.assertLessEqual(
                        np.linalg.norm(residual),
                        CENTER_TOLERANCE,
                        (
                            f"{channel} orbital {orbital} center is not equivalent "
                            f"to a lattice vector: center={center.real}, "
                            f"fractional={fractional_center}."
                        ),
                    )

    def test_position_matrix_is_hermitian_in_real_space(self):
        for channel in CHANNELS:
            with self.subTest(channel=channel):
                parsed = self.read_channel(channel)
                num_wann = parsed["num_wann"]
                max_norm = 0.0
                max_location = None

                for r_vector, block in parsed["r"].items():
                    minus_r = tuple(-value for value in r_vector)
                    matrix_r = position_matrices(block, num_wann)
                    matrix_minus_r = position_matrices(
                        parsed["r"].get(minus_r, {}),
                        num_wann,
                    )
                    difference = matrix_r - matrix_minus_r.conj().transpose(0, 2, 1)
                    difference_norm = np.linalg.norm(difference)
                    if difference_norm > max_norm:
                        max_norm = difference_norm
                        max_location = r_vector

                self.assertLessEqual(
                    max_norm,
                    HERMITICITY_TOLERANCE,
                    (
                        f"{channel} position matrix violates "
                        f"r(R)=r(-R)^dagger: maximum norm={max_norm:.6e} "
                        f"at R={max_location}."
                    ),
                )


if __name__ == "__main__":
    unittest.main()
