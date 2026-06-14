import tempfile
from pathlib import Path
import unittest

import numpy as np

from ssg4wann.mpi.parallel import calc_r_ent
from ssg4wann.parsergen.tb_parser import tb


def write_tb_fixture(path: Path):
    path.write_text(
        """ test tb
 1.0 0.0 0.0
 0.0 1.0 0.0
 0.0 0.0 1.0
 2
 1
 2

 0 0 0
 1 1 2.0 0.0
 2 1 4.0 0.0
 1 2 6.0 0.0
 2 2 8.0 0.0

 0 0 0
 1 1 0.2 0.0 0.4 0.0 0.6 0.0
 2 1 0.8 0.0 1.0 0.0 1.2 0.0
 1 2 1.4 0.0 1.6 0.0 1.8 0.0
 2 2 2.0 0.0 2.2 0.0 2.4 0.0
""",
        encoding="utf-8",
    )


class FakeOperator:
    def __init__(self, rotation=None, translation=None, bra_shift=None):
        self.rot_cart = np.eye(3) if rotation is None else np.asarray(rotation)
        self.translation_cart = (
            np.zeros((3, 1))
            if translation is None
            else np.asarray(translation).reshape(3, 1)
        )
        self._bra_shift = (
            np.zeros((3, 1))
            if bra_shift is None
            else np.asarray(bra_shift).reshape(3, 1)
        )

    def R_find(self, i, j, R, orbitals):
        return np.asarray(R).reshape(3, 1)

    def bra_cell_shift_cart(self, i, orbitals):
        return self._bra_shift


class TbParserTests(unittest.TestCase):
    def test_paths_are_derived_only_from_seed_name(self):
        noncollinear = object.__new__(tb)
        noncollinear.NONCOLLINEAR_channel = True
        self.assertEqual(
            noncollinear._resolve_paths("/work", "sample"),
            {"nc": "/work/sample_tb.dat"},
        )

        collinear = object.__new__(tb)
        collinear.NONCOLLINEAR_channel = False
        self.assertEqual(
            collinear._resolve_paths("/work", "sample"),
            {
                "up": "/work/sample.up_tb.dat",
                "dn": "/work/sample.dn_tb.dat",
            },
        )

    def test_collinear_channels_merge_like_hr_entries(self):
        with tempfile.TemporaryDirectory() as tempdir:
            write_tb_fixture(Path(tempdir) / "sample.up_tb.dat")
            write_tb_fixture(Path(tempdir) / "sample.dn_tb.dat")

            parser = tb(
                tempdir,
                "sample",
                NONCOLLINEAR_channel=False,
            )
            H_entry, r_entry, num_wann = parser.tb_entry()

            self.assertEqual(num_wann, 2)
            self.assertEqual(H_entry[(0, 0, 0)][(2, 2)]["up"], 4.0)
            self.assertEqual(H_entry[(0, 0, 0)][(2, 2)]["dn"], 4.0)
            np.testing.assert_allclose(
                r_entry[(0, 0, 0)][(1, 2)]["up"],
                np.array([0.7, 0.8, 0.9]),
            )
            np.testing.assert_allclose(
                r_entry[(0, 0, 0)][(1, 2)]["dn"],
                np.array([0.7, 0.8, 0.9]),
            )

    def test_read_normalizes_degeneracy_and_roundtrips(self):
        with tempfile.TemporaryDirectory() as tempdir:
            source = Path(tempdir) / "sample_tb.dat"
            write_tb_fixture(source)
            parsed = tb.raw_read(str(source))

            self.assertEqual(parsed["num_wann"], 2)
            self.assertEqual(parsed["nrpts"], 1)
            self.assertEqual(parsed["H"][(0, 0, 0)][(2, 2)], 4.0)
            np.testing.assert_allclose(
                parsed["r"][(0, 0, 0)][(1, 2)],
                np.array([0.7, 0.8, 0.9]),
            )

            H_records = [
                [(*r_tuple, i, j), value]
                for r_tuple, block in parsed["H"].items()
                for (i, j), value in block.items()
            ]
            r_records = [
                [(*r_tuple, i, j), value]
                for r_tuple, block in parsed["r"].items()
                for (i, j), value in block.items()
            ]
            tb._write_one(
                str(Path(tempdir) / "roundtrip_tb.dat"),
                parsed["lattice"],
                H_records,
                r_records,
                parsed["num_wann"],
            )
            roundtrip = tb.raw_read(str(Path(tempdir) / "roundtrip_tb.dat"))
            self.assertEqual(roundtrip["H"][(0, 0, 0)][(2, 2)], 4.0)
            np.testing.assert_allclose(
                roundtrip["r"][(0, 0, 0)][(1, 2)],
                np.array([0.7, 0.8, 0.9]),
            )

    def test_output_values_remain_space_separated_at_fixed_precision(self):
        with tempfile.TemporaryDirectory() as tempdir:
            output = Path(tempdir) / "wide_values_tb.dat"
            h_value = complex(1.2345678901234567e123, -9.876543210987654e-123)
            r_value = np.array(
                [
                    complex(-1.2345678901234567e123, -2.0),
                    complex(3.0, -4.567890123456789e99),
                    complex(-5.0, 6.0),
                ]
            )
            tb._write_one(
                str(output),
                np.eye(3),
                [[(0, 0, 0, 1, 1), h_value]],
                [[(0, 0, 0, 1, 1), r_value]],
                num_wann=1,
            )

            matrix_lines = [
                line.split()
                for line in output.read_text(encoding="utf-8").splitlines()
                if line.split()[:2] == ["1", "1"]
            ]
            self.assertEqual([len(parts) for parts in matrix_lines], [4, 8])

            parsed = tb.raw_read(str(output))
            self.assertEqual(parsed["H"][(0, 0, 0)][(1, 1)], h_value)
            np.testing.assert_allclose(
                parsed["r"][(0, 0, 0)][(1, 1)],
                r_value,
            )


class PositionSymmetrizationTests(unittest.TestCase):
    def run_kernel(self, operator, vector):
        r_entry = {
            (0, 0, 0): {
                "Rvec": np.zeros((3, 1), dtype=int),
                (1, 1): np.asarray(vector, dtype=complex),
            }
        }
        result = calc_r_ent(
            R=(0, 0, 0),
            num_wann=1,
            opset={0: [operator, False]},
            actdict={(0, 1): [(1, 1.0)]},
            r_entry=r_entry,
            orbSpin=[],
            nsymm=1,
            NONCOLLINEAR_channel=True,
        )
        return result[0][1]

    def test_identity_leaves_position_matrix_unchanged(self):
        vector = np.array([0.25, -0.5, 0.75])
        np.testing.assert_allclose(self.run_kernel(FakeOperator(), vector), vector)

    def test_rotation_is_pulled_back_to_target_components(self):
        rotation = np.array(
            [
                [0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        target = np.array([0.3, -0.2, 0.4])
        transformed = rotation @ target
        np.testing.assert_allclose(
            self.run_kernel(FakeOperator(rotation=rotation), transformed),
            target,
        )

    def test_affine_cell_and_symmetry_translations_are_removed(self):
        raw_home_cell_value = np.array([0.2, 0.3, 0.4])
        operator = FakeOperator(
            translation=np.array([0.5, 0.0, 0.0]),
            bra_shift=np.array([1.0, 0.0, 0.0]),
        )
        expected = raw_home_cell_value + np.array([0.5, 0.0, 0.0])
        np.testing.assert_allclose(
            self.run_kernel(operator, raw_home_cell_value),
            expected,
        )


if __name__ == "__main__":
    unittest.main()
