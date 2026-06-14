from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from ssg4wann.main import _write_hr_from_tb
from ssg4wann.parsergen.hr_parser import hr
from ssg4wann.parsergen.inload import Config, infoload


class OutputHrFromTbConfigTests(unittest.TestCase):
    def test_defaults_are_disabled(self):
        config = Config()
        self.assertFalse(config.output_hr_from_tb)

    def test_optional_tags_are_parsed(self):
        with tempfile.TemporaryDirectory() as tempdir:
            config_path = Path(tempdir) / "sg.in"
            config_path.write_text(
                """
                soc = False
                use_win = wannier90.win
                tb_mode = True
                output_hr_from_tb = True
                NONCOLLINEAR_channel = True
                """,
                encoding="utf-8",
            )
            config = infoload(str(config_path), rank=0)

        self.assertTrue(config.tb_mode)
        self.assertTrue(config.output_hr_from_tb)


class OutputHrFromTbWriterTests(unittest.TestCase):
    @staticmethod
    def config(**overrides):
        values = {
            "tb_mode": True,
            "output_hr_from_tb": True,
            "seed": "sample",
            "NONCOLLINEAR_channel": True,
            "chnl": True,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_no_output_outside_tb_mode(self):
        with tempfile.TemporaryDirectory() as tempdir:
            config = self.config(tb_mode=False)
            _write_hr_from_tb(
                config,
                tempdir,
                [[(0, 0, 0, 1, 1), 2.0 + 0.0j]],
                num_wann=1,
                nrpts=1,
            )
            self.assertFalse(Path(tempdir, "sample_symmed_hr.dat").exists())

    def test_no_output_when_option_is_disabled(self):
        with tempfile.TemporaryDirectory() as tempdir:
            config = self.config(output_hr_from_tb=False)
            _write_hr_from_tb(
                config,
                tempdir,
                [[(0, 0, 0, 1, 1), 2.0 + 0.0j]],
                num_wann=1,
                nrpts=1,
            )
            self.assertFalse(Path(tempdir, "sample_symmed_hr.dat").exists())

    def test_noncollinear_output_uses_standard_hr_writer(self):
        with tempfile.TemporaryDirectory() as tempdir:
            _write_hr_from_tb(
                self.config(),
                tempdir,
                [[(0, 0, 0, 1, 1), 2.0 + 0.5j]],
                num_wann=1,
                nrpts=1,
            )
            output = Path(tempdir) / "sample_symmed_hr.dat"
            dataframe, num_wann, nrpts = hr.raw_read(str(output))

            self.assertTrue(output.exists())
            self.assertEqual((num_wann, nrpts), (1, 1))
            self.assertEqual(dataframe.iloc[0].H, 2.0 + 0.5j)

    def test_collinear_output_is_split_into_spin_channels(self):
        with tempfile.TemporaryDirectory() as tempdir:
            records = [
                [(0, 0, 0, 1, 1), 2.0 + 0.0j],
                [(0, 0, 0, 2, 2), 3.0 + 0.0j],
            ]
            _write_hr_from_tb(
                self.config(NONCOLLINEAR_channel=False),
                tempdir,
                records,
                num_wann=1,
                nrpts=1,
            )

            up, _, _ = hr.raw_read(str(Path(tempdir) / "sample.up_symmed_hr.dat"))
            down, _, _ = hr.raw_read(str(Path(tempdir) / "sample.dn_symmed_hr.dat"))
            self.assertEqual(up.iloc[0].H, 2.0 + 0.0j)
            self.assertEqual(down.iloc[0].H, 3.0 + 0.0j)


if __name__ == "__main__":
    unittest.main()
