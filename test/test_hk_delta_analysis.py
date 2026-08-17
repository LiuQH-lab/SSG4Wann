import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ssg4wann.core.wannob import lat
from ssg4wann.parsergen.hr_parser import hr
from ssg4wann.parsergen.inload import infoload


EPS = np.finfo(float).eps


def write_one_band_hr(path: Path, onsite: float, hopping_x: float) -> None:
    path.write_text(
        f""" test hr
           1
           3
    1    1    1
    0    0    0    1    1 {onsite: .8f}  0.00000000
    1    0    0    1    1 {hopping_x: .8f}  0.00000000
   -1    0    0    1    1 {hopping_x: .8f}  0.00000000
""",
        encoding="utf-8",
    )


def write_unit_lattice_win(path: Path) -> None:
    path.write_text(
        """begin unit_cell_cart
1.0 0.0 0.0
0.0 1.0 0.0
0.0 0.0 1.0
end unit_cell_cart
""",
        encoding="utf-8",
    )


def load_lattice_from_config(config_path: Path, winpath: str) -> tuple[np.ndarray, np.ndarray]:
    win_file = Path(winpath)
    if not win_file.is_absolute():
        win_file = config_path.parent / win_file
    content = win_file.read_text(encoding="utf-8")
    return lat(content)


def load_matrix_hr(hr_path: Path, noncollinear_channel: bool) -> tuple[dict, int]:
    parser = hr(
        str(hr_path.parent),
        "unused",
        NONCOLLINEAR_channel=noncollinear_channel,
        hr4trans=str(hr_path),
    )
    hr_entry, num_wann = parser.hr_entry()
    return hr.convert(hr_entry, num_wann), num_wann


def segment_k_points(segment: dict, bands_num_points: int, permuK: np.ndarray) -> tuple[np.ndarray, float]:
    k_points, x_axis, _ = hr.Kpoints_gen(bands_num_points, [segment], permuK)
    return k_points, float(x_axis[-1] - x_axis[0])


def raw_bandwidth(
    matrix_hr: dict,
    num_wann: int,
    k_segments: list[dict],
    bands_num_points: int,
    permuK: np.ndarray,
    permutation: np.ndarray,
) -> float:
    eigenvalues = []
    for segment in k_segments:
        k_points, _ = segment_k_points(segment, bands_num_points, permuK)
        for kpoint in k_points:
            Hk = hr.Hk_gen(matrix_hr, num_wann, kpoint, permuK, permutation)
            eigenvalues.extend(np.linalg.eigvalsh(Hk))
    return float(np.max(eigenvalues) - np.min(eigenvalues))


def compute_hk_path_delta_metrics(
    raw_hr_path: Path,
    sym_hr_path: Path,
    config_path: Path,
    *,
    noncollinear_channel: bool | None = None,
) -> dict:
    config = infoload(str(config_path), rank=0)
    channel_flag = (
        config.NONCOLLINEAR_channel
        if noncollinear_channel is None
        else noncollinear_channel
    )
    if config.bands_num_points < 2:
        raise ValueError("bands_num_points must be at least 2 for path sampling.")
    if not config.kpath_segments:
        raise ValueError("sg.in must contain a kpoint_path block.")

    permutation, permuK = load_lattice_from_config(config_path, config.winpath)
    raw_matrix_hr, raw_num_wann = load_matrix_hr(raw_hr_path, channel_flag)
    sym_matrix_hr, sym_num_wann = load_matrix_hr(sym_hr_path, channel_flag)
    if raw_num_wann != sym_num_wann:
        raise ValueError(
            f"HR dimensions differ: raw has {raw_num_wann}, sym has {sym_num_wann}."
        )

    energy_scale = raw_bandwidth(
        raw_matrix_hr,
        raw_num_wann,
        config.kpath_segments,
        config.bands_num_points,
        permuK,
        permutation,
    )

    segments = []
    total_length = 0.0
    for segment in config.kpath_segments:
        k_points, segment_length = segment_k_points(
            segment, config.bands_num_points, permuK
        )
        abs_deltas = []
        rel_deltas = []
        rel_scale_deltas = []
        for kpoint in k_points:
            raw_hk = hr.Hk_gen(
                raw_matrix_hr, raw_num_wann, kpoint, permuK, permutation
            )
            sym_hk = hr.Hk_gen(
                sym_matrix_hr, sym_num_wann, kpoint, permuK, permutation
            )
            delta_norm = float(np.linalg.norm(sym_hk - raw_hk, ord="fro"))
            raw_norm = float(np.linalg.norm(raw_hk, ord="fro"))
            abs_deltas.append(delta_norm)
            rel_deltas.append(delta_norm / max(raw_norm, EPS))
            rel_scale_deltas.append(delta_norm / max(energy_scale, EPS))

        segments.append(
            {
                "label_start": segment["label_start"],
                "label_end": segment["label_end"],
                "length": segment_length,
                "mean_abs_delta": float(np.mean(abs_deltas)),
                "mean_rel_delta": float(np.mean(rel_deltas)),
                "mean_rel_delta_scale": float(np.mean(rel_scale_deltas)),
            }
        )
        total_length += segment_length

    if total_length <= 0:
        raise ValueError("The total k-path length must be positive.")

    total = {
        "abs_delta": 0.0,
        "rel_delta": 0.0,
        "rel_delta_scale": 0.0,
    }
    for segment in segments:
        weight = segment["length"] / total_length
        segment["weight"] = weight
        total["abs_delta"] += segment["mean_abs_delta"] * weight
        total["rel_delta"] += segment["mean_rel_delta"] * weight
        total["rel_delta_scale"] += segment["mean_rel_delta_scale"] * weight

    return {
        "energy_scale": energy_scale,
        "total_length": total_length,
        "total": total,
        "segments": segments,
    }


class HkDeltaAnalysisTests(unittest.TestCase):
    def test_path_weighted_delta_metrics_for_one_band_fixture(self):
        with tempfile.TemporaryDirectory() as tempdir:
            workdir = Path(tempdir)
            raw_hr = workdir / "raw_hr.dat"
            sym_hr = workdir / "sym_hr.dat"
            win = workdir / "wannier90.win"
            config = workdir / "sg.in"

            write_one_band_hr(raw_hr, onsite=2.0, hopping_x=0.5)
            write_one_band_hr(sym_hr, onsite=2.2, hopping_x=0.6)
            write_unit_lattice_win(win)
            config.write_text(
                """
                soc = False
                use_win = wannier90.win
                bands_num_points = 3
                NONCOLLINEAR_channel = True
                spin_direction = 0 0 1
                begin kpoint_path
                G 0.0 0.0 0.0 X 0.5 0.0 0.0
                X 0.5 0.0 0.0 Y 0.5 1.0 0.0
                end kpoint_path
                """,
                encoding="utf-8",
            )

            metrics = compute_hk_path_delta_metrics(raw_hr, sym_hr, config)

            self.assertAlmostEqual(metrics["energy_scale"], 2.0)
            self.assertAlmostEqual(metrics["total_length"], 3.0 * math.pi)

            first, second = metrics["segments"]
            self.assertAlmostEqual(first["length"], math.pi)
            self.assertAlmostEqual(second["length"], 2.0 * math.pi)
            self.assertAlmostEqual(first["weight"], 1.0 / 3.0)
            self.assertAlmostEqual(second["weight"], 2.0 / 3.0)

            expected_first_abs = np.mean([0.4, 0.2, 0.0])
            expected_first_rel = np.mean([0.4 / 3.0, 0.2 / 2.0, 0.0 / 1.0])
            expected_first_scale = np.mean([0.4 / 2.0, 0.2 / 2.0, 0.0 / 2.0])

            self.assertAlmostEqual(first["mean_abs_delta"], expected_first_abs)
            self.assertAlmostEqual(first["mean_rel_delta"], expected_first_rel)
            self.assertAlmostEqual(first["mean_rel_delta_scale"], expected_first_scale)
            self.assertAlmostEqual(second["mean_abs_delta"], 0.0)
            self.assertAlmostEqual(second["mean_rel_delta"], 0.0)
            self.assertAlmostEqual(second["mean_rel_delta_scale"], 0.0)

            self.assertAlmostEqual(
                metrics["total"]["abs_delta"], expected_first_abs / 3.0
            )
            self.assertAlmostEqual(
                metrics["total"]["rel_delta"], expected_first_rel / 3.0
            )
            self.assertAlmostEqual(
                metrics["total"]["rel_delta_scale"], expected_first_scale / 3.0
            )

    def assert_metrics_are_finite(self, metrics: dict) -> None:
        self.assertGreater(metrics["energy_scale"], 0.0)
        self.assertGreater(metrics["total_length"], 0.0)
        self.assertTrue(metrics["segments"])
        self.assertAlmostEqual(
            sum(segment["weight"] for segment in metrics["segments"]),
            1.0,
        )
        for key in ("abs_delta", "rel_delta", "rel_delta_scale"):
            self.assertTrue(np.isfinite(metrics["total"][key]), key)
            self.assertGreaterEqual(metrics["total"][key], 0.0)
        for segment in metrics["segments"]:
            for key in (
                "length",
                "weight",
                "mean_abs_delta",
                "mean_rel_delta",
                "mean_rel_delta_scale",
            ):
                self.assertTrue(np.isfinite(segment[key]), key)
                self.assertGreaterEqual(segment[key], 0.0)

    def test_fe_example_channel_metrics_are_finite(self):
        example = Path(__file__).resolve().parents[1] / "examples" / "Fe"
        config = example / "sg.in"
        for channel in ("up", "dn"):
            with self.subTest(channel=channel):
                metrics = compute_hk_path_delta_metrics(
                    example / f"wannier90.{channel}_hr.dat",
                    example / f"wannier90.{channel}_symmed_hr.dat",
                    config,
                    noncollinear_channel=False,
                )
                self.assert_metrics_are_finite(metrics)

    def test_fe_soc_example_metrics_are_finite(self):
        example = Path(__file__).resolve().parents[1] / "examples" / "Fe_SOC"
        metrics = compute_hk_path_delta_metrics(
            example / "wannier90_hr.dat",
            example / "wannier90_symmed_hr.dat",
            example / "sg.in",
            noncollinear_channel=True,
        )
        self.assert_metrics_are_finite(metrics)

    def test_nb3vs6_example_channel_metrics_are_finite_when_symmed_files_exist(self):
        example = Path(__file__).resolve().parents[1] / "examples" / "Nb3VS6"
        missing = [
            path
            for path in (
                example / "wannier90.up_symmed_hr.dat",
                example / "wannier90.dn_symmed_hr.dat",
            )
            if not path.exists()
        ]
        if missing:
            self.skipTest(
                "Nb3VS6 example does not include symmetrized HR files: "
                + ", ".join(path.name for path in missing)
            )

        for channel in ("up", "dn"):
            with self.subTest(channel=channel):
                metrics = compute_hk_path_delta_metrics(
                    example / f"wannier90.{channel}_hr.dat",
                    example / f"wannier90.{channel}_symmed_hr.dat",
                    example / "sg.in",
                    noncollinear_channel=False,
                )
                self.assert_metrics_are_finite(metrics)


if __name__ == "__main__":
    unittest.main()
