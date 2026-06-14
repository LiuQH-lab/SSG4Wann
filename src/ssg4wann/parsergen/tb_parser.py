from __future__ import annotations

from datetime import datetime
import os
import re

import numpy as np

from ..exceptions import ConfigParseError, WannierMatchError


class tb:
    _OUTPUT_VALUE_FORMAT = "{:26.16E}"
    _NUMBER_PATTERN = re.compile(
        r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?"
    )

    def __init__(
        self,
        cwd: str,
        seed: str,
        NONCOLLINEAR_channel: bool,
        tb4trans: str | None = None,
    ):
        self.NONCOLLINEAR_channel = NONCOLLINEAR_channel
        self.tb4trans = tb4trans
        self.paths = (
            {"nc": tb4trans}
            if tb4trans is not None
            else self._resolve_paths(cwd, seed)
        )
        self.lattice = None
        self.num_wann = None
        self.nrpts = None
        self.degeneracies = None
        self.raw_data_dict = {}
        self.rawload()

    def _resolve_paths(self, cwd: str, seed: str) -> dict[str, str]:
        if self.NONCOLLINEAR_channel:
            return {"nc": os.path.join(cwd, f"{seed}_tb.dat")}

        return {
            "up": os.path.join(cwd, f"{seed}.up_tb.dat"),
            "dn": os.path.join(cwd, f"{seed}.dn_tb.dat"),
        }

    def rawload(self) -> None:
        channels = (
            ["nc"]
            if self.NONCOLLINEAR_channel or self.tb4trans is not None
            else ["up", "dn"]
        )
        reference = None

        for channel in channels:
            filepath = self.paths[channel]
            print(f"loading: {filepath}")
            parsed = self.raw_read(filepath)
            metadata = (
                parsed["lattice"],
                parsed["num_wann"],
                parsed["nrpts"],
                parsed["degeneracies"],
                parsed["r_vectors"],
            )
            if reference is None:
                reference = metadata
                (
                    self.lattice,
                    self.num_wann,
                    self.nrpts,
                    self.degeneracies,
                    _,
                ) = metadata
            else:
                self._validate_channel_metadata(reference, metadata, filepath)

            self.raw_data_dict[channel] = {
                "H": parsed["H"],
                "r": parsed["r"],
            }

    @staticmethod
    def _validate_channel_metadata(reference, candidate, filepath: str) -> None:
        ref_lattice, ref_num_wann, ref_nrpts, ref_degen, ref_rvecs = reference
        lattice, num_wann, nrpts, degen, rvecs = candidate
        if not np.allclose(ref_lattice, lattice):
            raise ConfigParseError(f"Inconsistent lattice vectors in {filepath}.")
        if (ref_num_wann, ref_nrpts) != (num_wann, nrpts):
            raise ConfigParseError(f"Inconsistent tb dimensions in {filepath}.")
        if ref_degen != degen or ref_rvecs != rvecs:
            raise ConfigParseError(f"Inconsistent R-vector data in {filepath}.")

    @staticmethod
    def _next_nonempty(lines: list[str], cursor: int) -> tuple[str, int]:
        while cursor < len(lines) and not lines[cursor].strip():
            cursor += 1
        if cursor >= len(lines):
            raise ConfigParseError("Unexpected end of tb.dat.")
        return lines[cursor].strip(), cursor + 1

    @classmethod
    def _parse_numbers(cls, line: str) -> list[float]:
        return [
            float(value.replace("D", "E").replace("d", "e"))
            for value in cls._NUMBER_PATTERN.findall(line)
        ]

    @classmethod
    def _read_matrix_block(
        cls,
        lines: list[str],
        cursor: int,
        num_wann: int,
        nrpts: int,
        degeneracies: list[int],
        vector_values: bool,
    ):
        data = {}
        r_vectors = []
        value_count = 6 if vector_values else 2

        for r_index in range(nrpts):
            r_line, cursor = cls._next_nonempty(lines, cursor)
            r_parts = r_line.split()
            if len(r_parts) != 3:
                raise ConfigParseError(f"Invalid R-vector line in tb.dat: '{r_line}'")
            r_tuple = tuple(int(value) for value in r_parts)
            r_vectors.append(r_tuple)
            block = {}

            for _ in range(num_wann**2):
                entry_line, cursor = cls._next_nonempty(lines, cursor)
                parts = cls._parse_numbers(entry_line)
                if len(parts) < 2 + value_count:
                    raise ConfigParseError(f"Invalid matrix-element line in tb.dat: '{entry_line}'")

                i, j = int(parts[0]), int(parts[1])
                values = parts[2 : 2 + value_count]
                omega = degeneracies[r_index]
                if vector_values:
                    block[(i, j)] = np.array(
                        [
                            complex(values[0], values[1]),
                            complex(values[2], values[3]),
                            complex(values[4], values[5]),
                        ],
                        dtype=complex,
                    ) / omega
                else:
                    block[(i, j)] = complex(values[0], values[1]) / omega

            data[r_tuple] = block

        return data, r_vectors, cursor

    @classmethod
    def raw_read(cls, filepath: str) -> dict:
        try:
            with open(filepath, "r") as handle:
                lines = handle.readlines()
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"tb.dat file not found: {filepath}") from exc

        if len(lines) < 6:
            raise ConfigParseError(f"Incomplete tb.dat header in {filepath}.")

        cursor = 1
        lattice = []
        for _ in range(3):
            line, cursor = cls._next_nonempty(lines, cursor)
            values = cls._parse_numbers(line)
            if len(values) != 3:
                raise ConfigParseError(f"Invalid lattice-vector line in {filepath}: '{line}'")
            lattice.append(values)

        line, cursor = cls._next_nonempty(lines, cursor)
        num_wann = int(line)
        line, cursor = cls._next_nonempty(lines, cursor)
        nrpts = int(line)

        degeneracies = []
        while len(degeneracies) < nrpts:
            line, cursor = cls._next_nonempty(lines, cursor)
            degeneracies.extend(int(value) for value in line.split())
        if len(degeneracies) != nrpts:
            raise ConfigParseError(f"Wrong number of degeneracies in {filepath}.")

        H_data, h_r_vectors, cursor = cls._read_matrix_block(
            lines, cursor, num_wann, nrpts, degeneracies, vector_values=False
        )
        r_data, r_r_vectors, cursor = cls._read_matrix_block(
            lines, cursor, num_wann, nrpts, degeneracies, vector_values=True
        )
        if h_r_vectors != r_r_vectors:
            raise ConfigParseError(f"H and r blocks use different R-vector sequences in {filepath}.")

        return {
            "lattice": np.array(lattice, dtype=float).T,
            "num_wann": num_wann,
            "nrpts": nrpts,
            "degeneracies": degeneracies,
            "r_vectors": h_r_vectors,
            "H": H_data,
            "r": r_data,
        }

    def _entry(self, quantity: str) -> dict:
        result = {}
        direct_matrix = self.NONCOLLINEAR_channel or self.tb4trans is not None
        channels = ["nc"] if direct_matrix else ["up", "dn"]

        for channel in channels:
            for r_tuple, raw_block in self.raw_data_dict[channel][quantity].items():
                block = result.setdefault(
                    r_tuple,
                    {"Rvec": np.array(r_tuple, dtype=int).reshape(3, 1)},
                )
                for indices, value in raw_block.items():
                    if direct_matrix:
                        block[indices] = value
                    else:
                        block.setdefault(indices, {})[channel] = value
        return result

    def tb_entry(self):
        H_entry = self._entry("H")
        r_entry = self._entry("r")
        self.raw_data_dict.clear()
        return H_entry, r_entry, self.num_wann

    @staticmethod
    def hermitize_r(r_symm, total_wann: int):
        matrix_r = {}
        for key_tuple, value in r_symm:
            rx, ry, rz, i, j = key_tuple
            r_tuple = (rx, ry, rz)
            matrix_r.setdefault(
                r_tuple,
                np.zeros((3, total_wann, total_wann), dtype=complex),
            )[:, i - 1, j - 1] = np.asarray(value).reshape(3)

        hermitian_r = {}
        processed = set()
        for r_tuple, matrix in matrix_r.items():
            if r_tuple in processed:
                continue
            minus_r = tuple(-value for value in r_tuple)
            minus_matrix = matrix_r.get(minus_r)
            if minus_matrix is None:
                hermitian_r[r_tuple] = matrix
                hermitian_r[minus_r] = matrix.conj().transpose(0, 2, 1)
            else:
                averaged = 0.5 * (matrix + minus_matrix.conj().transpose(0, 2, 1))
                hermitian_r[r_tuple] = averaged
                hermitian_r[minus_r] = averaged.conj().transpose(0, 2, 1)
            processed.update((r_tuple, minus_r))

        result = []
        for r_tuple, matrix in hermitian_r.items():
            for i in range(total_wann):
                for j in range(total_wann):
                    result.append(
                        [
                            (*r_tuple, i + 1, j + 1),
                            matrix[:, i, j].copy(),
                        ]
                    )
        return result

    @staticmethod
    def _split_channel_records(records, num_wann: int, channel: str):
        output = []
        offset = 0 if channel == "up" else num_wann
        for key, value in records:
            r1, r2, r3, i, j = key
            in_channel = (
                i <= num_wann and j <= num_wann
                if channel == "up"
                else i > num_wann and j > num_wann
            )
            if in_channel:
                output.append(((r1, r2, r3, i - offset, j - offset), value))
            elif (i <= num_wann) != (j <= num_wann):
                raise WannierMatchError(
                    f"Spin-mixing entry ({i}, {j}) cannot be written to collinear tb.dat."
                )
        return output

    @classmethod
    def outwrite(
        cls,
        cwd: str,
        seed: str,
        lattice: np.ndarray,
        H_reco,
        r_reco,
        num_wann: int,
        NONCOLLINEAR_channel: bool,
    ) -> None:
        if NONCOLLINEAR_channel:
            cls._write_one(
                os.path.join(cwd, f"{seed}_symmed_tb.dat"),
                lattice,
                H_reco,
                r_reco,
                num_wann,
            )
            return

        for channel in ("up", "dn"):
            cls._write_one(
                os.path.join(cwd, f"{seed}.{channel}_symmed_tb.dat"),
                lattice,
                cls._split_channel_records(H_reco, num_wann, channel),
                cls._split_channel_records(r_reco, num_wann, channel),
                num_wann,
            )

    @staticmethod
    def _write_one(
        filepath: str,
        lattice: np.ndarray,
        H_reco,
        r_reco,
        num_wann: int,
    ) -> None:
        H_reco = sorted(H_reco, key=lambda rec: (*rec[0][:3], rec[0][4], rec[0][3]))
        r_reco = sorted(r_reco, key=lambda rec: (*rec[0][:3], rec[0][4], rec[0][3]))
        h_by_r = {}
        r_by_r = {}
        for key, value in H_reco:
            h_by_r.setdefault(tuple(key[:3]), []).append((key[3], key[4], value))
        for key, value in r_reco:
            r_by_r.setdefault(tuple(key[:3]), []).append((key[3], key[4], value))

        r_vectors = sorted(h_by_r)
        if set(r_vectors) != set(r_by_r):
            raise WannierMatchError("Symmetrized H and r contain different R-vector sets.")
        expected_entries = num_wann**2
        for r_tuple in r_vectors:
            if len(h_by_r[r_tuple]) != expected_entries or len(r_by_r[r_tuple]) != expected_entries:
                raise WannierMatchError(f"Incomplete matrix block for R={r_tuple}.")

        value_format = tb._OUTPUT_VALUE_FORMAT
        lattice_rows = np.asarray(lattice).T

        with open(filepath, "w") as handle:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            handle.write(f" written by SSG4Wann at {now}\n")
            for row in lattice_rows:
                handle.write(" ".join(value_format.format(float(value)) for value in row) + "\n")
            handle.write(f"{num_wann:12d}\n")
            handle.write(f"{len(r_vectors):12d}\n")
            degeneracies = [1] * len(r_vectors)
            for start in range(0, len(degeneracies), 15):
                handle.write("".join(f"{value:5d}" for value in degeneracies[start : start + 15]) + "\n")

            for r_tuple in r_vectors:
                handle.write("\n")
                handle.write(f"{r_tuple[0]:5d}{r_tuple[1]:5d}{r_tuple[2]:5d}\n")
                for i, j, value in h_by_r[r_tuple]:
                    handle.write(
                        f"{i:5d}{j:5d}"
                        f" {value_format.format(complex(value).real)}"
                        f" {value_format.format(complex(value).imag)}\n"
                    )

            for r_tuple in r_vectors:
                handle.write("\n")
                handle.write(f"{r_tuple[0]:5d}{r_tuple[1]:5d}{r_tuple[2]:5d}\n")
                for i, j, value in r_by_r[r_tuple]:
                    vector = np.asarray(value).reshape(3)
                    handle.write(
                        f"{i:5d}{j:5d}"
                        + " "
                        + " ".join(
                            value_format.format(component)
                            for item in vector
                            for component in (item.real, item.imag)
                        )
                        + "\n"
                    )
