#!/usr/bin/env python3
"""Check real-space Hermiticity of matrices stored in a Wannier90 tb.dat file.

Hermiticity requires only

    X_ij(R) = X_ji(-R)^*

and does not assume time-reversal symmetry.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ssg4wann.parsergen.tb_parser import tb


@dataclass
class HermiticityResult:
    maximum_norm: float
    maximum_element_error: float
    r_vector: tuple[int, int, int] | None
    component: int | None
    indices: tuple[int, int] | None
    missing_opposite_vectors: list[tuple[int, int, int]]


def dense_matrix(block: dict, num_wann: int, vector_values: bool) -> np.ndarray:
    shape = (3, num_wann, num_wann) if vector_values else (num_wann, num_wann)
    matrix = np.zeros(shape, dtype=complex)
    for key, value in block.items():
        if not isinstance(key, tuple):
            continue
        i, j = key
        if vector_values:
            matrix[:, i - 1, j - 1] = np.asarray(value).reshape(3)
        else:
            matrix[i - 1, j - 1] = value
    return matrix


def check_matrix_blocks(
    blocks: dict,
    num_wann: int,
    *,
    vector_values: bool = False,
) -> HermiticityResult:
    """Check X(R) = X(-R)^dagger for scalar- or vector-valued matrices."""
    maximum_norm = 0.0
    maximum_element_error = 0.0
    maximum_r = None
    maximum_component = None
    maximum_indices = None
    missing = []

    for r_vector, block in blocks.items():
        minus_r = tuple(-value for value in r_vector)
        if minus_r not in blocks:
            missing.append(r_vector)
            continue

        matrix_r = dense_matrix(block, num_wann, vector_values)
        matrix_minus_r = dense_matrix(blocks[minus_r], num_wann, vector_values)
        if vector_values:
            difference = matrix_r - matrix_minus_r.conj().transpose(0, 2, 1)
        else:
            difference = matrix_r - matrix_minus_r.conj().T

        difference_norm = float(np.linalg.norm(difference))
        element_index = np.unravel_index(np.argmax(np.abs(difference)), difference.shape)
        element_error = float(np.abs(difference[element_index]))

        if difference_norm > maximum_norm:
            maximum_norm = difference_norm
            maximum_element_error = element_error
            maximum_r = r_vector
            if vector_values:
                maximum_component = int(element_index[0])
                maximum_indices = (int(element_index[1]) + 1, int(element_index[2]) + 1)
            else:
                maximum_indices = (int(element_index[0]) + 1, int(element_index[1]) + 1)

    return HermiticityResult(
        maximum_norm=maximum_norm,
        maximum_element_error=maximum_element_error,
        r_vector=maximum_r,
        component=maximum_component,
        indices=maximum_indices,
        missing_opposite_vectors=sorted(missing),
    )


def print_result(name: str, result: HermiticityResult, tolerance: float) -> bool:
    passed = (
        not result.missing_opposite_vectors
        and result.maximum_norm <= tolerance
    )
    print(f"{name}: {'PASS' if passed else 'FAIL'}")
    print(f"  max Frobenius norm : {result.maximum_norm:.12e}")
    print(f"  max element error  : {result.maximum_element_error:.12e}")
    print(f"  at R               : {result.r_vector}")
    if result.component is not None:
        print(f"  Cartesian component: {'xyz'[result.component]}")
    print(f"  matrix indices     : {result.indices}")
    print(f"  tolerance          : {tolerance:.12e}")
    if result.missing_opposite_vectors:
        print(
            "  missing -R vectors : "
            f"{len(result.missing_opposite_vectors)} "
            f"(first: {result.missing_opposite_vectors[:5]})"
        )
    return passed


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Check X_ij(R) = X_ji(-R)^* in a Wannier90 tb.dat file. "
            "No time-reversal symmetry is assumed."
        )
    )
    parser.add_argument("tb_file", type=Path, help="Path to the *_tb.dat file")
    parser.add_argument(
        "--quantity",
        choices=("H", "r", "all"),
        default="all",
        help="Matrix block to check (default: all)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1.0e-10,
        help="Maximum allowed Frobenius norm (default: 1e-10)",
    )
    args = parser.parse_args()

    parsed = tb.raw_read(str(args.tb_file))
    checks = []
    if args.quantity in ("H", "all"):
        checks.append(("Hamiltonian H", parsed["H"], False))
    if args.quantity in ("r", "all"):
        checks.append(("Position matrix r", parsed["r"], True))

    passed = True
    print(f"File: {args.tb_file}")
    print("Criterion: X_ij(R) = X_ji(-R)^* (Hermiticity only; no TR assumption)")
    for name, blocks, vector_values in checks:
        result = check_matrix_blocks(
            blocks,
            parsed["num_wann"],
            vector_values=vector_values,
        )
        passed = print_result(name, result, args.tolerance) and passed

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
