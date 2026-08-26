# SSG4Wann Development Guide

This file defines the project-specific rules for coding agents and developers
working in this repository. Read the relevant sections before changing the
implementation.

`SSG4Wann` is scientific software. Code that runs without an exception is not
necessarily mathematically or physically correct. Preserve the symmetry,
basis, coordinate, and file-format conventions described below.

## Project Scope

`SSG4Wann` symmetrizes real-space Wannier tight-binding models with the
Oriented Spin Space Group (OSSG), or with its Magnetic Space Group (MSG)
subgroup when spin-orbit coupling is included.

The implementation follows the method described in:

> SSG4Wann: A code for Real-Space Hamiltonian Symmetrization with Oriented
> Spin Space Group

The main workflow is:

1. Parse `sg.in` and the Wannier90 `.win` file.
2. Read a Wannier90 `*_hr.dat` or `*_tb.dat` model.
3. Expand Wigner-Seitz degeneracy weights into explicit real-space terms.
4. Obtain OSSG or MSG operations from `findspingroup`.
5. Construct the orbital and spin representations of every operation.
6. Map Wannier centers, basis indices, and relative lattice vectors.
7. Apply the group average to the Hamiltonian and, in TB mode, the position
   matrix.
8. Write an explicit real-space model with unit degeneracy weights.

Scientific correctness takes priority over cosmetic simplification or small
performance gains.

## Core Algorithm Guardrails

- Keep the meaning of the SOC flag fixed:
  - `soc = False` selects the full OSSG through `payload["ssg"]["ops"]`.
  - `soc = True` selects the MSG subgroup through `payload["msg"]["ops"]`.
- Do not silently substitute another orbital, atom, spin channel, lattice
  vector, or symmetry operation when a Wannier-basis mapping fails. Raise an
  error with the operation and basis information needed for diagnosis.
- Keep the cell, orbital basis, and symmetry operation in the same coordinate
  setting. Do not transform the cell and the symmetry data with unrelated
  matrices.
- Do not use `forced_hermitianize` to hide an incorrect symmetry action,
  anti-unitary conjugation, lattice-vector mapping, or position-matrix
  translation.
- Do not remove symmetry operations merely because their spatial parts are
  equal. The spin operations may still be different and physically essential.
- Do not use the spin-only speedup outside the conditions under which the group
  decomposition and real-Hamiltonian constraint are valid.
- Do not treat the three Cartesian components of the TB position matrix as
  scalar Hamiltonians. Their vector rotation and affine translation terms must
  remain paired.
- Keep failures visible. Avoid silent fallbacks that produce a plausible file
  while concealing a basis, group, or coordinate inconsistency.

## Repository Map

- `src/ssg4wann/cli.py`: CLI, input discovery, and `sg.in` generation.
- `src/ssg4wann/api.py`: public Python API (`quick_run`).
- `src/ssg4wann/main.py`: top-level symmetrization and band workflows.
- `src/ssg4wann/parsergen/inload.py`: configuration model and validation.
- `src/ssg4wann/parsergen/hr_parser.py`: HR parsing, normalization, band
  conversion, and Hermitian post-processing.
- `src/ssg4wann/parsergen/tb_parser.py`: TB Hamiltonian and position-matrix
  parsing and writing.
- `src/ssg4wann/parsergen/generate.py`: generated configuration and output
  writers.
- `src/ssg4wann/core/wannob.py`: lattice, atom, projection, and Wannier-basis
  parsing.
- `src/ssg4wann/core/cartesian_tensors.py`: Cartesian-tensor orbital
  representations for `s`, `p`, `d`, and `f` orbitals.
- `src/ssg4wann/core/ops_act.py`: spatial, spin, and time-reversal actions on
  Wannier functions.
- `src/ssg4wann/core/sogroup.py`: spin-only/nontrivial-group decomposition.
- `src/ssg4wann/mpi/parallel.py`: operation preprocessing and group-average
  kernels.
- `src/ssg4wann/mpi/system.py`: MPI setup, task distribution, collection, and
  error handling.
- `test/`: unit tests and physical/file-format consistency checks.
- `examples/`: example inputs and reference outputs.

## Mathematical Conventions

### Group Average

The real-space Hamiltonian is

```text
H_ij(R) = <w_i0 | H | w_jR>.
```

The symmetrized Hamiltonian is

```text
H_symm,ij(R) = (1 / |G|) sum_S <w_i0 | S† H S | w_jR>.
```

Changes to `calc_ent()`, `calc_each()`, operation coefficients, or the
normalization denominator must preserve this definition.

### Coordinate Frames

`findspingroup` spatial rotations and translations are expressed in the direct
lattice frame. Orbital tensors and position vectors use the Cartesian frame.
The current conversion is

```text
R_cart = lattice @ R_lattice @ inverse(lattice)
t_cart = lattice @ t_lattice.
```

`wannob.lat()` returns `permutation` with lattice vectors as columns. Do not
transpose or reorder this matrix without deriving the full transformation and
adding regression tests.

### Orbital Basis Order

The real cubic-harmonic order must stay consistent with Wannier90:

```text
l = 0: [s]
l = 1: [pz, px, py]
l = 2: [dz2, dxz, dyz, dx2-y2, dxy]
l = 3: [fz3, fxz2, fyz2, fz(x2-y2), fxyz,
        fx(x2-3y2), fy(3x2-y2)]
```

`DEFAULT_ORBITALS`, `.win` projections, HR/TB matrix indices, and
`rotation_to_cubic_dmatrix()` must agree.

The Cartesian-tensor implementation is intentional. Do not replace it with an
Euler-angle/Wigner-D implementation without a demonstrated need and tests near
coordinate singularities.

### Spin and Anti-Unitary Operations

An OSSG operation is represented as

```text
S = {R_s || R_l | t},
D(S) = D_l(R_l) tensor D_s(R_s).
```

For the spin rotation:

- `det(R_s) = +1` represents a proper spin rotation.
- `det(R_s) = -1` carries an anti-unitary/time-reversal component.
- The proper rotation component is built from `det(R_s) * R_s`.
- `spin_direction` defines the spin quantization axis and must stay consistent
  with the first-principles calculation, including VASP `SAXIS` conventions.

The matrix-element coefficient order differs between unitary and anti-unitary
operations:

```text
unitary:
    conj(c_i) * H * c_j

anti-unitary:
    c_i * conj(H) * conj(c_j)
```

This distinction in `calc_ent()` and `calc_r_ent()` is not an implementation
detail and must not be collapsed into an incorrect common expression.

### Wannier Centers and Relative Lattice Vectors

For a Wannier function at `R + tau`, the spatial operation gives

```text
R_l(R + tau) + t.
```

The transformed intracell center is reduced modulo one, while the integer cell
shift is obtained with `floor`. The transformed relative lattice vector is

```text
R_new = floor(R_l(R + tau_j) + t)
        - floor(R_l tau_i + t).
```

Use the project tolerance near integer boundaries. Do not replace tolerant
comparisons with exact floating-point equality.

With `extend_LatVec = True`, `LatSet` must include every relative lattice vector
generated by the symmetry action, not only vectors present in the input file.

When a transformed relative lattice vector is absent from the input HR/TB
model, that symmetry operation is intentionally excluded for that matrix
element, and the average is normalized by the number of operations whose
transformed vectors are present (`nsymm - noe`). Do not treat an absent
real-space vector as a zero matrix element, and do not report this conditional
normalization as a bug unless the project convention is explicitly changed.

### Collinear and Noncollinear Indexing

- `NONCOLLINEAR_channel = True` reads one complete spinor Hamiltonian.
- `NONCOLLINEAR_channel = False` reads separate up/down files.
- `chnl = True` uses the internal order `up...up, down...down`.
- The historical interleaved order is represented by `chnl = False`, but the
  current configuration validation rejects using it with collinear input.

When changing index mapping or output splitting, verify both spin blocks.
Never silently drop or introduce spin-mixing matrix elements.

### Wannier90 Degeneracy

The parser expands Wigner-Seitz degeneracy weights as

```text
H_internal(R) = H_file(R) / degeneracy(R).
```

The symmetrized output uses unit degeneracy for every explicit R block. Do not
divide by the weight again during group averaging or restore the original
non-unit weights during output unless the Fourier-interpolation convention has
been fully re-derived and tested.

### Hermiticity

Real-space Hermiticity is

```text
X_ij(R) = conj(X_ji(-R)),
X(R) = X(-R)†.
```

It applies to the Hamiltonian and to each Cartesian component of the TB
position matrix. It does not imply time-reversal symmetry or a real-valued
Hamiltonian.

Small non-Hermitian components in an input HR/TB model may propagate through
the symmetry average. Diagnose the input and transformation before applying
`forced_hermitianize`; the post-processing option must not become a substitute
for correcting an algorithmic error.

### Spin-Only Group Optimization

Without SOC, the spin space group may decompose as

```text
G_SS = G_NS × G_SO.
```

For supported collinear or coplanar systems, the nontrivial spin-only group
imposes a complex-conjugation constraint and the real-space Hamiltonian becomes
real. The code may then average over `G_NS` after applying that constraint.

The current `spinonly_speedup` path is restricted to:

- `soc is False`;
- non-TB mode;
- non-`hard_ave` mode;
- a successful check that `|G| = |G_SO| * |G_NS|`.

When changing this path, compare speedup-on and speedup-off results within a
physically justified tolerance.

### TB Position Matrix

The position operator is a Cartesian vector with an affine transformation under
a space-group operation. `calc_r_ent()` must preserve all of the following:

1. Restore the transformed bra's absolute cell before home-cell reduction.
2. Add the bra cell shift to the relevant home-cell diagonal element.
3. Remove the symmetry translation contribution.
4. Pull vector components back with the inverse spatial rotation.
5. Apply the correct anti-unitary conjugation rule.

Changes in this area require focused tests for identity, rotation, translation,
cell shift, Wannier centers, and real-space Hermiticity.

## MPI Rules

- Rank 0 owns group discovery, file I/O, and final result assembly.
- Broadcast shared group and matrix data explicitly.
- Keep serial and MPI return structures compatible.
- Non-root ranks may receive `None` after result collection; only rank 0 may
  consume the gathered result.
- Task distribution and result order must not change numerical results or
  output ordering.
- Never allow multiple ranks to write the same output file.
- Preserve the large-payload file-buffer fallback unless an equally
  memory-safe replacement is provided.
- Run serial tests first. Add a multi-process smoke test when an MPI
  installation is available and the change affects parallel behavior.

## Git Branch and Release Workflow

The repository uses a long-lived development branch and a stable release
branch:

- `dev` is the integration branch for normal development.
- `main` contains reviewed, release-ready history. Do not commit directly to
  `main`.
- Changes move from `dev` to `main` through a GitHub pull request using
  **Rebase and merge**.
- Release tags are created only from the rebased commit on `main`.

Because GitHub's rebase merge creates new commit hashes on `main`, `dev` must
be realigned with `main` after every merged PR. Leaving the pre-rebase commits
on `dev` will recreate the divergent history that this workflow is intended to
avoid.

### Normal Development on `dev`

Start new work from an up-to-date and clean `dev` branch:

```bash
git switch dev
git pull --ff-only origin dev
git status --short
```

Make focused commits and push them to `dev`:

```bash
git add <changed-files>
git commit -m "fix: describe the change"
git push origin dev
```

Optional short-lived feature branches may target `dev`, but the release PR is
still made from `dev` to `main`. Do not mix unrelated work into a release PR.
If temporary/WIP commits should not remain in the public history, clean them
locally before opening the PR; the GitHub PR itself still uses **Rebase and
merge**, not **Squash and merge** or a merge commit.

If `main` advances independently while `dev` contains unmerged work, first
rebase the clean `dev` branch onto the latest `main`, validate it, and update
the remote with lease protection:

```bash
git fetch origin
git switch dev
git rebase origin/main
git push --force-with-lease origin dev
```

Do not rebase a dirty working tree, and do not use `--force` in place of
`--force-with-lease`.

### Preparing a Release

`pyproject.toml` is the single manually edited source of the package version.
`src/ssg4wann/version.py` reads installed distribution metadata, while
`uv.lock` and the editable `.venv` metadata are synchronized by `uv sync`.
Changing `pyproject.toml` alone is therefore not the complete release step.

Only bump the version when preparing a real release; ordinary development
commits and non-release synchronization do not require a version bump. Make
the version bump the final release-preparation commit on `dev`:

```bash
# Edit project.version in pyproject.toml first.
uv sync
.venv/bin/ssg4wann --version
git status --short
git add pyproject.toml uv.lock
git commit -m "chore(release): bump version to X.Y.Z"
git push origin dev
```

Confirm that the CLI prints `X.Y.Z` and review the lock-file change before
committing it. Do not manually duplicate the version in another Python file.
The corresponding Git tag is still created separately as `vX.Y.Z`.

Before opening or merging the release PR, run the checks appropriate to the
changes, including the complete serial test suite:

```bash
.venv/bin/python -m unittest discover -s test -p 'test_*.py' -v
.venv/bin/ssg4wann --version
.venv/bin/python scripts/check_release_version.py --tag vX.Y.Z
git diff origin/main...dev --check
```

Do not proceed to tagging when a required test fails. The tag-triggered
publish workflow validates the tag, version, and built artifacts, but it does
not replace the complete scientific test suite.

### Pull Request and Rebase Merge

Push `dev`, then create a GitHub pull request with:

```text
base: main
compare: dev
```

Review the complete diff and required checks, then select **Rebase and merge**
from the merge-button menu. If that choice is unavailable, enable **Allow
rebase merging** under the repository's pull-request settings. Do not tag the
pre-rebase commit on `dev`, because its hash is not the commit that lands on
`main`.

### Tagging and Publishing from `main`

After the PR is merged, update local `main` with a fast-forward pull and repeat
the release-version check against the commit that will actually be tagged:

```bash
git fetch origin
git switch main
git pull --ff-only origin main
.venv/bin/python scripts/check_release_version.py --tag vX.Y.Z
git tag -a vX.Y.Z -m "vX.Y.Z"
git push origin vX.Y.Z
```

The annotated tag must point to a commit on `main`. Pushing a `v*` tag triggers
the PyPI publish workflow, so do not create or push a release tag until the
release is intentionally approved and all required checks have passed. Agents
must not push tags or publish a release without explicit user authorization.

### Realigning `dev` After the Rebase Merge

After the PR is merged and `origin/main` has been fetched, make `dev` point to
the new rebased `main` commit:

```bash
git switch dev
git status --short
git reset --hard origin/main
git push --force-with-lease origin dev
```

Before the reset, require all of the following:

- the pull request has been merged successfully;
- `origin/main` contains the expected commits;
- the working tree is clean;
- all wanted `dev` work has been pushed or otherwise backed up.

`git reset --hard` discards uncommitted changes. Coding agents must follow the
repository's destructive-action rules and obtain explicit authorization before
performing the reset or force-updating the remote branch. Once complete,
`main`, `dev`, `origin/main`, and `origin/dev` should resolve to the same commit
until new development begins.

### Backup and Temporary Integration Branches

Backup branches are safety snapshots for exceptional history rewrites, not
part of the normal release cycle. Do not create a backup for every release,
and do not continue development on a backup branch.

The `backup/dev-pre-v1.0.2` branch preserves the pre-cleanup `dev` history. It
may be used to inspect old commits, compare an old tree, restore an individual
file, or cherry-pick a specifically identified missing change:

```bash
git log --oneline backup/dev-pre-v1.0.2
git diff backup/dev-pre-v1.0.2..dev
git restore --source=backup/dev-pre-v1.0.2 -- <path>
git cherry-pick <verified-missing-commit>
```

Never merge or cherry-pick the backup branch wholesale: much of that history
already exists on `main` under different commit hashes. Keep the backup frozen
until the cleaned history and subsequent releases are trusted, then delete it
only after explicit review.

`integration/v1.0.2` was a one-time migration branch used to move the real new
changes onto the clean `main` history. It is not required for future releases.
Once `git range-diff` or an equivalent review confirms that its commits landed
on `main`, both its local and remote references may be deleted. Normal future
releases use the direct `dev` to `main` pull-request workflow above and do not
need a release or integration branch.

## Development and Validation

Use Python 3.12 or newer. Prefer the repository environment and lock file; do
not update dependencies or `uv.lock` unless the task requires it.

```bash
# Full unit-test suite
.venv/bin/python -m unittest discover -s test -p 'test_*.py' -v

# One test module
.venv/bin/python -m unittest test.test_tb -v

# CLI help
.venv/bin/ssg4wann --help

# Generate a configuration
.venv/bin/ssg4wann --init -w examples/Fe
```

Do not run examples in place when they can overwrite reference outputs. Copy
the required inputs to a temporary directory for end-to-end tests.

### Minimum Checks by Change Type

For configuration or CLI changes, verify:

- boolean semantics and the OSSG/MSG selection;
- collinear/noncollinear and HR/TB file discovery;
- paths relative to the configuration directory;
- `--init` does not overwrite an existing configuration;
- new options are reflected in the data model, parser, template, README, and
  tests.

For HR/TB parser or writer changes, verify:

- degeneracy expansion;
- read/write/read round trips;
- R-vector sets, dimensions, and basis indices;
- up/down output splitting;
- fixed-precision values remain unambiguously separated;
- real-space Hermiticity;
- consistency between the TB Hamiltonian block and HR output.

For orbital or spin representation changes, test:

- identity and simple 90/180-degree rotations;
- inversion or improper spin rotations;
- anti-unitary/time-reversal operations;
- a non-z `spin_direction`;
- representation norm, orthogonality, or unitarity;
- successful mapping back into the Wannier basis.

For group-average changes, verify:

- an identity-only operation leaves the input unchanged;
- per-operation and direct group-average paths agree;
- missing R handling cannot divide by zero;
- `extend_LatVec` produces the required symmetry-related vectors;
- anti-unitary conjugation is correct;
- task ordering does not affect the result;
- relevant high-symmetry band degeneracies are restored.

For TB position-matrix changes, verify:

- identity invariance;
- Cartesian rotation behavior;
- cancellation of symmetry translation and bra cell shift;
- physically consistent home-cell diagonal Wannier centers;
- Hermiticity of every Cartesian component. (It is not necessarily, because the Hamiltonian output by Wannier90 may be non-Hermitian by a small numerical amount.)

Report the exact tests run, their results, and any checks not run. Do not hide
failures or relabel them as passing. Benchmark tolerances must come from tests,
the paper, or a documented reference calculation rather than guesswork.

## Coding and Workspace Discipline

- Keep Python 3.12+ compatibility.
- Reuse project tolerance constants and domain-specific exceptions.
- Prefer names that expose the physical meaning of a matrix or index.
- Do not catch broad exceptions in ways that conceal numerical or physical
  failures.
- Do not edit unrelated files or overwrite uncommitted user changes.
- Do not reformat large HR/TB reference files unless explicitly required.
- Do not commit generated POSCAR files, logs, MPI buffers, or temporary output.
- Preserve compatibility with Wannier90 and downstream tools such as
  WannierTools when changing output formats.
- If documentation and implementation disagree, establish the correct behavior
  from the implementation, paper, and tests, then update all affected
  documentation consistently.

## Completion Checklist

Before declaring a task complete:

- [ ] Read the algorithm and conventions relevant to the changed code.
- [ ] Preserve the SOC/OSSG/MSG, coordinate, orbital, and spin-index semantics.
- [ ] Preserve the correct anti-unitary conjugation order.
- [ ] Apply degeneracy normalization exactly once.
- [ ] Add or update the smallest meaningful regression tests.
- [ ] Run tests proportional to the risk of the change.
- [ ] Keep example inputs and unrelated user changes untouched.
- [ ] Report test results, limitations, and unresolved issues accurately.
