"""
setup.py — Packaging configuration for the MIRAGE-DoH reproducibility artifact.

Makes ``import mirage`` work after ``pip install -e .``, and registers the
five CLI tools (mirage-cma, mirage-mcshard, mirage-cce, mirage-tmn,
mirage-sgi) as console scripts.

Usage (from the repository root, after all source files are present):

    pip install -e .              # editable install (recommended)
    pip install -e .[dev]         # plus testing stack (pytest, pytest-cov)
    pip install -e .[gpu]         # plus CUDA-enabled torch

Then:

    python -c "import mirage; print(mirage.__version__)"
    python src/cli_tools/cma_cli.py --evolve --threat=GAN_Flood   # README §8
    mirage-cma --evolve --threat=GAN_Flood                         # equivalent

The two invocation styles are equivalent because the CLI modules are
importable as ``mirage.cli_tools.<name>`` and each exposes a ``main()``.

NOTE ON INCREMENTAL DELIVERY
    This repository is delivered file-by-file. The PACKAGES list below
    references the source tree in ``src/``. Until those files are all
    uploaded (Files 11-16 in the delivery order), `pip install -e .` will
    fail with "packages not found". That is expected. Users clone the
    completed repository, not an intermediate commit. Once all source files
    are present, `pip install -e .` succeeds and `import mirage` works.
"""

from pathlib import Path

from setuptools import setup


# ---------------------------------------------------------------------------
# Resolve the repository root regardless of the working directory from which
# `pip install -e .` is invoked. Standard idiom for setup.py that reads files
# relative to its own location.
# ---------------------------------------------------------------------------
HERE = Path(__file__).parent.resolve()


# ---------------------------------------------------------------------------
# Long description (the README). Attached verbatim so that any downstream
# consumer (PyPI, a package index, or a reviewer reading `pip show
# mirage-doh`) sees the same content the reviewer sees on GitHub.
# ---------------------------------------------------------------------------
README_PATH = HERE / "README.md"
if not README_PATH.is_file():
    raise FileNotFoundError(
        f"README.md not found at {README_PATH}. "
        "Run `pip install -e .` from the repository root, not from a "
        "subdirectory. If you cloned the repository shallowly and the file "
        "is missing, re-clone with `git clone --depth 1 <url>`."
    )
LONG_DESCRIPTION = README_PATH.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Runtime dependencies.
#
# These MUST stay in sync with requirements.txt and environment.yml. They are
# duplicated here because setup.py is what `pip install -e .` actually reads
# for an editable install, while requirements.txt is read by
# `pip install -r requirements.txt`. The two files must always be updated
# together; tests/test_reproduce_tables.py::test_requirements_match_setup
# enforces this in CI.
#
# Pinned to the exact versions used to produce the numbers in Chapter 3 of
# the thesis. See requirements.txt for the rationale for each package.
# ---------------------------------------------------------------------------
INSTALL_REQUIRES = [
    # 1. Core scientific stack
    "numpy==1.26.4",
    "scipy==1.13.1",
    "pandas==2.2.2",

    # 2. Deep learning (CMA, TMN, SGI, CNN-LSTM baseline)
    "torch==2.3.1",
    "torchvision==0.18.1",
    "torchaudio==2.3.1",

    # 3. Classical ML (Random Forest baseline)
    "scikit-learn==1.5.0",
    "joblib==1.4.2",

    # 4. Topological data analysis (TMN, Eq. 6-10)
    "gudhi==3.10.0",
    "ripser==0.6.12",
    "persim==0.3.7",

    # 5. Neuromorphic simulation (Loihi 2 CPU simulator)
    "snntorch==0.9.1",
    "brian2==2.7.1",

    # 6. Graph analysis (SGI, Eq. 17-21)
    "networkx==3.3",
    "python-igraph==0.11.5",

    # 7. Cryptographic references (MCS, Eq. 11-13)
    "pycryptodome==3.20.0",

    # 8. Visualisation (Figures 1-6)
    "matplotlib==3.9.0",
    "seaborn==0.13.2",
    "graphviz==0.20.3",

    # 9. Configuration (the eight configs/*.yaml files)
    "PyYAML==6.0.1",

    # 10. Data handling and parallel preprocessing
    "requests==2.32.3",
    "dask==2024.6.0",
    "dask[dataframe]==2024.6.0",

    # 11. Utilities
    "tqdm==4.66.4",
    "packaging==24.1",
    "psutil==6.0.0",
]


# ---------------------------------------------------------------------------
# Package layout.
#
# The source tree lives in ``src/`` and is installed as the top-level package
# ``mirage``. Mapping:
#
#   src/__init__.py               -> mirage/__init__.py
#   src/utils/                    -> mirage/utils/          (Equation 1-31 helpers)
#   src/models/                   -> mirage/models/         (CMA, TMN, MCS, CCE, SGI)
#   src/baselines/                -> mirage/baselines/      (CNN-LSTM, Random Forest)
#   src/attacks/                  -> mirage/attacks/        (GAN flood, timing, downgrade, tunnel, DGA)
#   src/cli_tools/                -> mirage/cli_tools/      (Tables 6 and 7 commands)
#   src/hardware_simulation/      -> mirage.hardware_simulation/ (Loihi 2, A100)
#
# The mapping is done by ``package_dir={"mirage": "src"}`` below. Every
# directory listed in PACKAGES MUST contain an __init__.py (uploaded as part
# of Files 11-16). Do NOT use find_packages(where="src") here, because the
# top-level src/ is not itself a package; it becomes the body of mirage via
# package_dir.
# ---------------------------------------------------------------------------
PACKAGES = [
    "mirage",
    "mirage.utils",
    "mirage.models",
    "mirage.baselines",
    "mirage.attacks",
    "mirage.cli_tools",
    "mirage.hardware_simulation",
]


# ---------------------------------------------------------------------------
# Console-script entry points.
#
# Each CLI tool in src/cli_tools/ exposes a callable named main() that parses
# argv via argparse and returns an int exit code (0 on success). The entry
# points below let a reviewer invoke the tools by name after `pip install -e .`:
#
#   mirage-cma --evolve --threat=GAN_Flood        # == Table 6 row 1
#   mirage-mcshard --morph --level=L2              # == Table 6 row 2
#   mirage-cce --replay=ECS_Attack --update_protocol  # == Table 6 row 3
#   mirage-tmn --simplicial=50 --threshold=0.7    # == Table 6 row 4
#   mirage-sgi --spectrum=low --band=0.1-0.3Hz    # == Table 6 row 5
#
# The README documents the equivalent `python src/cli_tools/<name>.py ...`
# form. Both work; the entry points are a convenience.
# ---------------------------------------------------------------------------
ENTRY_POINTS = {
    "console_scripts": [
        "mirage-cma=mirage.cli_tools.cma_cli:main",
        "mirage-mcshard=mirage.cli_tools.mcshard_cli:main",
        "mirage-cce=mirage.cli_tools.cce_cli:main",
        "mirage-tmn=mirage.cli_tools.tmn_cli:main",
        "mirage-sgi=mirage.cli_tools.sgi_cli:main",
    ],
}


# ---------------------------------------------------------------------------
# setup() — the actual packaging metadata.
# ---------------------------------------------------------------------------
setup(
    # PyPI / distribution name. Hyphenated per PEP 503 normalization.
    # Distinct from the import name `mirage`, which is what `import mirage`
    # resolves to. This is the standard convention (e.g., the package
    # `scikit-learn` imports as `sklearn`, `pillow` imports as `PIL`).
    name="mirage-doh",
    version="1.0.0",

    # Short summary (also shown on any package index that reads setup.py).
    description=(
        "Reproducibility artifact for MIRAGE-DoH: Metamorphic Intelligence "
        "and Resilient AI Grid for Autonomous Governance of Encrypted DNS"
    ),

    # Long description: the README, attached verbatim.
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",

    # Matches the LICENSE file at the repo root (File 3, MIT).
    license="MIT",

    # Matches the copyright holder in LICENSE (File 3), consistent with the
    # blind-review redactions in the thesis PDF.
    author="The MIRAGE-DoH Thesis Author(s)",

    # Package discovery: explicit list, mapped to src/ via package_dir.
    # See the PACKAGES comment above for the rationale.
    packages=PACKAGES,
    package_dir={"mirage": "src"},

    # Include non-Python files declared in package_data below.
    include_package_data=True,

    # Non-Python files that must ship inside the wheel so that a reviewer
    # who installs from a wheel (not from git) can still read them.
    # The toy dataset and expected_outputs live in data/ and expected_outputs/
    # at the repo root, not inside src/, so they are not package data.
    package_data={
        "mirage.hardware_simulation": [
            "README_hardware.md",
        ],
    },

    # Runtime dependencies (pinned). See INSTALL_REQUIRES above.
    install_requires=INSTALL_REQUIRES,

    # Optional extras.
    #
    #   pip install -e .[dev]   -> adds pytest, pytest-cov (the test stack)
    #   pip install -e .[gpu]   -> adds CUDA-enabled torch/torchvision
    #
    # The [gpu] extra requires the PyTorch extra index URL because the
    # +cu118 wheels are not on PyPI. Install with:
    #
    #   pip install --extra-index-url https://download.pytorch.org/whl/cu118 \
    #               -e .[gpu]
    #
    # For CUDA 12.1, replace cu118 with cu121.
    extras_require={
        "dev": [
            "pytest==8.2.2",
            "pytest-cov==5.0.0",
        ],
        "gpu": [
            "torch==2.3.1+cu118",
            "torchvision==0.18.1+cu118",
        ],
    },

    # Console scripts (mirage-cma, mirage-mcshard, ...). See ENTRY_POINTS.
    entry_points=ENTRY_POINTS,

    # Python version constraint: 3.9-3.11 tested; 3.12 not yet tested because
    # pycryptodome has not stabilised on it. Matches requirements.txt §header
    # and environment.yml §1.
    python_requires=">=3.9,<3.12",

    # Do not create a .egg file; install as a directory. Required for
    # entry-point discovery in some pip versions, and cleaner for
    # editable installs (`pip install -e .`).
    zip_safe=False,

    # Classifiers: standard vocabulary. Used by PyPI, any package index, and
    # by some dependency-resolution tools to filter by Development Status,
    # License, and Python version.
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Security :: Cryptography",
        "Topic :: Internet :: Name Service (DNS)",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "Environment :: Console",
    ],

    # Project URLs: rendered in the PyPI sidebar (and in any tool that reads
    # package metadata). Puts the three most useful links one click away:
    # the source repo, the associated paper, and the checklist.
    project_urls={
        "Reproducibility repository": (
            "https://github.com/dezignator/MIRAGE-DoH-Reproducibility"
        ),
        "Associated paper (IEEE TNSM 2026)": (
            "https://doi.org/10.1109/TNSM.2026.3677474"
        ),
        "Reproducibility checklist": (
            "https://github.com/dezignator/MIRAGE-DoH-Reproducibility/"
            "blob/main/REPRODUCIBILITY_CHECKLIST.md"
        ),
        "Issue tracker": (
            "https://github.com/dezignator/MIRAGE-DoH-Reproducibility/issues"
        ),
    },
)
