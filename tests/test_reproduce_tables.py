"""
test_reproduce_tables.py — End-to-end tests for table reproduction.

Verifies that the generation scripts create valid JSON files and that the
verification script can parse them.

Run with:
    pytest tests/test_reproduce_tables.py -v
"""

import json
import subprocess
import sys
from pathlib import Path
import pytest

# Adjust path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPECTED_DIR = REPO_ROOT / "expected_outputs"
TABLES_DIR = REPO_ROOT / "outputs" / "tables"


@pytest.fixture(autouse=True)
def setup_output_dir():
    """Ensure the output directories exist."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    yield
    # Do not clean up; leave files for inspection.


class TestTableReproduction:
    """Test the scripts in scripts/reproduce_tables/."""

    def test_table_5_generation(self):
        """Table 5: Shard configurations (no data required)."""
        script = REPO_ROOT / "scripts" / "reproduce_tables" / "table_5_mcs_configs.py"
        result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
        assert result.returncode == 0, f"Script failed: {result.stderr}"

        out_file = TABLES_DIR / "table_5_mcs_configs.json"
        assert out_file.exists(), "Table 5 JSON not created"

        data = json.loads(out_file.read_text())
        assert "pi_0" in data
        assert data["pi_0"]["nist_category"] == 3
        assert data["pi_1"]["nist_category"] == 5
        assert data["pi_2"]["nist_category"] == 5

    def test_table_8_generation(self):
        """Table 8: Core metrics (uses static thesis numbers)."""
        script = REPO_ROOT / "scripts" / "reproduce_tables" / "table_8_core_metrics.py"
        result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
        assert result.returncode == 0, f"Script failed: {result.stderr}"

        out_file = TABLES_DIR / "table_8_core_metrics.json"
        assert out_file.exists(), "Table 8 JSON not created"

        data = json.loads(out_file.read_text())
        assert "rows" in data
        # Verify the first row matches the thesis.
        row_0 = data["rows"][0]
        assert row_0["metric"] == "Detection Accuracy (%)"
        assert row_0["mirage"] == "99.98"


class TestVerificationScript:
    """Test the expected_outputs/verification_script.py."""

    def test_verifier_passes_on_table_5(self):
        """The verifier should pass Table 5 because it just reads config."""
        # First ensure Table 5 is generated.
        script_5 = REPO_ROOT / "scripts" / "reproduce_tables" / "table_5_mcs_configs.py"
        subprocess.run([sys.executable, str(script_5)], check=True)

        verifier = EXPECTED_DIR / "verification_script.py"
        result = subprocess.run(
            [sys.executable, str(verifier), "--rtol", "0.01"],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
        )
        # The verifier should succeed if the actual matches the expected.
        assert result.returncode == 0, f"Verifier failed: {result.stdout}\n{result.stderr}"
        assert "PASS: table_5_expected.json" in result.stdout
