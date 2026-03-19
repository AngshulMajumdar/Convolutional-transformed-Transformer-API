from __future__ import annotations

from pathlib import Path
import json
import subprocess
import sys


def main() -> None:
    outdir = Path("results/minislm_encdec_demo")
    subprocess.run([
        sys.executable,
        "scripts/compare_minislm_encdec_standard_vs_analysis_synthesis.py",
        "--seeds", "2",
        "--epochs", "4",
        "--n-samples", "32",
        "--outdir", str(outdir),
    ], check=True)
    with (outdir / "summary.json").open() as f:
        summary = json.load(f)
    print("miniSLM encoder-decoder demo summary")
    for row in summary:
        print(
            f"{row['model_type']}: loss={row['mean_final_loss']:.4f}, "
            f"best_token_acc={row['mean_best_token_acc']:.4f}, elapsed_ms={row['mean_elapsed_ms']:.1f}"
        )


if __name__ == "__main__":
    main()
