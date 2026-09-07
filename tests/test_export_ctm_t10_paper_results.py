"""Small standard-library fixture tests for the paper-result exporter."""

import contextlib
import hashlib
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

from export_ctm_t10_paper_results import BRIDGE_FILES, MODEL_TAG, T10_TAG, export_results


class ExportResultsTests(unittest.TestCase):
    def fixture(self, project):
        def write(path, value):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value) if isinstance(value, dict) else value, encoding="utf-8")

        ctm = project / "results/journal/ctm"
        write(ctm / "ctm_full_completed.json", {
            "completed": True, "stage": "full", "max_eval_samples": 0,
            "all_id_train_samples": True, "independent_runs": 11,
        })
        write(ctm / "ctm_full_all_runs.csv", "Method,AUROC\nctm,85\n")
        for seed in range(11):
            folder = ctm / "full" / f"fixture_run{seed}"
            write(folder / "ctm.csv", "Dataset,AUROC\nnearood,85\n")
            write(folder / "ctm.json", {
                "stage": "full", "method": "ctm", "max_eval_samples": 0,
                "all_id_train_samples": True,
            })
        for seed in range(3):
            folder = project / "results_openood/imagenet1k" / MODEL_TAG / f"seed{seed}" / T10_TAG
            write(folder / "openood_metrics_all_potentials.csv", "Potential,AUROC\ngaussian,65\nimq,60\n")
            write(folder / "run_config.json", {
                "seed": seed, "n_steps": 10, "max_eval_samples": 0,
                "mass_mode": "uniform", "mass_normalization": "none",
                "bandwidth_loss": "static", "potentials": ["gaussian", "imq"],
            })
            write(folder / "detector.pt", "must not export")
        bridge = project / "results/journal/static_reduction_bridge"
        write(bridge / "static_reduction_bridge_full_completed.json", {
            "completed": True, "stage": "full", "near_far_test_access": False,
            "seeds": [0, 1, 2],
        })
        for name in BRIDGE_FILES:
            write(bridge / "full" / name, "Seed,Value\n0,1\n")
        write(ctm / "smoke/ctm.csv", "Dataset,AUROC\nnearood,90\n")

    def test_export_excludes_weights_and_smoke_and_preserves_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            self.fixture(project)
            before = {str(p.relative_to(project)): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in project.rglob("*") if p.is_file()}
            with contextlib.redirect_stdout(io.StringIO()):
                first = export_results(project)
                second = export_results(project)
            self.assertNotEqual(first, second)
            self.assertTrue(first.is_file())
            with tarfile.open(first) as bundle:
                names = bundle.getnames()
                self.assertEqual(len(names), 40)
                self.assertFalse(any("smoke" in name or name.endswith(".pt") for name in names))
                manifest = json.load(bundle.extractfile("paper_results_export_manifest.json"))
                for record in manifest["files"]:
                    self.assertEqual(hashlib.sha256(bundle.extractfile(record["path"]).read()).hexdigest(), record["sha256"])
            for name, digest in before.items():
                self.assertEqual(hashlib.sha256((project / name).read_bytes()).hexdigest(), digest)

    def test_missing_t10_config_creates_no_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            self.fixture(project)
            missing = project / "results_openood/imagenet1k" / MODEL_TAG / "seed2" / T10_TAG / "run_config.json"
            missing.unlink()
            with self.assertRaisesRegex(ValueError, "Missing required input"):
                export_results(project)
            self.assertFalse((project / "archives").exists())

    def test_capped_config_creates_no_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            self.fixture(project)
            config_path = project / "results_openood/imagenet1k" / MODEL_TAG / "seed0" / T10_TAG / "run_config.json"
            config = json.loads(config_path.read_text())
            config["max_eval_samples"] = 128
            config_path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "does not match"):
                export_results(project)
            self.assertFalse((project / "archives").exists())


if __name__ == "__main__":
    unittest.main()
