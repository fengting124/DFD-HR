import csv
import sys
import tempfile
import unittest
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / 'scripts'
sys.path.insert(0, str(SCRIPTS_ROOT))

from experiment_lifecycle import (  # noqa: E402
    REGISTRY_FIELDS,
    assert_resolved,
    freeze_config,
    validate_public_text,
    validate_run_id,
    verify_checksums,
    write_checksums,
)


class ExperimentLifecycleTests(unittest.TestCase):
    def test_run_id_contract(self):
        run_id = 'dfdhr_ffppc23_mini-amp_20260720_001'
        self.assertEqual(validate_run_id(run_id), run_id)
        for invalid in ('run-1', 'dfdhr_ffpp_20260720_001', 'dfdhr_ffpp_x_2026_1'):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    validate_run_id(invalid)

    def test_public_registry_text_rejects_private_infrastructure(self):
        self.assertEqual(validate_public_text('controller role', 'node'), 'controller role')
        for private_text in ('/home/example/run', '/scratch/data', '10.0.0.1', 'ssh endpoint'):
            with self.subTest(private_text=private_text):
                with self.assertRaises(ValueError):
                    validate_public_text(private_text, 'field')

    def test_manifest_placeholders_are_rejected_recursively(self):
        assert_resolved({'value': ['ready', None]})
        with self.assertRaisesRegex(ValueError, 'manifest.value'):
            assert_resolved({'value': 'REPLACE_ME'})

    def test_config_freeze_merges_base_and_redirects_outputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            detector = root / 'detector.yaml'
            base = root / 'base.yaml'
            destination = root / 'run/config.resolved.yaml'
            detector.write_text('model_name: dfd_hr\nsave_feat: true\n', encoding='utf-8')
            base.write_text('label_dict:\n  real: 0\n', encoding='utf-8')

            config, digest = freeze_config(detector, base, destination, root / 'run')

            self.assertEqual(len(digest), 64)
            self.assertEqual(config['label_dict'], {'real': 0})
            self.assertFalse(config['save_feat'])
            self.assertTrue(config['save_ckpt'])
            self.assertEqual(Path(config['log_dir']), (root / 'run/training').resolve())
            self.assertEqual(yaml.safe_load(destination.read_text()), config)

    def test_checksums_detect_file_changes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            artifact = run_dir / 'artifact.txt'
            artifact.write_text('first\n', encoding='utf-8')
            write_checksums(run_dir)
            self.assertTrue(verify_checksums(run_dir))
            artifact.write_text('changed\n', encoding='utf-8')
            self.assertFalse(verify_checksums(run_dir))

    def test_registry_header_matches_lifecycle_schema(self):
        with (PROJECT_ROOT / 'registry/experiments.csv').open(newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            self.assertEqual(tuple(next(reader)), REGISTRY_FIELDS)


if __name__ == '__main__':
    unittest.main()
