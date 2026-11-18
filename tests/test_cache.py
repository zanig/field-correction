"""Cache compatibility depends on arrays, not how the archive was written."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

from data import check_preprocessing
from tests.helpers import fixture
from utils import cache_content_sha256, sha256


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.arrays = {
            "x": np.arange(12, dtype=np.float32).reshape(3, 4),
            "names": np.asarray(["hill", "duct"]),
            "preprocessing_version": np.asarray(2, dtype=np.int32),
        }
        self.original = self.root / "original.npz"
        np.savez_compressed(self.original, **self.arrays)

    def test_equivalent_arrays_survive_repackaging_and_byte_order_changes(self):
        repacked = self.root / "repacked.npz"
        arrays = dict(reversed(list(self.arrays.items())))
        arrays["x"] = np.asfortranarray(arrays["x"].astype(">f4"))
        np.savez(repacked, **arrays)
        self.assertNotEqual(sha256(self.original), sha256(repacked))
        self.assertEqual(cache_content_sha256(self.original), cache_content_sha256(repacked))
        checkpoint = {"preprocessing_version": 2,
                      "cache_sha256": sha256(self.original),
                      "cache_content_sha256": cache_content_sha256(self.original)}
        bank = fixture()
        bank.preprocessing_version = 2
        check_preprocessing(checkpoint, bank, repacked)

    def test_changed_values_shapes_dtypes_and_names_are_rejected(self):
        checkpoint = {"preprocessing_version": 2,
                      "cache_content_sha256": cache_content_sha256(self.original)}
        bank = fixture()
        bank.preprocessing_version = 2
        for change in ("value", "shape", "dtype", "name"):
            with self.subTest(change=change):
                arrays = {key: value.copy() for key, value in self.arrays.items()}
                if change == "value":
                    arrays["x"][0, 0] += 1
                elif change == "shape":
                    arrays["x"] = arrays["x"].reshape(4, 3)
                elif change == "dtype":
                    arrays["x"] = arrays["x"].view(np.uint32)
                else:
                    arrays["renamed"] = arrays.pop("x")
                changed = self.root / f"{change}.npz"
                np.savez_compressed(changed, **arrays)
                with self.assertRaisesRegex(ValueError, "array contents"):
                    check_preprocessing(checkpoint, bank, changed)


if __name__ == "__main__":
    unittest.main()
