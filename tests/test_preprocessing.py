"""Regression tests for ASCII field parsing and wall-surface distances."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

from prep import read_field, read_cell_field, read_faces, wall_distance


class PreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def write(self, name, text):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return path

    def test_uniform_internal_ignores_nonuniform_boundary(self):
        path = self.write("U", """internalField uniform (1 2 3);
boundaryField { inlet { value nonuniform List<vector> 2 ((4 5 6) (7 8 9)); } }
""")
        np.testing.assert_array_equal(read_field(path), [[1, 2, 3]])
        np.testing.assert_array_equal(read_cell_field(path, 3, 3), [[1, 2, 3]] * 3)

    def test_nonuniform_and_comments(self):
        path = self.write("U", """/* 123 (9 9 9) */
FoamFile { format ascii; class volVectorField; }
internalField nonuniform List<vector> 2 ((1e-3 -2 3) // comment
(4 5 6)); boundaryField { x { value uniform (9 9 9); } }
""")
        np.testing.assert_allclose(read_cell_field(path, 2, 3), [[.001, -2, 3], [4, 5, 6]])

    def test_named_and_included_field(self):
        self.write("interpolated/U_internalField", "U_internalField nonuniform List<vector> 2 ((1 2 3)(4 5 6));")
        path = self.write("U_LES", '#include "interpolated/U_internalField"\ninternalField $U_internalField;')
        np.testing.assert_array_equal(read_cell_field(path, 2, 3), [[1, 2, 3], [4, 5, 6]])

    def test_counts_components_binary_and_nonfinite_rejected(self):
        for text in ["internalField nonuniform List<scalar> 3 (1 2);",
                     "internalField nonuniform List<vector> 1 ((1 2));",
                     "FoamFile { format binary; } 1 (1)",
                     "internalField uniform nan;", "internalField uniform $missing;"]:
            with self.subTest(text=text), self.assertRaises(ValueError):
                read_field(self.write("bad", text))
        path = self.write("short", "internalField nonuniform List<vector> 1 ((1 2 3));")
        with self.assertRaises(ValueError):
            read_cell_field(path, 20, 3)

    def test_standalone_mesh_and_face_counts(self):
        np.testing.assert_array_equal(read_field(self.write("points", "2 ((1 2 3) (4 5 6))")), [[1, 2, 3], [4, 5, 6]])
        with self.assertRaises(ValueError):
            read_faces(self.write("faces", "2 (3(0 1 2))"))
        with self.assertRaises(ValueError):
            read_faces(self.write("faces", "1 (4(0 1 2))"))

    def mesh(self, points, faces, patch_type="wall"):
        self.write("constant/polyMesh/points", f"{len(points)}\n(\n" + "\n".join("(" + " ".join(map(str, p)) + ")" for p in points) + "\n)")
        self.write("constant/polyMesh/faces", f"{len(faces)}\n(\n" + "\n".join(str(len(f)) + "(" + " ".join(map(str, f)) + ")" for f in faces) + "\n)")
        self.write("constant/polyMesh/boundary", f"1 (wallNamedPatch {{ type {patch_type}; nFaces {len(faces)}; startFace 0; }})")

    def test_distance_to_face_interior_edges_and_vertices(self):
        self.mesh([(0, 0, 0), (2, 0, 0), (2, 0, 2), (0, 0, 2)], [[0, 1, 2, 3]])
        query = np.array([[.1, .01, .1], [3, 1, 1], [3, 1, 3], [1, 0, 1]], dtype=float)
        np.testing.assert_allclose(wall_distance(self.root, query).ravel(), [.01, np.sqrt(2), np.sqrt(3), 0], rtol=1e-6)

    def test_concave_polygon_does_not_fill_notch(self):
        # L-shaped wall in the xy plane; (1.5,1.5,0) lies in its missing corner.
        self.mesh([(0,0,0),(2,0,0),(2,1,0),(1,1,0),(1,2,0),(0,2,0)], [[0,1,2,3,4,5]])
        query = np.array([[1.5, 1.5, .1], [.5, 1.5, .1]])
        np.testing.assert_allclose(wall_distance(self.root, query).ravel(), [np.sqrt(.26), .1], rtol=1e-6)

    def test_patch_name_does_not_make_a_wall(self):
        self.mesh([(0,0,0),(1,0,0),(0,1,0)], [[0,1,2]], patch_type="patch")
        with self.assertRaisesRegex(ValueError, "no faces"):
            wall_distance(self.root, np.array([[0,0,1.]]))


if __name__ == "__main__":
    unittest.main()
