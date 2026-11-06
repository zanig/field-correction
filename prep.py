"""Build cache.npz from baseline OpenFOAM fields and reference LES/DNS data."""
import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from data import FEATURES, TRAIN_CASES, TEST_CASES


@dataclass(frozen=True)
class Case:
    name: str
    coords: np.ndarray
    features: np.ndarray
    u_rans: np.ndarray
    u_les: np.ndarray | None


def ascii_text(path: Path) -> str:
    """Read the supported ASCII subset; never interpret binary bytes as values."""
    text = path.read_bytes().decode("utf-8", errors="replace")
    text = re.sub(r"/\*.*?\*/|//[^\n]*", "", text, flags=re.DOTALL)
    if re.search(r"\bformat\s+binary\s*;", text):
        raise ValueError(f"{path}: binary OpenFOAM files are not supported; export ASCII")
    return text


def list_payload(text: str, path: Path) -> tuple[int, str]:
    """Read a counted list starting here, stopping before any boundary data."""
    match = re.match(r"\s*(\d+)\s*\(", text)
    if not match:
        raise ValueError(f"{path}: expected a counted OpenFOAM list")
    depth = 1
    for token in re.finditer(r"[()]", text[match.end():]):
        depth += 1 if token.group() == "(" else -1
        if depth == 0:
            return int(match.group(1)), text[match.end():match.end() + token.start()]
    raise ValueError(f"{path}: unterminated OpenFOAM list")


def read_field(path: Path, *, dtype=np.float32, expected_rows=None, _seen=()) -> np.ndarray:
    """Parse ASCII internal fields, named include fields and standalone lists.

    Uniform fields return one row; read_cell_field expands them using mesh size.
    Unsupported syntax and malformed counts fail instead of reading boundaries.
    """
    path = Path(path)
    if path.resolve() in _seen:
        raise ValueError(f"{path}: cyclic field include")
    text = ascii_text(path)
    body = re.sub(r"\bFoamFile\s*\{[^{}]*\}", "", text, count=1)
    internal = re.search(r"\binternalField\s+", body)
    if internal:
        value = body[internal.end():]
    elif re.search(r"\bclass\s+vol\w*Field\s*;", text):
        raise ValueError(f"{path}: missing internalField")
    else:
        named = re.match(r"\s*[A-Za-z_]\w*\s+(?=uniform\b|nonuniform\b)", body)
        value = body[named.end():] if named else body

    variable = re.match(r"\s*\$([A-Za-z_]\w*)\s*;", value)
    if variable:
        for include in re.findall(r'#include\s+"([^"]+)"', body):
            if Path(include).name == variable.group(1):
                return read_field(path.parent / include, dtype=dtype, expected_rows=expected_rows,
                                  _seen=(*_seen, path.resolve()))
        raise ValueError(f"{path}: cannot resolve internal field ${variable.group(1)}")

    uniform = re.match(r"\s*uniform\s+(\([^()]*\)|[^\s;]+)\s*;", value)
    components = None
    if uniform:
        tokens = [uniform.group(1)]
        count = 1
    else:
        nonuniform = re.match(r"\s*nonuniform\s+(?:List<\s*(\w+)\s*>|(scalar|vector|label|tensor|symmTensor|sphericalTensor)Field)\s*", value)
        if nonuniform:
            kind = nonuniform.group(1) or nonuniform.group(2)
            components = {"scalar": 1, "label": 1, "vector": 3, "tensor": 9,
                          "symmTensor": 6, "sphericalTensor": 1}.get(kind)
            if components is None:
                raise ValueError(f"{path}: unsupported list type {kind}")
            value = value[nonuniform.end():]
        count, payload = list_payload(value, path)
        tokens = re.findall(r"\([^()]*\)|[^\s()]+", payload)
    try:
        rows = [[float(v) for v in token.strip("()").split()] for token in tokens]
        result = np.asarray(rows, dtype=dtype)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{path}: malformed numeric field") from exc
    if count == 0:
        result = np.empty((0, components or 1), dtype=dtype)
    if len(tokens) != count or result.ndim != 2 or (components and result.shape[1] != components):
        raise ValueError(f"{path}: declared {count} rows, parsed shape {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError(f"{path}: field contains nonfinite values")
    if expected_rows is not None:
        if uniform:
            result = np.repeat(result, expected_rows, axis=0)
        elif len(result) != expected_rows:
            raise ValueError(f"{path}: expected {expected_rows} cells, got {len(result)}")
    return result


def read_cell_field(path: Path, n: int, components: int) -> np.ndarray:
    values = read_field(path, expected_rows=n)
    if values.shape != (n, components):
        raise ValueError(f"{path}: expected {(n, components)}, got {values.shape}")
    return values


def read_faces(path: Path) -> list[np.ndarray]:
    text = re.sub(r"\bFoamFile\s*\{[^{}]*\}", "", ascii_text(path), count=1)
    count, payload = list_payload(text, path)
    faces = []
    end = 0
    for match in re.finditer(r"(\d+)\s*\(([^()]*)\)", payload):
        if payload[end:match.start()].strip():
            raise ValueError(f"{path}: malformed face list")
        face = np.asarray([int(v) for v in match.group(2).split()], dtype=np.int64)
        if len(face) != int(match.group(1)) or len(face) < 3 or np.any(face < 0):
            raise ValueError(f"{path}: invalid face {len(faces)}")
        faces.append(face)
        end = match.end()
    if len(faces) != count or payload[end:].strip():
        raise ValueError(f"{path}: declared {count} faces, parsed {len(faces)}")
    return faces


def wall_distance(case_path: Path, coords: np.ndarray) -> np.ndarray:
    """Euclidean distance to the triangulated wall surface, including edges."""
    from vtkmodules.vtkCommonCore import vtkPoints, reference
    from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData, vtkStaticCellLocator, vtkGenericCell
    from vtkmodules.vtkFiltersCore import vtkTriangleFilter
    from vtkmodules.util.numpy_support import numpy_to_vtk

    mesh = case_path / "constant" / "polyMesh"
    text = ascii_text(mesh / "boundary")

    wall_faces: list[int] = []
    for m in re.finditer(r'([A-Za-z0-9_./-]+|"[^"\n]+")\s*\{([^{}]*)\}', text):
        body = m.group(2)
        n_faces = re.search(r"\bnFaces\s+(\d+)\s*;", body)
        start_face = re.search(r"\bstartFace\s+(\d+)\s*;", body)
        ptype = re.search(r"\btype\s+([A-Za-z0-9_./-]+)\s*;", body)
        if n_faces and start_face and ptype and ptype.group(1) == "wall":
            s = int(start_face.group(1))
            wall_faces.extend(range(s, s + int(n_faces.group(1))))

    if not wall_faces:
        raise ValueError(f"{mesh}: no faces in patches of type wall")
    points = read_field(mesh / "points", dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(coords).all():
        raise ValueError(f"{mesh}: invalid point coordinates")
    faces = read_faces(mesh / "faces")
    vertices = vtkPoints()
    vertices.SetData(numpy_to_vtk(points, deep=True))
    polygons = vtkCellArray()
    expected_triangles = 0
    for i in wall_faces:
        if i >= len(faces) or np.any(faces[i] >= len(points)):
            raise ValueError(f"{mesh}: invalid wall-face or vertex index")
        face = faces[i]
        polygons.InsertNextCell(len(face))
        for vertex in face:
            polygons.InsertCellPoint(int(vertex))
        expected_triangles += len(face) - 2
    surface = vtkPolyData()
    surface.SetPoints(vertices)
    surface.SetPolys(polygons)
    triangulator = vtkTriangleFilter()
    triangulator.SetInputData(surface)
    triangulator.Update()
    triangles = triangulator.GetOutput()
    if triangles.GetNumberOfCells() != expected_triangles:
        raise ValueError(f"{mesh}: wall triangulation failed; check degenerate faces")
    locator = vtkStaticCellLocator()
    locator.SetDataSet(triangles)
    locator.BuildLocator()
    closest, cell = [0.0, 0.0, 0.0], vtkGenericCell()
    cell_id, sub_id, dist2 = reference(0), reference(0), reference(0.0)
    distances = np.empty((len(coords), 1), dtype=np.float32)
    for i, point in enumerate(coords):
        locator.FindClosestPoint(point, closest, cell, cell_id, sub_id, dist2)
        if int(cell_id) < 0 or not np.isfinite(float(dist2)):
            raise ValueError(f"{mesh}: nearest-wall lookup failed for cell {i}")
        distances[i, 0] = np.sqrt(max(0.0, float(dist2)))
    return distances


def viscosity(case_path: Path) -> float:
    defs = case_path / "caseDef"
    if defs.exists():
        for raw in defs.read_text(errors="replace").splitlines():
            m = re.match(r"nu\s+([-+0-9.eE]+)\s*;", raw.split("//", 1)[0].strip())
            if m:
                return float(m.group(1))
    props = case_path / "constant" / "transportProperties"
    for raw in props.read_text(errors="replace").splitlines():
        line = raw.split("//", 1)[0].strip()
        if line.startswith("nu") and "$" not in line:
            nums = re.findall(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?", line)
            if nums:
                return float(nums[-1])


def case_paths(data_root: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for fam in sorted((data_root / "Parm_PH_29").glob("alpha_*")):
        for case in sorted(fam.iterdir()):
            if case.is_dir():
                paths[case.name] = case
    for case in sorted((data_root / "DUCT").glob("AR_*")):
        if case.is_dir():
            paths[case.name] = case
    for name in ("CBFS", "PH_Breuer", "NASA_2DWMH"):
        paths[name] = data_root / name
    return paths


def final_time_dir(case_path: Path) -> Path:
    times = sorted(
        (float(c.name), c) for c in case_path.iterdir()
        if c.is_dir() and c.name.replace(".", "", 1).isdigit()
    )
    return next(path for _, path in reversed(times) if (path / "U").exists())


def load_case(name: str, path: Path, *, with_target: bool = False) -> Case:
    time_dir = final_time_dir(path)
    coord_file = next(p for p in (path / "constant" / "C", path / "0" / "C", time_dir / "C") if p.exists())
    coords = read_field(coord_file).astype(np.float32)[:, :3]
    n = coords.shape[0]

    u_rans = read_cell_field(time_dir / "U", n, 3)
    scalars = {}
    for field in ("k", "omega", "p", "nut"):
        scalars[field] = read_cell_field(time_dir / field, n, 1)

    u_les = read_cell_field(path / "0" / "U_LES", n, 3) if with_target else None

    k, omega, p, nut = (scalars[f] for f in ("k", "omega", "p", "nut"))
    nu = np.float32(max(viscosity(path), 1e-12))
    k_pos = np.maximum(k, 1e-12)
    omega_abs = np.maximum(np.abs(omega), 1e-12)
    nut_pos = np.maximum(nut, 1e-12)
    d_wall = np.maximum(wall_distance(path, coords), 1e-8)

    u_mag = np.linalg.norm(u_rans, axis=1, keepdims=True)
    u_ref = np.float32(max(u_mag.mean(), 1e-6))
    p_dyn = p / (u_ref ** 2)

    def log_scale(x):
        return np.log(np.maximum(x, 1e-12)).astype(np.float32)

    features = np.concatenate([
        coords,
        (u_rans / u_ref).astype(np.float32),
        (u_mag / u_ref).astype(np.float32),
        log_scale(k_pos),
        log_scale(omega_abs),
        (np.sign(p_dyn) * np.log1p(np.abs(p_dyn))).astype(np.float32),
        log_scale(nut_pos),
        log_scale(k_pos / omega_abs),
        log_scale(np.sqrt(k_pos) / (omega_abs * d_wall)),  
        log_scale(nut_pos / nu),
        d_wall,
        log_scale(d_wall * np.sqrt(k_pos) / nu),          
    ], axis=1).astype(np.float32)

    return Case(name=name, coords=coords, features=features, u_rans=u_rans, u_les=u_les)


def build(data_root: Path, output: Path) -> None:
    paths = case_paths(data_root)
    print(f"train {len(TRAIN_CASES)} cases · test {len(TEST_CASES)} cases")
    out: dict[str, np.ndarray] = {}

    x, base, true, ids = [], [], [], []
    for i, name in enumerate(TRAIN_CASES):
        c = load_case(name, paths[name], with_target=True)
        x.append(c.features)
        base.append(c.u_rans)
        true.append(c.u_les)
        ids.append(np.full(c.features.shape[0], i, dtype=np.int32))
        print(f"  train {name:22s} {c.features.shape[0]:6d} pts")
    out["x_train"] = np.concatenate(x)
    out["base_train"] = np.concatenate(base)
    out["true_train"] = np.concatenate(true)
    out["id_train"] = np.concatenate(ids)
    out["names_train"] = np.asarray(TRAIN_CASES)

    # Store full test meshes so normalization & context match training, plus global indices of evaluation points
    x, base, ids, eval_idx, eval_id = [], [], [], [], []
    offset = 0
    for i, name in enumerate(TEST_CASES):
        c = load_case(name, paths[name], with_target=False)
        pts = np.loadtxt(data_root / "evaluation_points" / f"{name}_points.csv", delimiter=",").astype(np.float32)
        _, nearest = cKDTree(c.coords).query(pts, k=1)

        x.append(c.features)
        base.append(c.u_rans)
        ids.append(np.full(c.features.shape[0], i, dtype=np.int32))
        eval_idx.append((offset + nearest).astype(np.int64))
        eval_id.append(np.full(pts.shape[0], i, dtype=np.int32))
        offset += c.features.shape[0]
        print(f"  test  {name:22s} {c.features.shape[0]:6d} mesh pts ({pts.shape[0]:5d} eval pts)")

    out["x_test"] = np.concatenate(x)
    out["base_test"] = np.concatenate(base)
    out["id_test"] = np.concatenate(ids)
    out["names_test"] = np.asarray(TEST_CASES)
    out["eval_idx_test"] = np.concatenate(eval_idx)
    out["eval_id_test"] = np.concatenate(eval_id)
    out["features"] = np.asarray(FEATURES)
    out["preprocessing_version"] = np.asarray(2, dtype=np.int32)
    out["wall_distance_method"] = np.asarray("vtk_triangulated_wall_surface")

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **out)
    print(f"\nwrote {output} ({output.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", type=Path, help="the benchmark repository data/ directory")
    parser.add_argument("--out", type=Path, default=Path("cache.npz"))
    args = parser.parse_args()
    build(args.data_root, args.out)
