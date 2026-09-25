"""Input/output helpers for immersed FSI simulations.

CSV files are used for small, human-readable parameters.  NPZ files are used
for arrays, solution histories, final states, and restart data.  Restart
helpers return dictionaries that can be passed directly to ``solve_ib`` or
``solve_id`` as ``initial_fluid_state``, ``initial_solid_state``, and
``initial_lambda``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import csv
import json
import struct
import warnings
import zipfile

import numpy as np
from numpy.lib import format as npy_format


@dataclass(frozen=True)
class SaveRecord:
    """Paths and metadata returned by ``save_results``."""

    npz_path: Path
    csv_path: Path | None
    params: dict
    array_keys: tuple[str, ...]


@dataclass
class LoadedResults:
    """Parameters and arrays loaded from a saved simulation."""

    npz_path: Path
    csv_path: Path | None = None
    params: dict = field(default_factory=dict)
    arrays: dict = field(default_factory=dict)


@dataclass
class RestartState:
    """Final saved state prepared for a solver restart."""

    initial_fluid_state: dict
    initial_solid_state: dict
    initial_lambda: np.ndarray | None
    P: np.ndarray
    T: np.ndarray
    bdry: np.ndarray
    params: dict = field(default_factory=dict)
    path: Path | None = None
    start_step: int | None = None
    start_time: float | None = None
    d0_x_s: np.ndarray | None = None
    d0_y_s: np.ndarray | None = None

    @property
    def solid_mesh(self):
        """Return ``(P, T, bdry)`` for direct use in solver calls."""
        return self.P, self.T, self.bdry

    def solver_kwargs(self, include_mesh=True, include_offsets=True):
        """Return keyword arguments that can be passed to ``solve_ib/solve_id``."""
        kwargs = {
            "initial_fluid_state": self.initial_fluid_state,
            "initial_solid_state": self.initial_solid_state,
        }
        if self.initial_lambda is not None:
            kwargs["initial_lambda"] = self.initial_lambda
        if include_mesh:
            kwargs["solid_mesh"] = self.solid_mesh
        if include_offsets:
            if self.d0_x_s is not None:
                kwargs["d0_x_s"] = self.d0_x_s
            if self.d0_y_s is not None:
                kwargs["d0_y_s"] = self.d0_y_s
        return kwargs


def _csv_value(value):
    """Serialize a small Python/numpy value for one CSV cell."""
    if value is None:
        return ""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (bool, np.bool_)):
        return "true" if bool(value) else "false"
    if isinstance(value, np.generic):
        return str(value.item())
    if isinstance(value, np.ndarray):
        if value.ndim <= 1:
            return " ".join(str(v) for v in value.reshape(-1))
        return json.dumps(value.tolist())
    if isinstance(value, (list, tuple)):
        if all(np.isscalar(v) or isinstance(v, Path) for v in value):
            return " ".join(_csv_value(v) for v in value)
        return json.dumps(value, default=str)
    if isinstance(value, dict):
        return json.dumps(value, default=str, sort_keys=True)
    return str(value)


def write_params_csv(csv_file, params):
    """Write a one-row parameter CSV with stable column order."""
    csv_file = Path(csv_file)
    csv_file.parent.mkdir(parents=True, exist_ok=True)
    params = dict(params or {})
    with csv_file.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(params.keys())
        writer.writerow(_csv_value(value) for value in params.values())
    return csv_file


def read_params_csv(csv_file):
    """Read a one-row parameter CSV and return values as strings."""
    csv_file = Path(csv_file)
    if not csv_file.exists():
        raise FileNotFoundError(csv_file)
    with csv_file.open(newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        return {}
    if len(rows) < 2:
        return {}
    return dict(zip(rows[0], rows[1]))


def parse_bool(value, default=False):
    """Parse a CSV-style boolean value."""
    if value is None or value == "":
        return bool(default)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def parse_float(params, key, default=None):
    """Parse a floating-point value from a parameter dictionary."""
    try:
        return float(params[key])
    except (KeyError, TypeError, ValueError):
        return default


def parse_int(params, key, default=None):
    """Parse an integer value from a parameter dictionary."""
    try:
        return int(float(params[key]))
    except (KeyError, TypeError, ValueError):
        return default


def time_tag(value):
    """Return a filename-safe time tag."""
    return f"t{float(value):.6g}".replace("-", "m").replace(".", "p")


def result_paths(folder_or_npz, file_base=None):
    """Return ``(npz_path, csv_path)`` from a folder/base or full file path."""
    base = Path(folder_or_npz)
    if file_base is None:
        if base.suffix == ".npz":
            npz_path = base
        elif base.suffix == ".csv":
            npz_path = base.with_suffix(".npz")
        elif base.suffix:
            npz_path = base.with_suffix(".npz")
        else:
            raise ValueError("file_base is required when folder_or_npz is a folder.")
    else:
        npz_path = base / f"{file_base}.npz"
    return npz_path, npz_path.with_suffix(".csv")


def unique_output_path(path):
    """Return ``path`` or a numbered variant that does not exist."""
    path = Path(path)
    if not path.exists():
        return path
    for index in range(1, 1000):
        candidate = path.with_name(f"{path.stem}_{index}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not find a free output name based on {path}.")


def save_npz(npz_file, compressed=False, overwrite=True, **arrays):
    """Save arrays to NPZ using a temporary file and atomic replace."""
    npz_file = Path(npz_file)
    npz_file.parent.mkdir(parents=True, exist_ok=True)
    if npz_file.exists() and not overwrite:
        raise FileExistsError(npz_file)

    arrays = {key: value for key, value in arrays.items() if value is not None}
    temporary = npz_file.with_name(f".{npz_file.stem}.tmp{npz_file.suffix}")
    try:
        if compressed:
            np.savez_compressed(temporary, **arrays)
        else:
            np.savez(temporary, **arrays)
        temporary.replace(npz_file)
    finally:
        if temporary.exists():
            temporary.unlink()
    return npz_file


def load_npz(npz_file, keys=None, allow_pickle=False):
    """Load selected arrays from an NPZ archive as independent copies."""
    npz_file = Path(npz_file)
    with np.load(npz_file, allow_pickle=allow_pickle) as data:
        selected = data.files if keys is None else keys
        return {key: np.asarray(data[key]).copy() for key in selected if key in data}


def npz_array_names(npz_file):
    """Return array names stored in an NPZ file without loading the arrays."""
    npz_file = Path(npz_file)
    with zipfile.ZipFile(npz_file) as archive:
        return {
            Path(info.filename).stem
            for info in archive.infolist()
            if info.filename.endswith(".npy")
        }


def npz_array_header(npz_file, name):
    """Return ``(shape, fortran_order, dtype)`` for one array in an NPZ file."""
    npz_file = Path(npz_file)
    member = f"{name}.npy"
    with zipfile.ZipFile(npz_file) as archive:
        if member not in archive.namelist():
            raise KeyError(f"{npz_file} is missing array {name!r}.")
        with archive.open(member) as raw:
            version = npy_format.read_magic(raw)
            if version == (1, 0):
                shape, fortran_order, dtype = npy_format.read_array_header_1_0(raw)
            elif version == (2, 0):
                shape, fortran_order, dtype = npy_format.read_array_header_2_0(raw)
            else:
                raise ValueError(f"Unsupported .npy version {version} for {name!r}.")
    return tuple(shape), bool(fortran_order), np.dtype(dtype)


def _stored_member_data_start(npz_file, member):
    """Return the byte offset of an uncompressed NPZ member payload."""
    npz_file = Path(npz_file)
    with zipfile.ZipFile(npz_file) as archive:
        info = archive.getinfo(member)
        if info.compress_type != zipfile.ZIP_STORED:
            return None
        header_offset = info.header_offset

    with npz_file.open("rb") as raw_file:
        raw_file.seek(header_offset)
        header = raw_file.read(30)
        if len(header) != 30:
            raise EOFError(f"Could not read local ZIP header for {member}.")
        (
            signature,
            _version_needed,
            _flags,
            _compression,
            _mod_time,
            _mod_date,
            _crc,
            _compressed_size,
            _uncompressed_size,
            filename_len,
            extra_len,
        ) = struct.unpack("<IHHHHHIIIHH", header)
        if signature != 0x04034B50:
            raise ValueError(f"Invalid ZIP local header for {member}.")
        return header_offset + 30 + filename_len + extra_len


def _stored_npy_layout(npz_file, name):
    """Return layout metadata for an uncompressed, C-contiguous NPY member."""
    member = f"{name}.npy"
    try:
        member_start = _stored_member_data_start(npz_file, member)
    except KeyError:
        raise KeyError(f"{npz_file} is missing array {name!r}.") from None
    if member_start is None:
        return None

    with Path(npz_file).open("rb") as raw_file:
        raw_file.seek(member_start)
        version = npy_format.read_magic(raw_file)
        if version == (1, 0):
            shape, fortran_order, dtype = npy_format.read_array_header_1_0(raw_file)
        elif version == (2, 0):
            shape, fortran_order, dtype = npy_format.read_array_header_2_0(raw_file)
        else:
            return None
        data_start = raw_file.tell()

    dtype = np.dtype(dtype)
    if fortran_order or dtype.hasobject:
        return None
    return tuple(shape), dtype, data_start


def read_npz_array(npz_file, name):
    """Read one NPZ array as a copied numpy array."""
    return load_npz(npz_file, keys=[name])[name]


def _as_field_history(name, values):
    """Return a fluid field array with a leading frame axis."""
    values = np.asarray(values)
    if values.ndim == 2:
        return values.reshape((1,) + values.shape)
    if values.ndim >= 3:
        return values
    raise ValueError(f"{name!r} must be a 2D field or a field history.")


def read_npz_frame_chunk(npz_file, name, start, stop):
    """Read contiguous frames from an NPZ field history.

    Uncompressed NPZ files are read directly from the underlying NPY payload.
    Compressed or Fortran-ordered arrays fall back to loading the whole member.
    """
    npz_file = Path(npz_file)
    start = int(start)
    stop = int(stop)
    if start < 0 or stop < start:
        raise ValueError(f"Invalid frame range {start}:{stop}.")

    layout = _stored_npy_layout(npz_file, name)
    if layout is None:
        array = _as_field_history(name, read_npz_array(npz_file, name))
        return array[start:stop].copy()

    shape, dtype, data_start = layout
    if len(shape) < 3:
        if start != 0 or stop > 1:
            raise IndexError(f"{name!r} has one stored frame, requested {start}:{stop}.")
        n_values = int(np.prod(shape))
        nbytes = n_values * dtype.itemsize
        with npz_file.open("rb") as raw_file:
            raw_file.seek(data_start)
            buffer = raw_file.read(nbytes)
        if len(buffer) != nbytes:
            raise EOFError(f"Could not read {name!r} from {npz_file}.")
        return np.frombuffer(buffer, dtype=dtype, count=n_values).copy().reshape((1,) + shape)

    n_stored = shape[0]
    if stop > n_stored:
        raise IndexError(f"{name!r} has {n_stored} frames, requested {start}:{stop}.")

    frame_shape = shape[1:]
    frame_values = int(np.prod(frame_shape))
    frame_bytes = frame_values * dtype.itemsize
    n_frames = stop - start
    expected = frame_bytes * n_frames
    with npz_file.open("rb") as raw_file:
        raw_file.seek(data_start + frame_bytes * start)
        buffer = raw_file.read(expected)
    if len(buffer) != expected:
        raise EOFError(f"Could not read frames {start}:{stop} of {name!r}.")
    return np.frombuffer(buffer, dtype=dtype, count=frame_values * n_frames).copy().reshape(
        (n_frames,) + frame_shape
    )


def read_npz_frame(npz_file, name, index):
    """Read one frame from an NPZ field history."""
    index = int(index)
    if index < 0:
        shape, _fortran_order, _dtype = npz_array_header(npz_file, name)
        n_frames = 1 if len(shape) < 3 else shape[0]
        index += n_frames
    return read_npz_frame_chunk(npz_file, name, index, index + 1)[0]


def read_npz_final_frame(npz_file, name):
    """Read the final frame from an NPZ field history."""
    shape, _fortran_order, _dtype = npz_array_header(npz_file, name)
    if len(shape) < 3:
        return read_npz_frame_chunk(npz_file, name, 0, 1)[0]
    return read_npz_frame_chunk(npz_file, name, shape[0] - 1, shape[0])[0]


def _normalize_save_mode(save_mode, minimal_save=None):
    """Return one of ``history``, ``final``, or ``both``."""
    if minimal_save is not None:
        return "final" if minimal_save else "history"
    save_mode = str(save_mode).strip().lower()
    aliases = {
        "full": "history",
        "history": "history",
        "all": "history",
        "final": "final",
        "minimal": "final",
        "final_only": "final",
        "last": "final",
        "both": "both",
        "history_and_final": "both",
    }
    try:
        return aliases[save_mode]
    except KeyError as exc:
        raise ValueError("save_mode must be 'history', 'final', or 'both'.") from exc


def _last_frame(values, name):
    """Return the last saved frame from a list or numpy array."""
    if values is None:
        raise ValueError(f"{name} is missing.")
    if isinstance(values, (list, tuple)):
        if len(values) == 0:
            raise ValueError(f"{name} has no saved frames.")
        return np.asarray(values[-1])
    values = np.asarray(values)
    if values.shape[0] == 0:
        raise ValueError(f"{name} has no saved frames.")
    if values.ndim >= 2:
        return values[-1]
    return values


def _history_array(values, name, frame_indices=None):
    """Return a non-empty history array."""
    if values is None:
        raise ValueError(f"{name} is missing.")
    if isinstance(values, (list, tuple)) and len(values) == 0:
        raise ValueError(f"{name} has no saved frames.")
    values = np.asarray(values)
    if values.shape[0] == 0:
        raise ValueError(f"{name} has no saved frames.")
    if frame_indices is not None and values.shape[0] != len(frame_indices):
        raise ValueError(
            f"{name} stores {values.shape[0]} frames, "
            f"but frame_indices has {len(frame_indices)} entries."
        )
    return values


def _result_parts(result):
    """Return standard pieces from an FSIResult, dict, or legacy tuple."""
    if hasattr(result, "as_legacy_tuple"):
        u_f, v_f, p_f, u_s, v_s, a_s, P, T, bdry, total_iterations = result.as_legacy_tuple()
        return {
            "u_f": u_f,
            "v_f": v_f,
            "p_f": p_f,
            "u_s": u_s,
            "v_s": v_s,
            "a_s": a_s,
            "P": P,
            "T": T,
            "B": bdry,
            "total_iterations": total_iterations,
            "method": getattr(result, "method", ""),
            "transfer_kind": getattr(result, "transfer_kind", ""),
            "element_type": getattr(result, "element_type", ""),
            "final_lambda": getattr(result, "final_lambda", None),
            "iteration_history": getattr(result, "iteration_history", None),
        }

    if isinstance(result, dict):
        bdry = result.get("B", result.get("bdry"))
        total_iterations = result.get(
            "total_iterations",
            result.get("tot_it", result.get("max_it", 0)),
        )
        return {
            "u_f": result.get("u_f", result.get("usol_f")),
            "v_f": result.get("v_f", result.get("vsol_f")),
            "p_f": result.get("p_f", result.get("psol_f")),
            "u_s": result.get("u_s", result.get("usol_s")),
            "v_s": result.get("v_s", result.get("vsol_s")),
            "a_s": result.get("a_s", result.get("asol_s")),
            "P": result.get("P"),
            "T": result.get("T"),
            "B": bdry,
            "total_iterations": total_iterations,
            "method": result.get("method", ""),
            "transfer_kind": result.get("transfer_kind", ""),
            "element_type": result.get("element_type", ""),
            "final_lambda": result.get("final_lambda", result.get("lambda")),
            "iteration_history": result.get("iteration_history"),
        }

    if isinstance(result, (tuple, list)) and len(result) >= 10:
        u_f, v_f, p_f, u_s, v_s, a_s, P, T, bdry, total_iterations = result[:10]
        return {
            "u_f": u_f,
            "v_f": v_f,
            "p_f": p_f,
            "u_s": u_s,
            "v_s": v_s,
            "a_s": a_s,
            "P": P,
            "T": T,
            "B": bdry,
            "total_iterations": total_iterations,
            "method": "",
            "transfer_kind": "",
            "element_type": "",
            "final_lambda": None,
            "iteration_history": None,
        }

    raise TypeError("result must be an FSIResult, a dict, or a legacy result tuple.")


def _iteration_history_arrays(history):
    """Return numeric arrays for a list of iteration-history dictionaries."""
    if not history:
        return {}
    fields = sorted({key for row in history for key in row})
    values = np.full((len(history), len(fields)), np.nan, dtype=float)
    for i, row in enumerate(history):
        for j, field_name in enumerate(fields):
            value = row.get(field_name, np.nan)
            try:
                values[i, j] = float(value)
            except (TypeError, ValueError):
                values[i, j] = np.nan
    return {
        "iteration_history": values,
        "iteration_history_fields": np.asarray(fields, dtype="U64"),
    }


def result_arrays(
    result,
    *,
    rho_f=1.0,
    save_mode="history",
    frame_indices=None,
    extra_arrays=None,
    include_iteration_history=True,
):
    """Build the NPZ array dictionary for a solver result."""
    parts = _result_parts(result)
    save_mode = _normalize_save_mode(save_mode)

    P = np.asarray(parts["P"], dtype=float)
    T = np.asarray(parts["T"], dtype=int)
    B = np.asarray(parts["B"], dtype=int)
    if P.ndim != 2 or P.shape[0] != 2:
        raise ValueError("P must have shape (2, Nb).")
    if B.ndim != 2 or B.shape[0] < 4:
        raise ValueError("B/bdry must be a boundary array.")

    arrays = {
        "P": P,
        "T": T,
        "B": B,
        "bdry": B,
    }
    if parts["method"]:
        arrays["method"] = np.asarray(parts["method"])
    if parts["transfer_kind"]:
        arrays["transfer_kind"] = np.asarray(parts["transfer_kind"])
    if parts["element_type"]:
        arrays["element_type"] = np.asarray(parts["element_type"])
    if frame_indices is not None:
        arrays["frame_indices"] = np.asarray(frame_indices, dtype=np.int64)

    if save_mode in {"history", "both"}:
        u_f = _history_array(parts["u_f"], "u_f", frame_indices)
        v_f = _history_array(parts["v_f"], "v_f", frame_indices)
        arrays["u_f"] = u_f
        arrays["v_f"] = v_f
        if parts["p_f"] is not None and len(parts["p_f"]) != 0:
            p_f = _history_array(parts["p_f"], "p_f", frame_indices)
            arrays["p_f"] = p_f
            arrays["p_f_physical"] = float(rho_f) * p_f
        arrays["u_s"] = _history_array(parts["u_s"], "u_s", frame_indices)
        arrays["v_s"] = _history_array(parts["v_s"], "v_s", frame_indices)
        arrays["a_s"] = _history_array(parts["a_s"], "a_s", frame_indices)

    if save_mode in {"final", "both"}:
        arrays["u_f_final"] = _last_frame(parts["u_f"], "u_f")
        arrays["v_f_final"] = _last_frame(parts["v_f"], "v_f")
        if parts["p_f"] is not None and len(parts["p_f"]) != 0:
            p_final = _last_frame(parts["p_f"], "p_f")
            arrays["p_f_final"] = p_final
            arrays["p_f_physical_final"] = float(rho_f) * p_final
        arrays["u_s_final"] = _last_frame(parts["u_s"], "u_s")
        arrays["v_s_final"] = _last_frame(parts["v_s"], "v_s")
        arrays["a_s_final"] = _last_frame(parts["a_s"], "a_s")

    if parts["final_lambda"] is not None:
        arrays["final_lambda"] = np.asarray(parts["final_lambda"], dtype=float)
    if include_iteration_history:
        arrays.update(_iteration_history_arrays(parts["iteration_history"]))
    if extra_arrays:
        arrays.update({key: value for key, value in extra_arrays.items() if value is not None})
    return arrays


def save_results(
    folder,
    file_base=None,
    result=None,
    params=None,
    *,
    method=None,
    rho_f=1.0,
    save_mode="history",
    minimal_save=None,
    compressed=False,
    overwrite=True,
    unique=False,
    frame_indices=None,
    extra_arrays=None,
    extra_params=None,
    include_iteration_history=True,
    print_paths=True,
):
    """Save a solver result to paired NPZ and CSV files.

    ``folder`` can be either a directory, when ``file_base`` is provided, or a
    full output path such as ``DATA/case_ID.npz``.
    """
    if result is None:
        raise ValueError("result must be provided.")
    save_mode = _normalize_save_mode(save_mode, minimal_save=minimal_save)
    npz_path, csv_path = result_paths(folder, file_base=file_base)
    if unique:
        npz_path = unique_output_path(npz_path)
        csv_path = npz_path.with_suffix(".csv")

    parts = _result_parts(result)
    method_name = method or parts.get("method") or parts.get("transfer_kind") or ""
    arrays = result_arrays(
        result,
        rho_f=rho_f,
        save_mode=save_mode,
        frame_indices=frame_indices,
        extra_arrays=extra_arrays,
        include_iteration_history=include_iteration_history,
    )

    params_out = dict(params or {})
    params_out.update(extra_params or {})
    if method_name:
        params_out.setdefault("coupling_method", method_name)
    if parts.get("transfer_kind"):
        params_out.setdefault("transfer_kind", parts["transfer_kind"])
    if parts.get("element_type"):
        params_out.setdefault("element_type", parts["element_type"])
    params_out.setdefault("save_mode", save_mode)
    params_out.setdefault("compressed_npz", bool(compressed))
    params_out.setdefault("total_fixed_point_iterations", parts["total_iterations"])
    params_out.setdefault("saved_arrays", " ".join(sorted(arrays)))
    if frame_indices is not None:
        params_out.setdefault("saved_frames", len(frame_indices))

    if csv_path.exists() and not overwrite:
        raise FileExistsError(csv_path)
    csv_path = write_params_csv(csv_path, params_out)
    npz_path = save_npz(npz_path, compressed=compressed, overwrite=overwrite, **arrays)

    if print_paths:
        print(f"Saved results to {npz_path} and {csv_path}")
    return SaveRecord(npz_path=npz_path, csv_path=csv_path, params=params_out, array_keys=tuple(sorted(arrays)))


def load_results(npz_file, csv_file=None, keys=None, allow_pickle=False):
    """Load a saved result archive and its parameter CSV if present."""
    npz_file = Path(npz_file)
    if npz_file.suffix != ".npz":
        npz_file = npz_file.with_suffix(".npz")
    if csv_file is None:
        csv_path = npz_file.with_suffix(".csv")
    else:
        csv_path = Path(csv_file)

    params = read_params_csv(csv_path) if csv_path.exists() else {}
    arrays = load_npz(npz_file, keys=keys, allow_pickle=allow_pickle)
    return LoadedResults(npz_path=npz_file, csv_path=csv_path if csv_path.exists() else None, params=params, arrays=arrays)


def normalize_solid_history(values, nb, name):
    """Return a solid history with shape ``(n_frames, 2*Nb)``."""
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2:
        raise ValueError(f"{name} must be 1D or 2D, got shape {values.shape}.")
    if values.shape[1] != 2 * nb and values.shape[0] == 2 * nb:
        values = values.T
    if values.shape[1] != 2 * nb:
        raise ValueError(
            f"{name} must have 2*Nb={2 * nb} columns, got shape {values.shape}."
        )
    return values


def append_history(old_history, new_history, drop_duplicate_initial=True):
    """Append two histories, optionally skipping the first frame of the new one."""
    old_history = np.asarray(old_history)
    new_history = np.asarray(new_history)
    if old_history.ndim == new_history.ndim - 1:
        old_history = old_history.reshape((1,) + old_history.shape)
    if new_history.ndim == old_history.ndim - 1:
        new_history = new_history.reshape((1,) + new_history.shape)
    if old_history.shape[1:] != new_history.shape[1:]:
        raise ValueError(
            f"history shapes are incompatible: {old_history.shape} and {new_history.shape}."
        )
    start = 1 if drop_duplicate_initial and new_history.shape[0] > 0 else 0
    return np.concatenate((old_history, new_history[start:]), axis=0)


def _first_array(arrays, *names):
    """Return the first available array from a dictionary."""
    for name in names:
        if name in arrays:
            return np.asarray(arrays[name])
    return None


def _final_field(arrays, base_name, shape=None, missing_zero=False):
    """Return the final fluid field for ``base_name``."""
    final = _first_array(arrays, f"{base_name}_final", f"{base_name}_physical_final")
    if final is None:
        values = _first_array(arrays, base_name, f"{base_name}_physical")
        if values is not None:
            values = np.asarray(values, dtype=float)
            final = values[-1] if values.ndim >= 3 else values
    if final is None and missing_zero and shape is not None:
        final = np.zeros(shape, dtype=float)
    if final is None:
        raise KeyError(f"Missing {base_name!r} or {base_name + '_final'!r}.")
    final = np.asarray(final, dtype=float)
    if shape is not None and final.shape != tuple(shape):
        raise ValueError(f"{base_name} final field has shape {final.shape}, expected {shape}.")
    return final.copy()


def _final_solid_vector(arrays, base_name, nb, required=True):
    """Return the final solid vector for ``base_name``."""
    final = _first_array(arrays, f"{base_name}_final")
    if final is None:
        values = _first_array(arrays, base_name)
        if values is not None:
            values = np.asarray(values, dtype=float)
            if values.ndim == 1:
                final = values
            elif values.ndim == 2:
                final = normalize_solid_history(values, nb, base_name)[-1]
    if final is None:
        if required:
            raise KeyError(f"Missing {base_name!r} or {base_name + '_final'!r}.")
        return None
    final = np.asarray(final, dtype=float).reshape(-1)
    if final.size != 2 * nb:
        raise ValueError(f"{base_name} final vector must have length {2 * nb}.")
    return final.copy()


def _infer_fluid_shape(arrays, nx_f=None, ny_f=None):
    """Infer padded fluid field shape."""
    if nx_f is not None and ny_f is not None:
        return int(ny_f) + 2, int(nx_f) + 2
    for key in ("u_f_final", "u_f", "v_f_final", "v_f", "p_f_final", "p_f"):
        if key not in arrays:
            continue
        arr = np.asarray(arrays[key])
        if arr.ndim == 2:
            return arr.shape
        if arr.ndim >= 3:
            return arr.shape[-2:]
    raise ValueError("Could not infer fluid shape; pass nx_f and ny_f.")


def _restart_step_and_time(arrays, params, dt):
    """Infer restart step and physical time from arrays/metadata."""
    if "frame_indices" in arrays:
        start_step = int(np.asarray(arrays["frame_indices"]).reshape(-1)[-1])
    elif "u_s" in arrays:
        start_step = max(int(np.asarray(arrays["u_s"]).shape[0]) - 1, 0)
    else:
        start_step = parse_int(params, "Nt", None)

    start_time = parse_float(params, "restart_target_time", None)
    if start_time is None:
        start_time = parse_float(params, "final_time", None)
    if start_time is None and dt is not None and start_step is not None:
        start_time = float(start_step) * float(dt)
    return start_step, start_time


def restart_state_from_result(result, params=None, dt=None):
    """Build a ``RestartState`` from an in-memory solver result."""
    parts = _result_parts(result)
    P = np.asarray(parts["P"], dtype=float)
    T = np.asarray(parts["T"], dtype=int)
    B = np.asarray(parts["B"], dtype=int)
    nb = P.shape[1]
    initial_fluid_state = {
        "u_f": _last_frame(parts["u_f"], "u_f").copy(),
        "v_f": _last_frame(parts["v_f"], "v_f").copy(),
    }
    if parts["p_f"] is not None and len(parts["p_f"]) != 0:
        initial_fluid_state["p_f"] = _last_frame(parts["p_f"], "p_f").copy()
    initial_solid_state = {
        "u_s": _last_frame(parts["u_s"], "u_s").reshape(2 * nb).copy(),
        "v_s": _last_frame(parts["v_s"], "v_s").reshape(2 * nb).copy(),
        "a_s": _last_frame(parts["a_s"], "a_s").reshape(2 * nb).copy(),
    }
    initial_lambda = None
    if parts["final_lambda"] is not None:
        initial_lambda = np.asarray(parts["final_lambda"], dtype=float).reshape(2 * nb).copy()
    start_step = len(np.asarray(parts["u_s"])) - 1
    start_time = None if dt is None else start_step * float(dt)
    return RestartState(
        initial_fluid_state=initial_fluid_state,
        initial_solid_state=initial_solid_state,
        initial_lambda=initial_lambda,
        P=P,
        T=T,
        bdry=B,
        params=dict(params or {}),
        start_step=start_step,
        start_time=start_time,
    )


def load_restart_state(
    npz_file,
    csv_file=None,
    *,
    nx_f=None,
    ny_f=None,
    dt=None,
    gamma=0.5,
    allow_missing_pressure=True,
):
    """Load the final state of a saved run for use as a solver restart."""
    loaded = load_results(npz_file, csv_file=csv_file)
    arrays = loaded.arrays
    params = loaded.params
    if dt is None:
        dt = parse_float(params, "dt", None)

    P = _first_array(arrays, "P")
    if P is None:
        raise KeyError("Restart archive is missing P.")
    P = np.asarray(P, dtype=float)
    if P.ndim != 2 or P.shape[0] != 2:
        raise ValueError(f"P must have shape (2, Nb), got {P.shape}.")
    nb = P.shape[1]

    T = _first_array(arrays, "T")
    B = _first_array(arrays, "B", "bdry")
    if T is None or B is None:
        raise KeyError("Restart archive must contain T and B/bdry.")
    T = np.asarray(T, dtype=int)
    B = np.asarray(B, dtype=int)

    shape = _infer_fluid_shape(arrays, nx_f=nx_f, ny_f=ny_f)
    u_f = _final_field(arrays, "u_f", shape=shape)
    v_f = _final_field(arrays, "v_f", shape=shape)
    p_f = _final_field(
        arrays,
        "p_f",
        shape=shape,
        missing_zero=allow_missing_pressure,
    )

    u_s = _final_solid_vector(arrays, "u_s", nb)
    v_s = _final_solid_vector(arrays, "v_s", nb)
    a_s = _final_solid_vector(arrays, "a_s", nb, required=False)
    if a_s is None:
        if dt is None:
            raise ValueError("Cannot reconstruct a_s without dt.")
        if "v_s" not in arrays:
            raise ValueError("Cannot reconstruct a_s without v_s history.")
        v_hist = normalize_solid_history(arrays["v_s"], nb, "v_s")
        if v_hist.shape[0] < 2:
            raise ValueError("Cannot reconstruct a_s from fewer than two v_s frames.")
        a_s = np.zeros(2 * nb)
        gamma = float(gamma)
        for n in range(v_hist.shape[0] - 1):
            a_s = (v_hist[n + 1] - v_hist[n]) / (gamma * dt) - ((1.0 - gamma) / gamma) * a_s

    initial_lambda = _first_array(arrays, "final_lambda", "lambda_final", "lam_final", "lambda", "lam")
    if initial_lambda is not None:
        initial_lambda = np.asarray(initial_lambda, dtype=float).reshape(-1)
        if initial_lambda.size != 2 * nb:
            warnings.warn(
                "Saved lambda has incompatible size and will be ignored.",
                RuntimeWarning,
                stacklevel=2,
            )
            initial_lambda = None

    d0_x_s = _first_array(arrays, "d0_x_s")
    d0_y_s = _first_array(arrays, "d0_y_s")
    if d0_x_s is not None:
        d0_x_s = np.asarray(d0_x_s, dtype=float).reshape(nb).copy()
    if d0_y_s is not None:
        d0_y_s = np.asarray(d0_y_s, dtype=float).reshape(nb).copy()

    start_step, start_time = _restart_step_and_time(arrays, params, dt)
    return RestartState(
        initial_fluid_state={"u_f": u_f, "v_f": v_f, "p_f": p_f},
        initial_solid_state={"u_s": u_s, "v_s": v_s, "a_s": a_s},
        initial_lambda=initial_lambda,
        P=P,
        T=T,
        bdry=B,
        params=params,
        path=loaded.npz_path,
        start_step=start_step,
        start_time=start_time,
        d0_x_s=d0_x_s,
        d0_y_s=d0_y_s,
    )


restart = load_restart_state


def save_restart_results(
    output,
    previous_npz,
    new_result,
    *,
    params=None,
    rho_f=1.0,
    compressed=False,
    overwrite=True,
    print_paths=True,
):
    """Save a restarted run, appending old and new solid histories when present."""
    previous = load_results(previous_npz)
    arrays = result_arrays(new_result, rho_f=rho_f, save_mode="both")
    for key in ("u_s", "v_s", "a_s"):
        if key in previous.arrays and key in arrays:
            arrays[key] = append_history(previous.arrays[key], arrays[key])
            arrays[f"{key}_final"] = arrays[key][-1]

    params_out = dict(previous.params)
    params_out.update(params or {})
    params_out["restart"] = True
    params_out["restart_source_file"] = str(Path(previous_npz))
    output_npz, output_csv = result_paths(output)
    write_params_csv(output_csv, params_out)
    save_npz(output_npz, compressed=compressed, overwrite=overwrite, **arrays)
    if print_paths:
        print(f"Saved restarted results to {output_npz} and {output_csv}")
    return SaveRecord(output_npz, output_csv, params_out, tuple(sorted(arrays)))
