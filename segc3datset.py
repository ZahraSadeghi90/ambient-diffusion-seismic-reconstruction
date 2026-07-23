from obspy.io.segy.segy import _read_segy
import json
from pathlib import Path

import numpy as np


DATA_ROOT = Path("/home/zahras90/scratch/ambient/data/segc3")
RAW_ROOT = DATA_ROOT

RAW_FILES = [
    RAW_ROOT / "SEG_45Shot_shots1-9.sgy",
    RAW_ROOT / "SEG_45Shot_shots10-18.sgy",
    RAW_ROOT / "SEG_45Shot_shots19-27.sgy",
    RAW_ROOT / "SEG_45Shot_shots28-36.sgy",
    RAW_ROOT / "SEG_45Shot_shots37-45.sgy",
]

OUT_ROOT = DATA_ROOT / "gx_fixed_p70_d10_pergathermask"

PATCH_T = 128
PATCH_X = 128
STRIDE_T = 64
STRIDE_X = 64

N_TIME = 500
N_GY = 201
DT_SECONDS = 0.008

TEST_SHOT = 1
TEST_GX = 4220000

CORRUPTION_PROBABILITY = 0.70
DELTA_PROBABILITY = 0.10

TRAIN_SEED = 1234
TEST_SEED = 5678

EPS = 1e-8


def get_header_value(header, names):
    for name in names:
        if hasattr(header, name):
            return getattr(header, name)

    raise AttributeError(
        f"None of these SEG-Y header fields exist: {names}"
    )


def read_seg_c3(files):
    records = []

    for file_path in files:
        file_path = Path(file_path)

        if not file_path.is_file():
            raise FileNotFoundError(
                f"SEG-Y file not found: {file_path}"
            )

        print(f"Reading: {file_path}", flush=True)
        segy = _read_segy(str(file_path), headonly=False)

        for trace_object in segy.traces:
            header = trace_object.header

            shot = get_header_value(
                header,
                [
                    "energy_source_point_number",
                    "original_field_record_number",
                    "field_record_number",
                ],
            )
            gx = get_header_value(
                header,
                ["group_coordinate_x"],
            )
            gy = get_header_value(
                header,
                ["group_coordinate_y"],
            )

            trace = np.asarray(
                trace_object.data,
                dtype=np.float32,
            )

            if trace.size < N_TIME:
                continue

            records.append(
                (
                    int(shot),
                    int(gx),
                    int(gy),
                    trace[:N_TIME].copy(),
                )
            )

    return records


def build_constant_gx_gathers(records):
    grouped = {}

    for shot, gx, gy, trace in records:
        grouped.setdefault((shot, gx), []).append((gy, trace))

    gathers = []

    for (shot, gx), traces in grouped.items():
        traces = sorted(traces, key=lambda item: item[0])

        if len(traces) != N_GY:
            continue

        gy_values = [int(item[0]) for item in traces]

        if len(set(gy_values)) != N_GY:
            print(
                f"Skipping shot={shot}, gx={gx}: "
                f"expected {N_GY} unique gy coordinates, "
                f"found {len(set(gy_values))}.",
                flush=True,
            )
            continue

        gather = np.stack(
            [item[1] for item in traces],
            axis=1,
        ).astype(np.float32)

        if gather.shape != (N_TIME, N_GY):
            continue

        gathers.append(
            {
                "shot": int(shot),
                "gx": int(gx),
                "gy_values": gy_values,
                "data": gather,
            }
        )

    return sorted(
        gathers,
        key=lambda item: (item["shot"], item["gx"]),
    )


def make_border_covering_starts(length, patch_size, stride):
    if length < patch_size:
        raise ValueError(
            f"Length {length} is smaller than patch size {patch_size}."
        )

    starts = list(range(0, length - patch_size + 1, stride))
    final_start = length - patch_size

    if starts[-1] != final_start:
        starts.append(final_start)

    return starts


def normalize_patch_to_minus_one_one(
    raw_patch,
    preserve_constant=False,
):
    raw_patch = np.asarray(raw_patch, dtype=np.float32)

    xmin = float(np.min(raw_patch))
    xmax = float(np.max(raw_patch))

    if not np.isfinite(xmin) or not np.isfinite(xmax):
        return None

    amplitude_range = xmax - xmin

    if amplitude_range < EPS:
        if preserve_constant:
            return np.zeros_like(raw_patch, dtype=np.float32)

        return None

    return (
        2.0 * (raw_patch - xmin) / amplitude_range - 1.0
    ).astype(np.float32)


def make_trace_mask(shape, missing_probability, rng):
    nt, nx = shape

    observed_columns = (
        rng.random(nx) >= missing_probability
    ).astype(np.float32)

    return np.broadcast_to(
        observed_columns[None, :],
        (nt, nx),
    ).copy()


def make_conditional_delta_mask(
    primary_mask,
    delta_probability,
    rng,
):
    primary_mask = np.asarray(
        primary_mask,
        dtype=np.float32,
    )

    if primary_mask.ndim != 2:
        raise ValueError(
            f"Expected a 2D primary mask, got shape {primary_mask.shape}."
        )

    if not 0.0 <= delta_probability <= 1.0:
        raise ValueError(
            f"delta_probability must be in [0, 1], got {delta_probability}."
        )

    reference_row = primary_mask[0]

    if not np.all(primary_mask == reference_row[None, :]):
        raise ValueError(
            "The primary mask must be constant along the time axis."
        )

    surviving_indices = np.flatnonzero(reference_row > 0.5)

    if surviving_indices.size == 0:
        raise RuntimeError(
            "The primary mask contains no surviving traces."
        )

    number_to_remove = int(
        round(delta_probability * surviving_indices.size)
    )
    number_to_remove = min(
        max(number_to_remove, 0),
        surviving_indices.size,
    )

    final_columns = reference_row.copy()

    if number_to_remove > 0:
        removed_indices = rng.choice(
            surviving_indices,
            size=number_to_remove,
            replace=False,
        )
        final_columns[removed_indices] = 0.0

    final_mask = np.broadcast_to(
        final_columns[None, :],
        primary_mask.shape,
    ).copy().astype(np.float32)

    if np.any(final_mask > primary_mask):
        raise RuntimeError(
            "The conditional delta mask is not a subset of the primary mask."
        )

    actual_removed = int(
        np.sum(
            (primary_mask[0] > 0.5)
            & (final_mask[0] < 0.5)
        )
    )

    if actual_removed != number_to_remove:
        raise RuntimeError(
            f"Expected to remove {number_to_remove} surviving traces, "
            f"but removed {actual_removed}."
        )

    return final_mask


def extract_patches(
    raw_gather,
    gather_mask,
    gather_mask_delta,
    preserve_constant=False,
):
    nt, nx = raw_gather.shape

    t_starts = make_border_covering_starts(
        nt,
        PATCH_T,
        STRIDE_T,
    )
    x_starts = make_border_covering_starts(
        nx,
        PATCH_X,
        STRIDE_X,
    )

    patches = []

    for t0 in t_starts:
        for x0 in x_starts:
            t1 = t0 + PATCH_T
            x1 = x0 + PATCH_X

            raw_patch = raw_gather[t0:t1, x0:x1]
            mask_patch = gather_mask[t0:t1, x0:x1]
            mask_delta_patch = gather_mask_delta[t0:t1, x0:x1]

            expected_shape = (PATCH_T, PATCH_X)

            if raw_patch.shape != expected_shape:
                raise RuntimeError(
                    f"Unexpected patch shape {raw_patch.shape} "
                    f"at t0={t0}, x0={x0}."
                )

            normalized_patch = normalize_patch_to_minus_one_one(
                raw_patch,
                preserve_constant=preserve_constant,
            )

            if normalized_patch is None:
                print(
                    f"Skipping invalid or constant patch "
                    f"at t0={t0}, x0={x0}.",
                    flush=True,
                )
                continue

            patches.append(
                {
                    "gt": normalized_patch,
                    "mask": mask_patch.astype(np.float32),
                    "mask_delta": mask_delta_patch.astype(np.float32),
                    "t0": int(t0),
                    "t1": int(t1),
                    "x0": int(x0),
                    "x1": int(x1),
                }
            )

    return patches, t_starts, x_starts


def save_patch(out_dir, index, patch, p_int, d_int):
    stem = f"{index:08d}"

    np.save(out_dir / f"{stem}.npy", patch["gt"])
    np.save(out_dir / f"{stem}_mask_{p_int}.npy", patch["mask"])
    np.save(
        out_dir / f"{stem}_mask_delta_{d_int}.npy",
        patch["mask_delta"],
    )


def make_training_split(gathers, out_dir, rng):
    out_dir.mkdir(parents=True, exist_ok=True)

    p_int = int(round(CORRUPTION_PROBABILITY * 100))
    d_int = int(round(DELTA_PROBABILITY * 100))

    selected_gathers = [
        gather
        for gather in gathers
        if gather["shot"] != TEST_SHOT
    ]

    if not selected_gathers:
        raise RuntimeError("No training gathers were selected.")

    patch_index = 0

    for gather_number, gather_info in enumerate(selected_gathers, start=1):
        shot = int(gather_info["shot"])
        gx = int(gather_info["gx"])
        raw_gather = gather_info["data"]

        gather_mask = make_trace_mask(
            raw_gather.shape,
            CORRUPTION_PROBABILITY,
            rng,
        )
        gather_mask_delta = make_conditional_delta_mask(
            gather_mask,
            DELTA_PROBABILITY,
            rng,
        )

        patches, _, _ = extract_patches(
            raw_gather,
            gather_mask,
            gather_mask_delta,
        )

        for patch in patches:
            save_patch(
                out_dir,
                patch_index,
                patch,
                p_int,
                d_int,
            )
            patch_index += 1

        print(
            f"Train gather {gather_number}/{len(selected_gathers)} | "
            f"shot={shot} | gx={gx} | total patches={patch_index}",
            flush=True,
        )

    return {
        "number_of_gathers": len(selected_gathers),
        "number_of_patches": patch_index,
    }


def make_test_split(gathers, out_dir, rng):
    out_dir.mkdir(parents=True, exist_ok=True)

    p_int = int(round(CORRUPTION_PROBABILITY * 100))
    d_int = int(round(DELTA_PROBABILITY * 100))

    selected_gathers = [
        gather
        for gather in gathers
        if (
            gather["shot"] == TEST_SHOT
            and gather["gx"] == TEST_GX
        )
    ]

    if len(selected_gathers) != 1:
        available_gx = sorted(
            gather["gx"]
            for gather in gathers
            if gather["shot"] == TEST_SHOT
        )

        raise RuntimeError(
            f"Expected exactly one test gather for "
            f"shot={TEST_SHOT}, gx={TEST_GX}, "
            f"but found {len(selected_gathers)}. "
            f"Available gx values for shot {TEST_SHOT}: {available_gx}"
        )

    gather_info = selected_gathers[0]
    raw_gather = gather_info["data"]

    gather_mask = make_trace_mask(
        raw_gather.shape,
        CORRUPTION_PROBABILITY,
        rng,
    )
    gather_mask_delta = make_conditional_delta_mask(
        gather_mask,
        DELTA_PROBABILITY,
        rng,
    )

    patches, t_starts, x_starts = extract_patches(
        raw_gather,
        gather_mask,
        gather_mask_delta,
        preserve_constant=True,
    )

    test_metadata = []

    for patch_index, patch in enumerate(patches):
        save_patch(
            out_dir,
            patch_index,
            patch,
            p_int,
            d_int,
        )

        test_metadata.append(
            {
                "index": int(patch_index),
                "filename": f"{patch_index:08d}.npy",
                "mask_filename": f"{patch_index:08d}_mask_{p_int}.npy",
                "mask_delta_filename": (
                    f"{patch_index:08d}_mask_delta_{d_int}.npy"
                ),
                "t0": int(patch["t0"]),
                "t1": int(patch["t1"]),
                "x0": int(patch["x0"]),
                "x1": int(patch["x1"]),
            }
        )

    test_gather_metadata = {
        "shot": int(gather_info["shot"]),
        "gx": int(gather_info["gx"]),
        "gy_values": [
            int(value)
            for value in gather_info["gy_values"]
        ],
        "gather_shape": [N_TIME, N_GY],
        "patch_shape": [PATCH_T, PATCH_X],
        "stride": [STRIDE_T, STRIDE_X],
        "t_starts": [int(value) for value in t_starts],
        "x_starts": [int(value) for value in x_starts],
        "number_of_patches": len(patches),
        "primary_observed_columns": int(
            np.sum(gather_mask[0] > 0.5)
        ),
        "delta_observed_columns": int(
            np.sum(gather_mask_delta[0] > 0.5)
        ),
        "delta_removed_columns": int(
            np.sum(
                (gather_mask[0] > 0.5)
                & (gather_mask_delta[0] < 0.5)
            )
        ),
        "combined_observed_columns": int(
            np.sum(gather_mask_delta[0] > 0.5)
        ),
    }

    return test_metadata, test_gather_metadata


def save_json(path, content):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(content, file, indent=2)


def main():
    train_dir = (
        OUT_ROOT
        / "train_exclude_shot1"
        / "repo_style"
    )
    test_dir = (
        OUT_ROOT
        / "test_shot1_gx4220000"
        / "repo_style"
    )

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    train_rng = np.random.default_rng(TRAIN_SEED)
    test_rng = np.random.default_rng(TEST_SEED)

    print("Reading SEG C3 files...", flush=True)
    records = read_seg_c3(RAW_FILES)

    print(
        f"Number of traces read: {len(records)}",
        flush=True,
    )

    print(
        f"Building {N_TIME} x {N_GY} constant-gx gathers...",
        flush=True,
    )
    gathers = build_constant_gx_gathers(records)

    print(
        f"Number of valid gathers: {len(gathers)}",
        flush=True,
    )

    shot_ids = sorted({gather["shot"] for gather in gathers})
    print(f"Shot IDs found: {shot_ids}", flush=True)

    train_result = make_training_split(
        gathers,
        train_dir,
        train_rng,
    )

    test_metadata, test_gather_metadata = make_test_split(
        gathers,
        test_dir,
        test_rng,
    )

    expected_t_starts = make_border_covering_starts(
        N_TIME,
        PATCH_T,
        STRIDE_T,
    )
    expected_x_starts = make_border_covering_starts(
        N_GY,
        PATCH_X,
        STRIDE_X,
    )

    expected_test_patches = (
        len(expected_t_starts) * len(expected_x_starts)
    )

    if expected_test_patches != 21:
        raise RuntimeError(
            f"Expected 21 patches from the configured geometry, "
            f"but obtained {expected_test_patches}. "
            f"t_starts={expected_t_starts}, "
            f"x_starts={expected_x_starts}"
        )

    if len(test_metadata) != 21:
        raise RuntimeError(
            f"Expected 21 saved test patches, "
            f"but found {len(test_metadata)}."
        )

    summary = {
        "dataset": "SEG C3",
        "time_window": {
            "first_sample_index": 0,
            "number_of_samples": N_TIME,
            "sampling_interval_seconds": DT_SECONDS,
            "sample_time_range_seconds": [
                0.0,
                (N_TIME - 1) * DT_SECONDS,
            ],
        },
        "gather_definition": (
            "constant gx gather with columns sorted by gy"
        ),
        "gather_shape": [N_TIME, N_GY],
        "patch_size": [PATCH_T, PATCH_X],
        "stride": [STRIDE_T, STRIDE_X],
        "border_policy": (
            "append the final border-aligned start when the regular "
            "stride does not reach the final sample"
        ),
        "time_patch_starts": expected_t_starts,
        "space_patch_starts": expected_x_starts,
        "train_selection": (
            "all valid constant-gx gathers except shot 1"
        ),
        "test_selection": {
            "shot": TEST_SHOT,
            "gx": TEST_GX,
        },
        "corruption_probability": CORRUPTION_PROBABILITY,
        "delta_probability": DELTA_PROBABILITY,
        "mask_policy": (
            "one primary trace mask per complete gather, followed by "
            "a conditional final mask that removes delta_probability "
            "of the traces surviving the primary mask; both masks are "
            "then patched and saved"
        ),
        "normalization": (
            "independent min-max normalization of each clean patch "
            "to [-1, 1]"
        ),
        "saved_files_per_patch": [
            "normalized clean patch",
            "primary mask",
            "delta mask",
        ],
        "train_gathers": train_result["number_of_gathers"],
        "train_patches": train_result["number_of_patches"],
        "test_gathers": 1,
        "test_patches": len(test_metadata),
        "train_dir": str(train_dir),
        "test_dir": str(test_dir),
        "train_seed": TRAIN_SEED,
        "test_seed": TEST_SEED,
    }

    save_json(
        OUT_ROOT / "summary.json",
        summary,
    )
    save_json(
        OUT_ROOT / "test_metadata_21patches.json",
        test_metadata,
    )
    save_json(
        OUT_ROOT / "test_gather_metadata.json",
        test_gather_metadata,
    )

    print("Dataset creation completed.", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
