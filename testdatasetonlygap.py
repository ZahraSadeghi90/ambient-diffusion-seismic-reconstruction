#!/usr/bin/env python3

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from obspy.io.segy.segy import _read_segy


DATA_ROOT = Path("/home/zahras90/scratch/ambient/data/segc3")
RAW_FILE = DATA_ROOT / "SEG_45Shot_shots1-9.sgy"

LARGE_GAP_PERCENTAGES = [15, 25, 40, 50, 60, 70]
MISSING_PERCENTAGES = LARGE_GAP_PERCENTAGES
DELTA_PERCENTAGE = 10

PATCH_T = 128
PATCH_X = 128
STRIDE_T = 64
STRIDE_X = 64

N_TIME = 500
N_GY = 201
DT_SECONDS = 0.008

TEST_SHOT = 1
TEST_GX = 4220000

TEST_SEED = 5678
EPS = 1e-8


def get_header_value(header, names):
    for name in names:
        if hasattr(header, name):
            return getattr(header, name)

    raise AttributeError(
        f"None of these SEG-Y header fields exist: {names}"
    )


def load_target_gather():
    if not RAW_FILE.is_file():
        raise FileNotFoundError(
            f"SEG-Y file not found: {RAW_FILE}"
        )

    print(
        f"Reading target gather from: {RAW_FILE}",
        flush=True,
    )

    segy = _read_segy(
        str(RAW_FILE),
        headonly=False,
    )

    traces = []

    for trace_object in segy.traces:
        header = trace_object.header

        shot = int(
            get_header_value(
                header,
                [
                    "energy_source_point_number",
                    "original_field_record_number",
                    "field_record_number",
                ],
            )
        )

        gx = int(
            get_header_value(
                header,
                ["group_coordinate_x"],
            )
        )

        if shot != TEST_SHOT or gx != TEST_GX:
            continue

        gy = int(
            get_header_value(
                header,
                ["group_coordinate_y"],
            )
        )

        trace = np.asarray(
            trace_object.data,
            dtype=np.float32,
        )

        if trace.size < N_TIME:
            raise RuntimeError(
                f"Trace shot={shot}, gx={gx}, gy={gy} contains "
                f"{trace.size} samples; expected at least {N_TIME}."
            )

        traces.append(
            (
                gy,
                trace[:N_TIME].copy(),
            )
        )

    if len(traces) != N_GY:
        raise RuntimeError(
            f"Expected {N_GY} traces for shot={TEST_SHOT}, "
            f"gx={TEST_GX}, but found {len(traces)}."
        )

    traces.sort(
        key=lambda item: item[0]
    )

    gy_values = [
        int(gy)
        for gy, _ in traces
    ]

    if len(set(gy_values)) != N_GY:
        raise RuntimeError(
            f"Expected {N_GY} unique gy values, but found "
            f"{len(set(gy_values))}."
        )

    gather = np.stack(
        [trace for _, trace in traces],
        axis=1,
    ).astype(np.float32)

    expected_shape = (
        N_TIME,
        N_GY,
    )

    if gather.shape != expected_shape:
        raise RuntimeError(
            f"Target gather shape is {gather.shape}; "
            f"expected {expected_shape}."
        )

    print(
        f"Loaded gather shape: {gather.shape}",
        flush=True,
    )

    return gather, gy_values


def make_border_covering_starts(
    length,
    patch_size,
    stride,
):
    if length < patch_size:
        raise ValueError(
            f"Length {length} is smaller than patch size {patch_size}."
        )

    starts = list(
        range(
            0,
            length - patch_size + 1,
            stride,
        )
    )

    final_start = length - patch_size

    if starts[-1] != final_start:
        starts.append(final_start)

    return starts


def normalize_patch_to_minus_one_one(
    raw_patch,
):
    raw_patch = np.asarray(
        raw_patch,
        dtype=np.float32,
    )

    minimum = float(
        np.min(raw_patch)
    )
    maximum = float(
        np.max(raw_patch)
    )

    if not np.isfinite(minimum) or not np.isfinite(maximum):
        raise RuntimeError(
            "A target patch contains non-finite values."
        )

    amplitude_range = maximum - minimum

    if amplitude_range < EPS:
        normalized = np.zeros_like(
            raw_patch,
            dtype=np.float32,
        )
    else:
        normalized = (
            2.0
            * (raw_patch - minimum)
            / amplitude_range
            - 1.0
        ).astype(np.float32)

    return normalized, minimum, maximum


def broadcast_column_mask(
    column_mask,
):
    column_mask = np.asarray(
        column_mask,
        dtype=np.float32,
    )

    if column_mask.shape != (N_GY,):
        raise ValueError(
            f"Column mask shape is {column_mask.shape}; "
            f"expected {(N_GY,)}."
        )

    return np.broadcast_to(
        column_mask[None, :],
        (N_TIME, N_GY),
    ).copy()


def make_large_gap_primary_masks():
    masks = {}

    for missing_percentage in MISSING_PERCENTAGES:
        number_missing = int(
            round(
                missing_percentage
                * N_GY
                / 100.0
            )
        )

        number_missing = min(
            max(number_missing, 1),
            N_GY - 1,
        )

        gap_start = (
            N_GY - number_missing
        ) // 2

        gap_end = (
            gap_start + number_missing
        )

        column_mask = np.ones(
            N_GY,
            dtype=np.float32,
        )

        column_mask[
            gap_start:gap_end
        ] = 0.0

        masks[missing_percentage] = (
            broadcast_column_mask(
                column_mask
            )
        )

        observed = int(
            np.sum(
                column_mask > 0.5
            )
        )

        actual_missing_percentage = (
            100.0
            * number_missing
            / N_GY
        )

        print(
            f"Large-gap primary mask p={missing_percentage}% | "
            f"gap=[{gap_start}:{gap_end}) | "
            f"missing={number_missing} | "
            f"observed={observed} | "
            f"actual missing={actual_missing_percentage:.3f}%",
            flush=True,
        )

    return masks


def make_conditional_delta_mask(
    primary_mask,
    missing_percentage,
):
    primary_mask = np.asarray(
        primary_mask,
        dtype=np.float32,
    )

    if primary_mask.shape != (N_TIME, N_GY):
        raise ValueError(
            f"Primary mask shape is {primary_mask.shape}; "
            f"expected {(N_TIME, N_GY)}."
        )

    reference_row = primary_mask[0]

    if not np.all(
        primary_mask == reference_row[None, :]
    ):
        raise RuntimeError(
            "Primary mask must be constant along time."
        )

    surviving_indices = np.flatnonzero(
        reference_row > 0.5
    )

    if surviving_indices.size == 0:
        raise RuntimeError(
            f"The {missing_percentage}% primary mask has no "
            "surviving traces."
        )

    number_to_remove = int(
        math.ceil(
            DELTA_PERCENTAGE
            * surviving_indices.size
            / 100.0
        )
    )

    number_to_remove = min(
        max(number_to_remove, 0),
        surviving_indices.size,
    )

    rng = np.random.default_rng(
        TEST_SEED
        + 1000
        + missing_percentage
    )

    final_columns = reference_row.copy()

    if number_to_remove > 0:
        removed_indices = rng.choice(
            surviving_indices,
            size=number_to_remove,
            replace=False,
        )

        final_columns[
            removed_indices
        ] = 0.0

    final_mask = broadcast_column_mask(
        final_columns
    )

    if np.any(
        final_mask > primary_mask
    ):
        raise RuntimeError(
            "Final delta mask is not a subset of the primary mask."
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


def extract_patch_records(
    raw_gather,
    primary_mask,
    final_mask,
):
    t_starts = make_border_covering_starts(
        N_TIME,
        PATCH_T,
        STRIDE_T,
    )

    x_starts = make_border_covering_starts(
        N_GY,
        PATCH_X,
        STRIDE_X,
    )

    records = []

    for t0 in t_starts:
        for x0 in x_starts:
            t1 = t0 + PATCH_T
            x1 = x0 + PATCH_X

            raw_patch = raw_gather[
                t0:t1,
                x0:x1,
            ]

            primary_patch = primary_mask[
                t0:t1,
                x0:x1,
            ]

            final_patch = final_mask[
                t0:t1,
                x0:x1,
            ]

            expected_shape = (
                PATCH_T,
                PATCH_X,
            )

            if raw_patch.shape != expected_shape:
                raise RuntimeError(
                    f"Unexpected raw patch shape {raw_patch.shape} "
                    f"at t0={t0}, x0={x0}."
                )

            normalized_patch, minimum, maximum = (
                normalize_patch_to_minus_one_one(
                    raw_patch
                )
            )

            records.append(
                {
                    "clean": normalized_patch,
                    "primary_mask": primary_patch.astype(
                        np.float32
                    ),
                    "final_mask": final_patch.astype(
                        np.float32
                    ),
                    "t0": int(t0),
                    "t1": int(t1),
                    "x0": int(x0),
                    "x1": int(x1),
                    "normalization_min": minimum,
                    "normalization_max": maximum,
                }
            )

    expected_count = (
        len(t_starts)
        * len(x_starts)
    )

    if expected_count != 21:
        raise RuntimeError(
            f"Configured geometry produces {expected_count} patches, "
            "not 21."
        )

    if len(records) != 21:
        raise RuntimeError(
            f"Extracted {len(records)} patches; expected 21."
        )

    return records, t_starts, x_starts


def save_five_patch_examples(
    records,
    test_dir,
    missing_percentage,
):
    selected_indices = np.linspace(
        0,
        len(records) - 1,
        5,
        dtype=int,
    )

    fig, axes = plt.subplots(
        5,
        5,
        figsize=(17, 13),
        constrained_layout=True,
    )

    for column, patch_index in enumerate(
        selected_indices
    ):
        record = records[
            int(patch_index)
        ]

        clean = record["clean"]
        primary_mask = record["primary_mask"]
        final_mask = record["final_mask"]

        primary_corrupted = (
            clean * primary_mask
        )

        final_corrupted = (
            clean * final_mask
        )

        axes[0, column].imshow(
            clean,
            cmap="seismic",
            aspect="auto",
            vmin=-1.0,
            vmax=1.0,
            interpolation="nearest",
        )

        axes[1, column].imshow(
            primary_corrupted,
            cmap="seismic",
            aspect="auto",
            vmin=-1.0,
            vmax=1.0,
            interpolation="nearest",
        )

        axes[2, column].imshow(
            final_corrupted,
            cmap="seismic",
            aspect="auto",
            vmin=-1.0,
            vmax=1.0,
            interpolation="nearest",
        )

        axes[3, column].imshow(
            primary_mask,
            cmap="gray",
            aspect="auto",
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
        )

        axes[4, column].imshow(
            final_mask,
            cmap="gray",
            aspect="auto",
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
        )

        axes[0, column].set_title(
            f"Patch {int(patch_index)}"
        )

        for row in range(5):
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])

    axes[0, 0].set_ylabel(
        "Clean"
    )
    axes[1, 0].set_ylabel(
        f"After primary\np={missing_percentage}%"
    )
    axes[2, 0].set_ylabel(
        f"After further\ncorruption d={DELTA_PERCENTAGE}%"
    )
    axes[3, 0].set_ylabel(
        "Primary mask"
    )
    axes[4, 0].set_ylabel(
        "Final mask"
    )

    fig.suptitle(
        f"SEG C3 test patches | shot={TEST_SHOT}, "
        f"gx={TEST_GX}, p={missing_percentage}%, "
        f"d={DELTA_PERCENTAGE}%",
        fontsize=15,
    )

    output_path = (
        test_dir
        / (
            f"five_patch_examples_"
            f"p{missing_percentage}_"
            f"d{DELTA_PERCENTAGE}.png"
        )
    )

    fig.savefig(
        output_path,
        dpi=250,
        bbox_inches="tight",
    )

    plt.close(fig)

    if not output_path.is_file():
        raise RuntimeError(
            f"Figure was not created: {output_path}"
        )

    print(
        f"  Figure saved: {output_path}",
        flush=True,
    )


def save_percentage_dataset(
    raw_gather,
    gy_values,
    missing_percentage,
    primary_mask,
    final_mask,
):
    out_root = (
        DATA_ROOT
        / (
            f"gx_fixed_largegap_p{missing_percentage}_"
            f"d{DELTA_PERCENTAGE}_test"
        )
    )

    test_dir = (
        out_root
        / f"test_shot{TEST_SHOT}_gx{TEST_GX}"
        / "repo_style"
    )

    test_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    records, t_starts, x_starts = (
        extract_patch_records(
            raw_gather,
            primary_mask,
            final_mask,
        )
    )

    metadata = []

    for index, record in enumerate(records):
        stem = f"{index:08d}"

        clean_filename = (
            f"{stem}.npy"
        )

        primary_filename = (
            f"{stem}_mask_{missing_percentage}.npy"
        )

        final_filename = (
            f"{stem}_mask_delta_{DELTA_PERCENTAGE}.npy"
        )

        np.save(
            test_dir / clean_filename,
            record["clean"],
        )

        np.save(
            test_dir / primary_filename,
            record["primary_mask"],
        )

        np.save(
            test_dir / final_filename,
            record["final_mask"],
        )

        metadata.append(
            {
                "index": int(index),
                "filename": clean_filename,
                "mask_filename": primary_filename,
                "mask_delta_filename": final_filename,
                "t0": record["t0"],
                "t1": record["t1"],
                "x0": record["x0"],
                "x1": record["x1"],
                "normalization_min": (
                    record["normalization_min"]
                ),
                "normalization_max": (
                    record["normalization_max"]
                ),
            }
        )

    primary_observed = int(
        np.sum(
            primary_mask[0] > 0.5
        )
    )

    final_observed = int(
        np.sum(
            final_mask[0] > 0.5
        )
    )

    primary_missing = (
        N_GY - primary_observed
    )

    delta_removed = (
        primary_observed - final_observed
    )

    final_missing = (
        N_GY - final_observed
    )

    actual_primary_missing_percentage = (
        100.0
        * primary_missing
        / N_GY
    )

    actual_delta_percentage_of_survivors = (
        100.0
        * delta_removed
        / primary_observed
    )

    actual_final_missing_percentage = (
        100.0
        * final_missing
        / N_GY
    )

    theoretical_final_missing_percentage = (
        missing_percentage
        + (
            100.0 - missing_percentage
        )
        * DELTA_PERCENTAGE
        / 100.0
    )

    gather_metadata = {
        "dataset": "SEG C3",
        "split": "test only",
        "shot": TEST_SHOT,
        "gx": TEST_GX,
        "gy_values": gy_values,
        "gather_shape": [
            N_TIME,
            N_GY,
        ],
        "patch_shape": [
            PATCH_T,
            PATCH_X,
        ],
        "stride": [
            STRIDE_T,
            STRIDE_X,
        ],
        "t_starts": t_starts,
        "x_starts": x_starts,
        "number_of_patches": len(records),
        "requested_primary_missing_percentage": (
            missing_percentage
        ),
        "primary_missing_columns": (
            primary_missing
        ),
        "primary_observed_columns": (
            primary_observed
        ),
        "actual_primary_missing_percentage": (
            actual_primary_missing_percentage
        ),
        "requested_delta_percentage_of_survivors": (
            DELTA_PERCENTAGE
        ),
        "delta_removed_columns": (
            delta_removed
        ),
        "actual_delta_percentage_of_survivors": (
            actual_delta_percentage_of_survivors
        ),
        "final_missing_columns": (
            final_missing
        ),
        "final_observed_columns": (
            final_observed
        ),
        "actual_final_missing_percentage": (
            actual_final_missing_percentage
        ),
        "theoretical_final_missing_percentage": (
            theoretical_final_missing_percentage
        ),
        "normalization": (
            "Independent clean-patch min-max normalization "
            "to [-1, 1]."
        ),
        "large_gap_start_index": (
            int(
                np.flatnonzero(
                    primary_mask[0] < 0.5
                )[0]
            )
        ),
        "large_gap_end_index_exclusive": (
            int(
                np.flatnonzero(
                    primary_mask[0] < 0.5
                )[-1]
                + 1
            )
        ),
        "mask_policy": (
            "The primary mask contains one deterministic contiguous "
            "gap centered along the receiver axis. Then "
            "DELTA_PERCENTAGE of the traces surviving the primary "
            "large-gap mask are removed randomly. The saved "
            "mask_delta file is the final combined mask."
        ),
        "raw_file": str(RAW_FILE),
        "test_dir": str(test_dir),
        "test_seed": TEST_SEED,
    }

    with (
        out_root
        / "test_metadata_21patches.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )

    with (
        out_root
        / "test_gather_metadata.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            gather_metadata,
            file,
            indent=2,
        )

    with (
        out_root
        / "summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            gather_metadata,
            file,
            indent=2,
        )

    save_five_patch_examples(
        records,
        test_dir,
        missing_percentage,
    )

    print(
        f"\nCreated p={missing_percentage}%, "
        f"d={DELTA_PERCENTAGE}% test dataset:",
        flush=True,
    )

    print(
        f"  Folder: {test_dir}",
        flush=True,
    )

    print(
        f"  Patches: {len(records)}",
        flush=True,
    )

    print(
        f"  Primary mask: "
        f"missing={primary_missing}, "
        f"observed={primary_observed}, "
        f"actual missing="
        f"{actual_primary_missing_percentage:.3f}%",
        flush=True,
    )

    print(
        f"  Further corruption: "
        f"removed={delta_removed} from "
        f"{primary_observed} surviving traces, "
        f"actual="
        f"{actual_delta_percentage_of_survivors:.3f}%",
        flush=True,
    )

    print(
        f"  Final mask: "
        f"missing={final_missing}, "
        f"observed={final_observed}, "
        f"actual missing="
        f"{actual_final_missing_percentage:.3f}%",
        flush=True,
    )

    print(
        f"  Theoretical final missing percentage: "
        f"{theoretical_final_missing_percentage:.3f}%",
        flush=True,
    )


def main():
    raw_gather, gy_values = (
        load_target_gather()
    )

    primary_masks = (
        make_large_gap_primary_masks()
    )

    for missing_percentage in MISSING_PERCENTAGES:
        primary_mask = (
            primary_masks[
                missing_percentage
            ]
        )

        final_mask = (
            make_conditional_delta_mask(
                primary_mask,
                missing_percentage,
            )
        )

        save_percentage_dataset(
            raw_gather,
            gy_values,
            missing_percentage,
            primary_mask,
            final_mask,
        )

    print(
        "\nFinished creating all six large-gap test-only datasets.",
        flush=True,
    )


if __name__ == "__main__":
    main()
