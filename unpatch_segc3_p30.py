#!/usr/bin/env python3

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from obspy.io.segy.segy import _read_segy


DATA_ROOT = Path("/home/zahras90/scratch/ambient/data/segc3")

RAW_FILE = DATA_ROOT / "SEG_45Shot_shots1-9.sgy"

PATCH_ROOT = DATA_ROOT / "gx_fixed_largegap_p40_d10_test"
PATCH_DIR = PATCH_ROOT / "test_shot1_gx4220000" / "repo_style"
METADATA_FILE = PATCH_ROOT / "test_metadata_21patches.json"

RECON_ROOT = Path(
    "/home/zahras90/scratch/ambient/inverse/inverse_segc3_128_70on40gap_pkl10000A/cp40_dp=10/cp=40_dp=10"
)
OUTPUT_DIR = RECON_ROOT / "unpatched"

PATCH_T = 128
PATCH_X = 128
GATHER_T = 500
GATHER_X = 201
DT_SECONDS = 0.008

TEST_SHOT = 1
TEST_GX = 4220000

CORRUPTION_PERCENT =40
DELTA_PERCENT = 10

EPS = 1e-12
CLEAN_VALIDATION_TOLERANCE = 1e-5
MASK_BINARY_TOLERANCE = 1e-6

APPLY_HARD_DATA_CONSISTENCY = True
HARD_DATA_CONSISTENCY_TOLERANCE = 1e-6
WINDOW_FLOOR = 1e-3


def get_header_value(header, names):
    for name in names:
        if hasattr(header, name):
            return getattr(header, name)

    raise AttributeError(
        f"None of these SEG-Y header fields exist: {names}"
    )


def load_raw_test_gather():
    if not RAW_FILE.is_file():
        raise FileNotFoundError(
            f"SEG-Y file does not exist: {RAW_FILE}"
        )

    print(f"Reading raw test gather from: {RAW_FILE}", flush=True)
    segy = _read_segy(str(RAW_FILE), headonly=False)

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

        if trace.size < GATHER_T:
            raise RuntimeError(
                f"Trace shot={shot}, gx={gx}, gy={gy} contains "
                f"{trace.size} samples; expected at least {GATHER_T}."
            )

        traces.append(
            (
                gy,
                trace[:GATHER_T].copy(),
            )
        )

    if len(traces) != GATHER_X:
        raise RuntimeError(
            f"Expected {GATHER_X} traces for shot={TEST_SHOT}, "
            f"gx={TEST_GX}, but found {len(traces)}."
        )

    traces = sorted(
        traces,
        key=lambda item: item[0],
    )
    gy_values = [
        int(item[0])
        for item in traces
    ]

    if len(set(gy_values)) != GATHER_X:
        raise RuntimeError(
            f"Expected {GATHER_X} unique gy values, but found "
            f"{len(set(gy_values))}."
        )

    gather = np.stack(
        [item[1] for item in traces],
        axis=1,
    ).astype(np.float32)

    expected_shape = (
        GATHER_T,
        GATHER_X,
    )

    if gather.shape != expected_shape:
        raise RuntimeError(
            f"Unexpected raw gather shape {gather.shape}; "
            f"expected {expected_shape}."
        )

    return gather, gy_values


def load_metadata():
    if not METADATA_FILE.is_file():
        raise FileNotFoundError(
            f"Metadata file does not exist: {METADATA_FILE}"
        )

    with METADATA_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    if not isinstance(metadata, list):
        raise RuntimeError(
            f"Metadata must be a list: {METADATA_FILE}"
        )

    if len(metadata) != 21:
        raise RuntimeError(
            f"Expected exactly 21 metadata entries, "
            f"but found {len(metadata)}."
        )

    metadata = sorted(
        metadata,
        key=lambda item: int(item["index"]),
    )

    for expected_index, item in enumerate(metadata):
        actual_index = int(item["index"])

        if actual_index != expected_index:
            raise RuntimeError(
                f"Metadata indices are not consecutive: expected "
                f"{expected_index}, found {actual_index}."
            )

        for key in (
            "t0",
            "t1",
            "x0",
            "x1",
        ):
            if key not in item:
                raise KeyError(
                    f"Metadata entry {actual_index} is missing '{key}'."
                )

        t0 = int(item["t0"])
        t1 = int(item["t1"])
        x0 = int(item["x0"])
        x1 = int(item["x1"])

        if t1 - t0 != PATCH_T:
            raise RuntimeError(
                f"Metadata entry {actual_index} has invalid time size: "
                f"t0={t0}, t1={t1}."
            )

        if x1 - x0 != PATCH_X:
            raise RuntimeError(
                f"Metadata entry {actual_index} has invalid trace size: "
                f"x0={x0}, x1={x1}."
            )

        if (
            t0 < 0
            or x0 < 0
            or t1 > GATHER_T
            or x1 > GATHER_X
        ):
            raise RuntimeError(
                f"Metadata entry {actual_index} is outside the gather: "
                f"t0={t0}, t1={t1}, x0={x0}, x1={x1}."
            )

    expected_coordinates = [
        (t0, x0)
        for t0 in [0, 64, 128, 192, 256, 320, 372]
        for x0 in [0, 64, 73]
    ]
    actual_coordinates = [
        (
            int(item["t0"]),
            int(item["x0"]),
        )
        for item in metadata
    ]

    if actual_coordinates != expected_coordinates:
        raise RuntimeError(
            "Metadata patch coordinates do not match the expected "
            "500 x 201 patch geometry.\n"
            f"Expected: {expected_coordinates}\n"
            f"Found: {actual_coordinates}"
        )

    return metadata


def squeeze_patch(array, source_name):
    array = np.squeeze(
        np.asarray(
            array,
            dtype=np.float32,
        )
    )

    expected_shape = (
        PATCH_T,
        PATCH_X,
    )

    if array.shape != expected_shape:
        raise RuntimeError(
            f"Unexpected patch shape in {source_name}: {array.shape}; "
            f"expected {expected_shape}."
        )

    if not np.all(np.isfinite(array)):
        raise RuntimeError(
            f"Non-finite values found in: {source_name}"
        )

    return array


def normalize_raw_patch(raw_patch):
    minimum = float(
        np.min(raw_patch)
    )
    maximum = float(
        np.max(raw_patch)
    )

    if not np.isfinite(minimum) or not np.isfinite(maximum):
        raise RuntimeError(
            "The raw reference patch contains non-finite values."
        )

    amplitude_range = maximum - minimum

    if amplitude_range <= EPS:
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


def denormalize_patch(
    normalized_patch,
    minimum,
    maximum,
):
    minimum = float(minimum)
    maximum = float(maximum)

    if maximum < minimum:
        raise ValueError(
            f"Invalid normalization range: "
            f"min={minimum}, max={maximum}."
        )

    amplitude_range = maximum - minimum

    if amplitude_range <= EPS:
        return np.full_like(
            normalized_patch,
            minimum,
            dtype=np.float32,
        )

    return (
        0.5
        * (normalized_patch + 1.0)
        * amplitude_range
        + minimum
    ).astype(np.float32)


def load_clean_patch(index):
    path = PATCH_DIR / f"{index:08d}.npy"

    if not path.is_file():
        raise FileNotFoundError(
            f"Clean patch does not exist: {path}"
        )

    return squeeze_patch(
        np.load(
            path,
            allow_pickle=False,
        ),
        str(path),
    )


def load_primary_mask(index):
    path = (
        PATCH_DIR
        / f"{index:08d}_mask_{CORRUPTION_PERCENT}.npy"
    )

    if not path.is_file():
        raise FileNotFoundError(
            f"Primary mask does not exist: {path}"
        )

    mask = squeeze_patch(
        np.load(
            path,
            allow_pickle=False,
        ),
        str(path),
    )

    distance_to_binary = np.minimum(
        np.abs(mask),
        np.abs(mask - 1.0),
    )

    if float(np.max(distance_to_binary)) > MASK_BINARY_TOLERANCE:
        raise RuntimeError(
            f"Mask contains non-binary values: {path}"
        )

    return (
        mask >= 0.5
    ).astype(np.float32)


def extract_sample_index(path):
    match = re.fullmatch(
        r"sample(\d+)",
        path.name,
    )

    if match is None:
        return None

    return int(match.group(1))


def find_sample_directories():
    if not RECON_ROOT.is_dir():
        raise NotADirectoryError(
            f"Reconstruction directory does not exist: {RECON_ROOT}"
        )

    sample_dirs = []

    for path in RECON_ROOT.iterdir():
        if not path.is_dir():
            continue

        sample_index = extract_sample_index(path)

        if sample_index is not None:
            sample_dirs.append(
                (
                    sample_index,
                    path,
                )
            )

    sample_dirs.sort(
        key=lambda item: item[0],
    )

    if len(sample_dirs) != 21:
        raise RuntimeError(
            f"Expected exactly 21 sampleNNN directories below "
            f"{RECON_ROOT}, but found {len(sample_dirs)}."
        )

    expected_indices = list(range(21))
    actual_indices = [
        index
        for index, _ in sample_dirs
    ]

    if actual_indices != expected_indices:
        raise RuntimeError(
            f"Sample directory indices are incorrect.\n"
            f"Expected: {expected_indices}\n"
            f"Found: {actual_indices}"
        )

    return sample_dirs


def find_reconstructed_patch(sample_dir):
    recon_files = sorted(
        sample_dir.rglob("recon.npy")
    )

    if len(recon_files) != 1:
        found = "\n".join(
            str(path)
            for path in recon_files
        )

        raise RuntimeError(
            f"Expected exactly one recon.npy below {sample_dir}, "
            f"but found {len(recon_files)}.\n{found}"
        )

    path = recon_files[0]

    patch = squeeze_patch(
        np.load(
            path,
            allow_pickle=False,
        ),
        str(path),
    )

    return patch, path


def make_sqrt_hann_window():
    time_window = np.sqrt(
        np.hanning(PATCH_T).astype(
            np.float64
        )
    )
    trace_window = np.sqrt(
        np.hanning(PATCH_X).astype(
            np.float64
        )
    )

    window = np.outer(
        time_window,
        trace_window,
    )

    return np.maximum(
        window,
        float(WINDOW_FLOOR),
    ).astype(np.float64)


def add_patch(
    accumulator,
    weight_sum,
    patch,
    t0,
    x0,
    window,
):
    t1 = t0 + PATCH_T
    x1 = x0 + PATCH_X

    accumulator[
        t0:t1,
        x0:x1,
    ] += patch * window

    weight_sum[
        t0:t1,
        x0:x1,
    ] += window


def finalize_weighted_average(
    accumulator,
    weight_sum,
    name,
):
    if np.any(weight_sum <= 0):
        missing_count = int(
            np.sum(weight_sum <= 0)
        )

        raise RuntimeError(
            f"{name} has {missing_count} samples "
            f"with zero accumulated weight."
        )

    return (
        accumulator / weight_sum
    ).astype(np.float32)


def snr_db(reference, estimate):
    reference = np.asarray(
        reference,
        dtype=np.float64,
    )
    estimate = np.asarray(
        estimate,
        dtype=np.float64,
    )

    signal_power = float(
        np.sum(reference**2)
    )
    error_power = float(
        np.sum(
            (reference - estimate) ** 2
        )
    )

    if signal_power <= 0:
        raise RuntimeError(
            "Reference energy is zero; SNR is undefined."
        )

    if error_power <= EPS:
        return float("inf")

    return float(
        10.0
        * np.log10(
            signal_power / error_power
        )
    )


def missing_only_snr_db(
    reference,
    estimate,
    observed_mask,
):
    missing = observed_mask < 0.5

    if not np.any(missing):
        return float("nan")

    reference_missing = np.asarray(
        reference[missing],
        dtype=np.float64,
    )
    estimate_missing = np.asarray(
        estimate[missing],
        dtype=np.float64,
    )

    signal_power = float(
        np.sum(reference_missing**2)
    )
    error_power = float(
        np.sum(
            (
                reference_missing
                - estimate_missing
            )
            ** 2
        )
    )

    if signal_power <= 0:
        return float("nan")

    if error_power <= EPS:
        return float("inf")

    return float(
        10.0
        * np.log10(
            signal_power / error_power
        )
    )


def robust_symmetric_limit(
    array,
    percentile=99.5,
):
    limit = float(
        np.percentile(
            np.abs(
                np.asarray(
                    array,
                    dtype=np.float64,
                )
            ),
            percentile,
        )
    )

    if not np.isfinite(limit) or limit <= 0:
        return 1.0

    return limit


def save_comparison_plot(
    reference,
    corrupted,
    reconstructed,
    residual,
    corrupted_snr,
    reconstructed_snr,
):
    amplitude_limit = robust_symmetric_limit(
        reference
    )

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(20, 7),
        constrained_layout=True,
    )

    panels = [
        (
            reference,
            "Reference",
        ),
        (
            corrupted,
            f"Corrupted\nSNR = {corrupted_snr:.3f} dB",
        ),
        (
            reconstructed,
            f"Reconstructed\nSNR = {reconstructed_snr:.3f} dB",
        ),
        (
            residual,
            "Residual: reference - reconstructed",
        ),
    ]

    image = None

    for axis, (
        data,
        title,
    ) in zip(axes, panels):
        image = axis.imshow(
            data,
            cmap="seismic",
            aspect="auto",
            vmin=-amplitude_limit,
            vmax=amplitude_limit,
            interpolation="nearest",
            extent=[
                0,
                GATHER_X - 1,
                (GATHER_T - 1) * DT_SECONDS,
                0,
            ],
        )
        axis.set_title(title)
        axis.set_xlabel("Trace index")
        axis.set_ylabel("Time (s)")

    colorbar = fig.colorbar(
        image,
        ax=axes,
        location="right",
        fraction=0.025,
        pad=0.02,
    )
    colorbar.set_label("Amplitude")

    fig.savefig(
        OUTPUT_DIR / "comparison.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_individual_plot(
    data,
    title,
    filename,
    limit,
):
    fig, axis = plt.subplots(
        1,
        1,
        figsize=(7, 9),
        constrained_layout=True,
    )

    image = axis.imshow(
        data,
        cmap="seismic",
        aspect="auto",
        vmin=-limit,
        vmax=limit,
        interpolation="nearest",
        extent=[
            0,
            GATHER_X - 1,
            (GATHER_T - 1) * DT_SECONDS,
            0,
        ],
    )

    axis.set_title(title)
    axis.set_xlabel("Trace index")
    axis.set_ylabel("Time (s)")

    fig.colorbar(
        image,
        ax=axis,
        fraction=0.046,
        pad=0.04,
    )

    fig.savefig(
        OUTPUT_DIR / filename,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"Patch directory: {PATCH_DIR}", flush=True)
    print(f"Metadata file: {METADATA_FILE}", flush=True)
    print(f"Reconstruction directory: {RECON_ROOT}", flush=True)
    print(f"Output directory: {OUTPUT_DIR}", flush=True)

    metadata = load_metadata()
    sample_dirs = find_sample_directories()
    reference, gy_values = load_raw_test_gather()

    window = make_sqrt_hann_window()

    reconstructed_accumulator = np.zeros(
        (
            GATHER_T,
            GATHER_X,
        ),
        dtype=np.float64,
    )
    reconstructed_weight_sum = np.zeros_like(
        reconstructed_accumulator
    )

    mask_accumulator = np.zeros_like(
        reconstructed_accumulator
    )
    mask_weight_sum = np.zeros_like(
        reconstructed_accumulator
    )

    patch_records = []

    for index, sample_dir in sample_dirs:
        meta = metadata[index]

        t0 = int(meta["t0"])
        t1 = int(meta["t1"])
        x0 = int(meta["x0"])
        x1 = int(meta["x1"])

        raw_reference_patch = reference[
            t0:t1,
            x0:x1,
        ]

        expected_clean_normalized, minimum, maximum = (
            normalize_raw_patch(
                raw_reference_patch
            )
        )

        saved_clean_normalized = load_clean_patch(
            index
        )

        clean_validation_error = float(
            np.max(
                np.abs(
                    saved_clean_normalized
                    - expected_clean_normalized
                )
            )
        )

        if (
            clean_validation_error
            > CLEAN_VALIDATION_TOLERANCE
        ):
            raise RuntimeError(
                f"Saved clean patch {index} does not match "
                f"the raw SEG-Y patch normalization. "
                f"Maximum absolute difference: "
                f"{clean_validation_error}."
            )

        primary_mask = load_primary_mask(
            index
        )

        reconstructed_normalized, reconstruction_file = (
            find_reconstructed_patch(
                sample_dir
            )
        )

        reconstructed_patch = denormalize_patch(
            reconstructed_normalized,
            minimum,
            maximum,
        )

        add_patch(
            reconstructed_accumulator,
            reconstructed_weight_sum,
            reconstructed_patch,
            t0,
            x0,
            window,
        )

        add_patch(
            mask_accumulator,
            mask_weight_sum,
            primary_mask,
            t0,
            x0,
            window,
        )

        patch_records.append(
            {
                "index": int(index),
                "sample_directory": str(sample_dir),
                "reconstruction_file": str(
                    reconstruction_file
                ),
                "t0": t0,
                "t1": t1,
                "x0": x0,
                "x1": x1,
                "normalization_min": minimum,
                "normalization_max": maximum,
                "clean_validation_max_abs_difference": (
                    clean_validation_error
                ),
            }
        )

        print(
            f"Processed patch {index:02d}: "
            f"{reconstruction_file}",
            flush=True,
        )

    reconstructed_before_hard_consistency = finalize_weighted_average(
        reconstructed_accumulator,
        reconstructed_weight_sum,
        "Reconstructed gather",
    )

    observed_mask_float = finalize_weighted_average(
        mask_accumulator,
        mask_weight_sum,
        "Observed mask",
    )

    binary_distance = np.minimum(
        np.abs(observed_mask_float),
        np.abs(observed_mask_float - 1.0),
    )
    maximum_mask_disagreement = float(
        np.max(binary_distance)
    )

    if maximum_mask_disagreement > MASK_BINARY_TOLERANCE:
        raise RuntimeError(
            "Overlapping mask patches are inconsistent. "
            f"Maximum distance from a binary value: "
            f"{maximum_mask_disagreement}."
        )

    observed_mask = (
        observed_mask_float >= 0.5
    ).astype(np.float32)

    corrupted = (
        reference * observed_mask
    ).astype(np.float32)

    observed_locations = observed_mask >= 0.5

    if APPLY_HARD_DATA_CONSISTENCY:
        reconstructed = np.where(
            observed_locations,
            corrupted,
            reconstructed_before_hard_consistency,
        ).astype(np.float32)
    else:
        reconstructed = (
            reconstructed_before_hard_consistency.copy()
        )

    if np.any(observed_locations):
        maximum_observed_error_before_hard_consistency = float(
            np.max(
                np.abs(
                    reconstructed_before_hard_consistency[
                        observed_locations
                    ]
                    - corrupted[observed_locations]
                )
            )
        )
        maximum_observed_error_after_hard_consistency = float(
            np.max(
                np.abs(
                    reconstructed[observed_locations]
                    - corrupted[observed_locations]
                )
            )
        )
    else:
        maximum_observed_error_before_hard_consistency = float(
            "nan"
        )
        maximum_observed_error_after_hard_consistency = float(
            "nan"
        )

    if (
        APPLY_HARD_DATA_CONSISTENCY
        and maximum_observed_error_after_hard_consistency
        > HARD_DATA_CONSISTENCY_TOLERANCE
    ):
        raise RuntimeError(
            "Hard data consistency failed. Maximum observed-data "
            f"error is "
            f"{maximum_observed_error_after_hard_consistency}."
        )

    residual_before_hard_consistency = (
        reference - reconstructed_before_hard_consistency
    ).astype(np.float32)

    residual = (
        reference - reconstructed
    ).astype(np.float32)

    corrupted_residual = (
        reference - corrupted
    ).astype(np.float32)

    corrupted_snr = snr_db(
        reference,
        corrupted,
    )
    reconstructed_snr_before_hard_consistency = snr_db(
        reference,
        reconstructed_before_hard_consistency,
    )
    reconstructed_snr = snr_db(
        reference,
        reconstructed,
    )

    corrupted_missing_snr = missing_only_snr_db(
        reference,
        corrupted,
        observed_mask,
    )
    reconstructed_missing_snr_before_hard_consistency = (
        missing_only_snr_db(
            reference,
            reconstructed_before_hard_consistency,
            observed_mask,
        )
    )
    reconstructed_missing_snr = missing_only_snr_db(
        reference,
        reconstructed,
        observed_mask,
    )

    np.save(
        OUTPUT_DIR / "reference_gather.npy",
        reference,
    )
    np.save(
        OUTPUT_DIR / "corrupted_gather.npy",
        corrupted,
    )
    np.save(
        OUTPUT_DIR
        / "reconstructed_gather_before_hard_consistency.npy",
        reconstructed_before_hard_consistency,
    )
    np.save(
        OUTPUT_DIR / "reconstructed_gather.npy",
        reconstructed,
    )
    np.save(
        OUTPUT_DIR
        / (
            "residual_reference_minus_reconstructed_"
            "before_hard_consistency.npy"
        ),
        residual_before_hard_consistency,
    )
    np.save(
        OUTPUT_DIR
        / "residual_reference_minus_reconstructed.npy",
        residual,
    )
    np.save(
        OUTPUT_DIR
        / "residual_reference_minus_corrupted.npy",
        corrupted_residual,
    )
    np.save(
        OUTPUT_DIR / "observed_mask.npy",
        observed_mask,
    )
    np.save(
        OUTPUT_DIR / "overlap_weight_sum.npy",
        reconstructed_weight_sum.astype(
            np.float32
        ),
    )

    amplitude_limit = robust_symmetric_limit(
        reference
    )
    residual_limit = robust_symmetric_limit(
        residual
    )

    save_comparison_plot(
        reference,
        corrupted,
        reconstructed,
        residual,
        corrupted_snr,
        reconstructed_snr,
    )
    save_individual_plot(
        reference,
        "Reference",
        "reference.png",
        amplitude_limit,
    )
    save_individual_plot(
        corrupted,
        f"Corrupted | SNR = {corrupted_snr:.3f} dB",
        "corrupted.png",
        amplitude_limit,
    )
    save_individual_plot(
        reconstructed_before_hard_consistency,
        (
            "Reconstructed before hard consistency | "
            f"SNR = "
            f"{reconstructed_snr_before_hard_consistency:.3f} dB"
        ),
        "reconstructed_before_hard_consistency.png",
        amplitude_limit,
    )
    save_individual_plot(
        reconstructed,
        (
            "Reconstructed after hard consistency | "
            f"SNR = {reconstructed_snr:.3f} dB"
        ),
        "reconstructed.png",
        amplitude_limit,
    )
    save_individual_plot(
        residual,
        "Residual: reference - reconstructed",
        "residual.png",
        residual_limit,
    )

    results = {
        "snr_formula": (
            "10*log10(||S||_F^2 / ||S-S_inter||_F^2)"
        ),
        "reference_shape": list(reference.shape),
        "sampling_interval_seconds": DT_SECONDS,
        "test_shot": TEST_SHOT,
        "test_gx": TEST_GX,
        "gy_values": gy_values,
        "corruption_percent": CORRUPTION_PERCENT,
        "delta_percent": DELTA_PERCENT,
        "corrupted_snr_db": corrupted_snr,
        "reconstructed_snr_before_hard_consistency_db": (
            reconstructed_snr_before_hard_consistency
        ),
        "reconstructed_snr_db": reconstructed_snr,
        "corrupted_missing_only_snr_db": (
            corrupted_missing_snr
        ),
        (
            "reconstructed_missing_only_snr_"
            "before_hard_consistency_db"
        ): reconstructed_missing_snr_before_hard_consistency,
        "reconstructed_missing_only_snr_db": (
            reconstructed_missing_snr
        ),
        "apply_hard_data_consistency": (
            APPLY_HARD_DATA_CONSISTENCY
        ),
        "hard_data_consistency_tolerance": (
            HARD_DATA_CONSISTENCY_TOLERANCE
        ),
        (
            "maximum_observed_error_"
            "before_hard_consistency"
        ): maximum_observed_error_before_hard_consistency,
        (
            "maximum_observed_error_"
            "after_hard_consistency"
        ): maximum_observed_error_after_hard_consistency,
        "window": "separable sqrt-Hann",
        "window_floor": WINDOW_FLOOR,
        "number_of_patches": len(metadata),
        "maximum_mask_overlap_disagreement": (
            maximum_mask_disagreement
        ),
        "raw_file": str(RAW_FILE),
        "metadata_file": str(METADATA_FILE),
        "patch_directory": str(PATCH_DIR),
        "reconstruction_root": str(RECON_ROOT),
        "output_directory": str(OUTPUT_DIR),
        "patches": patch_records,
    }

    with (
        OUTPUT_DIR / "metrics_and_files.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=2,
        )

    print("\nUnpatching completed.", flush=True)
    print(
        f"Corrupted full-gather SNR: "
        f"{corrupted_snr:.6f} dB",
        flush=True,
    )
    print(
        f"Reconstructed full-gather SNR before hard consistency: "
        f"{reconstructed_snr_before_hard_consistency:.6f} dB",
        flush=True,
    )
    print(
        f"Reconstructed full-gather SNR after hard consistency: "
        f"{reconstructed_snr:.6f} dB",
        flush=True,
    )
    print(
        f"Corrupted missing-only SNR: "
        f"{corrupted_missing_snr:.6f} dB",
        flush=True,
    )
    print(
        f"Reconstructed missing-only SNR before hard consistency: "
        f"{reconstructed_missing_snr_before_hard_consistency:.6f} dB",
        flush=True,
    )
    print(
        f"Reconstructed missing-only SNR after hard consistency: "
        f"{reconstructed_missing_snr:.6f} dB",
        flush=True,
    )
    print(
        f"Maximum observed-data error before hard consistency: "
        f"{maximum_observed_error_before_hard_consistency:.6e}",
        flush=True,
    )
    print(
        f"Maximum observed-data error after hard consistency: "
        f"{maximum_observed_error_after_hard_consistency:.6e}",
        flush=True,
    )
    print(
        f"Results saved in: {OUTPUT_DIR}",
        flush=True,
    )


if __name__ == "__main__":
    main()
