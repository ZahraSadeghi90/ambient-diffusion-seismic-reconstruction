#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import glob
import shutil
import random
from typing import List

import numpy as np


GX_FIXED_NPY_DIR = "/spidmdisk/sadeghi/Ambient/march2026/segc3/shots/gx_fixed_npy"

OUT_ROOT = "/spidmdisk/sadeghi/Ambient/may2026/segc3/data/gx_fixed_p70_d10_pergathermask"
TRAIN_OUT_DIR = os.path.join(OUT_ROOT, "train_exclude_shot1")
TEST_OUT_DIR = os.path.join(OUT_ROOT, "test_shot1")

TEST_SHOT = 1

PATCH_SIZE = 128
STRIDE = 64
N_TIME_KEEP = 625

CORRUPTION_PROBABILITY = 0.70
DELTA_PROBABILITY = 0.10

SEED_GLOBAL = 2026
SEED_MASK_BASE = 10000

CLEAN_OUTPUT_FIRST = True

EPS = 1e-12
MIN_ABS_MAX = 1e-8
MIN_RMS = 1e-9


def parse_shot_id(path: str) -> int:
    name = os.path.basename(path)
    m = re.match(r"shot_(\d+)_gx_", name)
    if m is None:
        raise ValueError(f"Cannot parse shot id from filename: {name}")
    return int(m.group(1))


def parse_gx(path: str) -> int:
    name = os.path.basename(path)
    m = re.match(r"shot_\d+_gx_(-?\d+)\.npy", name)
    if m is None:
        raise ValueError(f"Cannot parse gx from filename: {name}")
    return int(m.group(1))


def get_patch_starts(n: int, patch_size: int, stride: int) -> List[int]:
    if n < patch_size:
        return []
    starts = list(range(0, n - patch_size + 1, stride))
    last = n - patch_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def read_gather(path: str) -> np.ndarray:
    x = np.load(path).astype(np.float32)
    x = np.squeeze(x)

    if x.ndim != 2:
        raise ValueError(f"Expected 2D gather, got shape {x.shape} from {path}")

    if x.shape[0] < x.shape[1]:
        x = x.T

    x = x[:N_TIME_KEEP, :]
    return x.astype(np.float32)


def normalize_patch(x_patch_raw: np.ndarray):
    x_patch_raw = x_patch_raw.astype(np.float32)

    patch_min = float(np.min(x_patch_raw))
    patch_max = float(np.max(x_patch_raw))
    denom = patch_max - patch_min

    if denom < EPS:
        return None, patch_min, patch_max

    x_patch = 2.0 * (x_patch_raw - patch_min) / (denom + EPS) - 1.0
    return x_patch.astype(np.float32), patch_min, patch_max


def make_full_gather_masks(nt: int, nx: int, p: float, delta: float, seed: int):
    rng = np.random.default_rng(seed)

    n_keep = int(np.floor(nx * (1.0 - p)))
    n_keep = max(1, min(n_keep, nx))

    keep_idx = np.sort(rng.choice(nx, size=n_keep, replace=False))

    A = np.zeros((nt, nx), dtype=np.float32)
    A[:, keep_idx] = 1.0

    n_remove = int(np.floor(n_keep * delta))

    if n_remove > 0:
        shuffled = keep_idx.copy()
        rng.shuffle(shuffled)
        remove_idx = shuffled[:n_remove]
        keep_tilde_idx = np.setdiff1d(keep_idx, remove_idx, assume_unique=True)
    else:
        keep_tilde_idx = keep_idx.copy()

    Atilde = np.zeros((nt, nx), dtype=np.float32)
    Atilde[:, keep_tilde_idx] = 1.0

    return A, Atilde, keep_idx, keep_tilde_idx


def process_files(files: List[str], out_dir: str, save_metadata: bool, split_name: str):
    repo_dir = os.path.join(out_dir, "repo_style")
    aux_dir = os.path.join(out_dir, "auxiliary")

    os.makedirs(repo_dir, exist_ok=True)
    if save_metadata:
        os.makedirs(aux_dir, exist_ok=True)

    cp_tag = int(round(100 * CORRUPTION_PROBABILITY))
    dp_tag = int(round(100 * DELTA_PROBABILITY))

    records = []
    gather_records = []

    saved = 0
    skipped_small = 0
    skipped_low_energy = 0

    for gather_idx, path in enumerate(files):
        shot_id = parse_shot_id(path)
        gx_val = parse_gx(path)
        x_raw = read_gather(path)

        nt, nx = x_raw.shape

        if nt < PATCH_SIZE or nx < PATCH_SIZE:
            skipped_small += 1
            continue

        A_full, Atilde_full, keep_idx, keep_tilde_idx = make_full_gather_masks(
            nt=nt,
            nx=nx,
            p=CORRUPTION_PROBABILITY,
            delta=DELTA_PROBABILITY,
            seed=SEED_MASK_BASE + gather_idx,
        )

        t_starts = get_patch_starts(nt, PATCH_SIZE, STRIDE)
        x_starts = get_patch_starts(nx, PATCH_SIZE, STRIDE)

        gather_name = os.path.splitext(os.path.basename(path))[0]

        if save_metadata:
            gather_aux_dir = os.path.join(aux_dir, gather_name)
            os.makedirs(gather_aux_dir, exist_ok=True)

            np.save(os.path.join(gather_aux_dir, "x_raw.npy"), x_raw.astype(np.float32))
            np.save(os.path.join(gather_aux_dir, "A_full.npy"), A_full.astype(np.float32))
            np.save(os.path.join(gather_aux_dir, "Atilde_full.npy"), Atilde_full.astype(np.float32))
            np.save(os.path.join(gather_aux_dir, "keep_idx.npy"), keep_idx.astype(np.int32))
            np.save(os.path.join(gather_aux_dir, "keep_tilde_idx.npy"), keep_tilde_idx.astype(np.int32))

        gather_patch_bases = []

        for top in t_starts:
            for left in x_starts:
                x_patch_raw = x_raw[top:top + PATCH_SIZE, left:left + PATCH_SIZE]
                A_patch = A_full[top:top + PATCH_SIZE, left:left + PATCH_SIZE]
                Atilde_patch = Atilde_full[top:top + PATCH_SIZE, left:left + PATCH_SIZE]

                raw_maxabs = float(np.max(np.abs(x_patch_raw)))
                raw_rms = float(np.sqrt(np.mean(x_patch_raw ** 2)))

                if raw_maxabs <= MIN_ABS_MAX or raw_rms <= MIN_RMS:
                    skipped_low_energy += 1
                    continue

                x_patch, patch_min, patch_max = normalize_patch(x_patch_raw)

                if x_patch is None:
                    skipped_low_energy += 1
                    continue

                y_patch = A_patch * x_patch
                ytilde_patch = Atilde_patch * x_patch

                base = f"{saved:08d}"

                np.save(os.path.join(repo_dir, f"{base}_gt.npy"), x_patch[None, :, :].astype(np.float32))
                np.save(os.path.join(repo_dir, f"{base}_y.npy"), y_patch[None, :, :].astype(np.float32))
                np.save(os.path.join(repo_dir, f"{base}_ytilde.npy"), ytilde_patch[None, :, :].astype(np.float32))
                np.save(os.path.join(repo_dir, f"{base}_mask_{cp_tag}.npy"), A_patch[None, :, :].astype(np.float32))
                np.save(os.path.join(repo_dir, f"{base}_mask_delta_{dp_tag}.npy"), Atilde_patch[None, :, :].astype(np.float32))

                if save_metadata:
                    records.append({
                        "file_base": base,
                        "source_file": os.path.basename(path),
                        "source_path": path,
                        "shot_id": int(shot_id),
                        "gx": int(gx_val),
                        "gather_index": int(gather_idx),
                        "top": int(top),
                        "left": int(left),
                        "patch_size": int(PATCH_SIZE),
                        "stride": int(STRIDE),
                        "full_shape": [int(nt), int(nx)],
                        "patch_min": float(patch_min),
                        "patch_max": float(patch_max),
                        "raw_maxabs": float(raw_maxabs),
                        "raw_rms": float(raw_rms),
                        "normalization": "per_patch_minmax_minus1_1_before_masking",
                        "mask_generation": "one_full_mask_per_gather_then_patched",
                    })

                gather_patch_bases.append(base)
                saved += 1

        if save_metadata:
            gather_records.append({
                "source_file": os.path.basename(path),
                "source_path": path,
                "shot_id": int(shot_id),
                "gx": int(gx_val),
                "gather_index": int(gather_idx),
                "shape": [int(nt), int(nx)],
                "n_patches_saved": int(len(gather_patch_bases)),
                "patch_bases": gather_patch_bases,
                "mask_keep_count": int(len(keep_idx)),
                "mask_tilde_keep_count": int(len(keep_tilde_idx)),
                "corruption_probability": float(CORRUPTION_PROBABILITY),
                "delta_probability": float(DELTA_PROBABILITY),
            })

        print(
            f"{split_name} | {gather_idx + 1}/{len(files)} | "
            f"{os.path.basename(path)} | patches saved so far = {saved}"
        )

    summary = {
        "split": split_name,
        "input_dir": GX_FIXED_NPY_DIR,
        "out_dir": out_dir,
        "repo_dir": repo_dir,
        "n_input_files": int(len(files)),
        "n_saved_patches": int(saved),
        "n_skipped_small_gathers": int(skipped_small),
        "n_skipped_low_energy_patches": int(skipped_low_energy),
        "patch_size": int(PATCH_SIZE),
        "stride": int(STRIDE),
        "n_time_keep": int(N_TIME_KEEP),
        "corruption_probability": float(CORRUPTION_PROBABILITY),
        "delta_probability": float(DELTA_PROBABILITY),
        "normalization": "per_patch_minmax_minus1_1_before_masking",
        "mask_generation": "one_full_mask_per_gather_then_patched",
        "saved_metadata": bool(save_metadata),
    }

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    if save_metadata:
        metadata = {
            "summary": summary,
            "gather_records": gather_records,
            "patch_records": records,
        }

        with open(os.path.join(out_dir, "patch_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

    print(f"\nDone {split_name}.")
    print(f"Repo style: {repo_dir}")
    print(f"Saved patches: {saved}")

    return saved


def main():
    random.seed(SEED_GLOBAL)
    np.random.seed(SEED_GLOBAL)

    if CLEAN_OUTPUT_FIRST and os.path.isdir(OUT_ROOT):
        shutil.rmtree(OUT_ROOT)

    os.makedirs(OUT_ROOT, exist_ok=True)

    files = sorted(glob.glob(os.path.join(GX_FIXED_NPY_DIR, "*.npy")))

    if len(files) == 0:
        raise RuntimeError(f"No .npy files found in {GX_FIXED_NPY_DIR}")

    train_files = []
    test_files = []

    for path in files:
        shot_id = parse_shot_id(path)

        if shot_id == TEST_SHOT:
            test_files.append(path)
        else:
            train_files.append(path)

    if len(test_files) == 0:
        raise RuntimeError(f"No files found for TEST_SHOT={TEST_SHOT}")

    print("Dataset split")
    print(f"Total files: {len(files)}")
    print(f"Test shot: {TEST_SHOT}")
    print(f"Test files: {len(test_files)}")
    print(f"Train files: {len(train_files)}")

    train_saved = process_files(
        files=train_files,
        out_dir=TRAIN_OUT_DIR,
        save_metadata=False,
        split_name="train",
    )

    test_saved = process_files(
        files=test_files,
        out_dir=TEST_OUT_DIR,
        save_metadata=True,
        split_name="test",
    )

    split_summary = {
        "test_shot": int(TEST_SHOT),
        "total_files": int(len(files)),
        "train_files": int(len(train_files)),
        "test_files": int(len(test_files)),
        "train_saved_patches": int(train_saved),
        "test_saved_patches": int(test_saved),
        "train_out_dir": TRAIN_OUT_DIR,
        "test_out_dir": TEST_OUT_DIR,
        "leakage_status": f"All files from shot_{TEST_SHOT} are excluded from training.",
    }

    with open(os.path.join(OUT_ROOT, "split_summary.json"), "w") as f:
        json.dump(split_summary, f, indent=2)

    print("\nAll done.")
    print(f"Train directory: {TRAIN_OUT_DIR}")
    print(f"Test directory: {TEST_OUT_DIR}")
    print(f"Split summary: {os.path.join(OUT_ROOT, 'split_summary.json')}")


if __name__ == "__main__":
    main()
