#!/usr/bin/env python3
"""Create tightly-scoped ARM64 VC3D release binaries for a bounds ablation.

This does not build PR #1264. It copies the official 05ff9ea macOS-arm64
release dylibs and changes only the immediate in eight verified instructions:

    sub Wbound, Wdimension, #2  ->  sub Wbound, Wdimension, #1

The helper-only variant patches the two Vec3f loc_valid comparisons in
libvc_core. The full-bounds variant additionally patches the six stale copies
in SurfTrackerData::lookup_int, valid_int, and lookup_int_loc.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Patch:
    offset: int
    before: bytes
    after: bytes
    symbol: str
    instruction_before: str
    instruction_after: str


CORE_SHA256 = "1affb73fe548864e2f7a0a8afc9ddad1a52b5f3aaa17482d6a95e0fb6ede5032"
TRACER_SHA256 = "9f8e24d50fcbb8f94e9600caa861f86699799867fd148a97ebd66165b9d86235"
CORE_CODE_OFFSET = 24_008
CORE_CODE_SIZE = 0x121C64
TRACER_CODE_OFFSET = 28_224
TRACER_CODE_SIZE = 0x90060

CORE_PATCHES = (
    Patch(
        0xDFD2C,
        bytes.fromhex("29090051"),
        bytes.fromhex("29050051"),
        "loc_valid(cv::Mat_<cv::Vec3f> const&, cv::Vec2d const&)",
        "sub w9, w9, #2",
        "sub w9, w9, #1",
    ),
    Patch(
        0xDFD48,
        bytes.fromhex("6b090051"),
        bytes.fromhex("6b050051"),
        "loc_valid(cv::Mat_<cv::Vec3f> const&, cv::Vec2d const&)",
        "sub w11, w11, #2",
        "sub w11, w11, #1",
    ),
)

TRACER_PATCHES = (
    Patch(
        0x8E58C,
        bytes.fromhex("a90a0051"),
        bytes.fromhex("a9060051"),
        "SurfTrackerData::lookup_int",
        "sub w9, w21, #2",
        "sub w9, w21, #1",
    ),
    Patch(
        0x8E5A0,
        bytes.fromhex("c90a0051"),
        bytes.fromhex("c9060051"),
        "SurfTrackerData::lookup_int",
        "sub w9, w22, #2",
        "sub w9, w22, #1",
    ),
    Patch(
        0x8E820,
        bytes.fromhex("a80a0051"),
        bytes.fromhex("a8060051"),
        "SurfTrackerData::valid_int",
        "sub w8, w21, #2",
        "sub w8, w21, #1",
    ),
    Patch(
        0x8E834,
        bytes.fromhex("c80a0051"),
        bytes.fromhex("c8060051"),
        "SurfTrackerData::valid_int",
        "sub w8, w22, #2",
        "sub w8, w22, #1",
    ),
    Patch(
        0x8E9F4,
        bytes.fromhex("c90a0051"),
        bytes.fromhex("c9060051"),
        "SurfTrackerData::lookup_int_loc",
        "sub w9, w22, #2",
        "sub w9, w22, #1",
    ),
    Patch(
        0x8EA10,
        bytes.fromhex("e90a0051"),
        bytes.fromhex("e9060051"),
        "SurfTrackerData::lookup_int_loc",
        "sub w9, w23, #2",
        "sub w9, w23, #1",
    ),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def patch_copy(source: Path, target: Path, expected_sha256: str,
               patches: tuple[Patch, ...], code_offset: int,
               code_size: int) -> dict:
    actual_sha256 = sha256(source)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"{source}: SHA-256 {actual_sha256} != expected {expected_sha256}"
        )

    original = source.read_bytes()
    modified = bytearray(original)
    records = []
    for patch in patches:
        observed = bytes(
            modified[patch.offset : patch.offset + len(patch.before)]
        )
        if observed != patch.before:
            raise RuntimeError(
                f"{source}: 0x{patch.offset:x} contains {observed.hex()}, "
                f"expected {patch.before.hex()}"
            )
        modified[patch.offset : patch.offset + len(patch.before)] = patch.after
        records.append(
            {
                "file_offset_hex": f"0x{patch.offset:x}",
                "before_hex": patch.before.hex(),
                "after_hex": patch.after.hex(),
                "symbol": patch.symbol,
                "instruction_before": patch.instruction_before,
                "instruction_after": patch.instruction_after,
            }
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(modified)
    pre_sign = target.read_bytes()
    raw_diff_offsets = [
        index
        for index, (before, after) in enumerate(zip(original, pre_sign))
        if before != after
    ]
    expected_diff_offsets = sorted(
        patch.offset + byte_index
        for patch in patches
        for byte_index, (before, after) in enumerate(
            zip(patch.before, patch.after)
        )
        if before != after
    )
    if raw_diff_offsets != expected_diff_offsets:
        raise RuntimeError(
            f"{target}: unexpected pre-sign byte differences "
            f"{raw_diff_offsets!r} != {expected_diff_offsets!r}"
        )

    pre_sign_sha256 = sha256(target)
    subprocess.run(
        ["codesign", "--force", "--sign", "-", str(target)],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["codesign", "--verify", "--strict", str(target)],
        check=True,
        capture_output=True,
        text=True,
    )
    signed = target.read_bytes()
    code_diff_offsets = [
        code_offset + index
        for index, (before, after) in enumerate(
            zip(
                original[code_offset : code_offset + code_size],
                signed[code_offset : code_offset + code_size],
            )
        )
        if before != after
    ]
    if code_diff_offsets != expected_diff_offsets:
        raise RuntimeError(
            f"{target}: signed __text differences {code_diff_offsets!r} "
            f"!= {expected_diff_offsets!r}"
        )

    return {
        "source": str(source.resolve()),
        "target": str(target.resolve()),
        "source_sha256": actual_sha256,
        "patched_pre_sign_sha256": pre_sign_sha256,
        "patched_signed_sha256": sha256(target),
        "source_size": len(original),
        "patched_size": len(signed),
        "code_section_offset": code_offset,
        "code_section_size": code_size,
        "pre_sign_changed_byte_offsets": [
            f"0x{offset:x}" for offset in raw_diff_offsets
        ],
        "signed_code_changed_byte_offsets": [
            f"0x{offset:x}" for offset in code_diff_offsets
        ],
        "patches": records,
        "codesign": "codesign --force --sign - TARGET; codesign --verify --strict TARGET",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--frameworks",
        required=True,
        type=Path,
        help="Official VC3D.app/Contents/Frameworks directory",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    core = args.frameworks / "libvc_core.dylib"
    tracer = args.frameworks / "libvc_tracer.dylib"
    helper_dir = args.output / "helper-only"
    full_dir = args.output / "full-bounds"

    manifest = {
        "schema_version": 1,
        "release_revision": "05ff9ea",
        "platform": "macos-arm64",
        "purpose": (
            "Executable bounds ablation only; this is not a build of PR #1264."
        ),
        "semantic_scope": (
            "Only high-side complete-cell bounds change from dimension-2 "
            "to dimension-1. Low-side checks, sentinel checks, interpolation, "
            "and all other release instructions remain unchanged."
        ),
        "helper_only": {
            "libvc_core.dylib": patch_copy(
                core,
                helper_dir / "libvc_core.dylib",
                CORE_SHA256,
                CORE_PATCHES,
                CORE_CODE_OFFSET,
                CORE_CODE_SIZE,
            )
        },
        "full_bounds": {
            "libvc_core.dylib": patch_copy(
                core,
                full_dir / "libvc_core.dylib",
                CORE_SHA256,
                CORE_PATCHES,
                CORE_CODE_OFFSET,
                CORE_CODE_SIZE,
            ),
            "libvc_tracer.dylib": patch_copy(
                tracer,
                full_dir / "libvc_tracer.dylib",
                TRACER_SHA256,
                TRACER_PATCHES,
                TRACER_CODE_OFFSET,
                TRACER_CODE_SIZE,
            ),
        },
    }
    manifest_path = args.output / "patch-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(manifest_path)


if __name__ == "__main__":
    main()
