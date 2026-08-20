#!/usr/bin/env python3
"""Mechanically verify a sealed GapBalance bundle without running or scoring it."""

from __future__ import annotations

import argparse
from pathlib import Path

from execute_gapbalance_runpod_development import load_hashed, verify_bundle


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--bundle-manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-payload-sha256", required=True)
    args = parser.parse_args()
    plan = load_hashed(args.plan)
    manifest = verify_bundle(args.bundle_root, args.bundle_manifest, plan)
    if manifest["payload_sha256"] != args.expected_manifest_payload_sha256:
        raise RuntimeError("bundle manifest payload identity mismatch")
    print("GAPBALANCE_RUNPOD_BUNDLE_VERIFIED_NOT_SCORED", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
