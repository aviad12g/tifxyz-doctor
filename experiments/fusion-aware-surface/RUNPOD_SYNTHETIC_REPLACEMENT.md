# RunPod synthetic execution replacement

This document freezes an operational replacement for the synthetic half of the
held-out evaluation before any RunPod held-out inference. It changes the compute
platform and nothing scientific.

## Authority

The seven RTX 4090 RunPod executions are the sole authoritative primary
synthetic execution. The Kaggle execution is retained as a sealed secondary
cross-platform replication. We will never choose the more favorable platform,
substitute Kaggle for RunPod, or suppress a disagreement. A disagreement will
be reported with both outcomes and investigated only for operational causes.

The already accepted Kaggle baseline had not been opened when this hierarchy
was frozen. Kaggle may finish asynchronously, but it may not delay scoring,
publication, or submission of the completed RunPod primary.

## Scientific invariants

All seven jobs use the exact embedded launchers generated from held-out plan
commit `46d030245c14ecb2be63796a308581c462fc2e29`. Model-state hashes,
thresholds, test cells, painter seeds, shard membership, ray endpoints, runtime
versions, gates, and scorer code remain unchanged. Each primary job contains
100 cells in the same ten shards. Partial primary output cannot be scored.

No held-out endpoint, cache, manifest, or scientific result was inspected to
make this replacement. The reason is execution latency only.

## Hardware and isolation

The primary hardware is seven NVIDIA GeForce RTX 4090 GPUs, one visible GPU and
one frozen launcher process per scientific job, using on-demand community-cloud
instances. The preferred deployment is one seven-GPU pod so the immutable 3.28
GB synthetic bundle is uploaded only once. That bundle contains the source
model, six trained states, all ledger-bound code, thresholds, and launchers; it
omits the unused real-scroll image archives. A single eight-GPU host with one idle GPU,
then seven independent one-GPU pods, are the frozen capacity fallbacks. Every
layout has exactly seven isolated workers and one visible GPU per worker.

The immutable base image is
`runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` with manifest digest
`sha256:61a4aafb0094cd773f11eefa378929d5a687bd775febeb78eac62fc824141fb5`.
Every frozen launcher installs and verifies the already specified PyTorch
2.5.1/CUDA 12.1 runtime and companion package versions before inference.

## Hard budget gate

The total RunPod authorization is USD 20.00. The controller accepts no GPU
above USD 0.40 per GPU-hour, reserves USD 1.50 for non-compute charges, and
stops all primary pods at USD 18.50 of accrued compute. Reaching that cutoff
produces an incomplete execution, not a scientific result. The controller must
never rely on a user remembering to stop a pod.

## Sealing and next gate

During caching, only provider status and operational logs on a failed job may
be inspected. Probability/ray caches, endpoints, job indexes, and cache
manifests remain sealed. After all seven RunPod jobs complete, their deliveries
are hash-validated and publicly frozen before the paired one-shot scorers are
launched. A private, hash-bound Kaggle dataset may be used only to transport the
RunPod-produced primary caches into the sealed synthetic scorer; it cannot
replace their origin or authority. The provider-neutral adapter and delivery
manifest must be public before either scorer result is unsealed. Kaggle
replication stays secondary under the authority rule above.

The machine-readable contract is
`runpod_synthetic_replacement_plan.json`; its canonical `payload_sha256` binds
all execution, budget, provenance, and authority fields.
