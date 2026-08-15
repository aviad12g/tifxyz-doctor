# August 2026 Progress Prize form copy

Official form: <https://docs.google.com/forms/d/e/1FAIpQLSev2vJobu521iB6OuyehDktzYTEo131F4iUGwt3Qxa9a1fk6A/viewform>

Deadline verified on August 12, 2026: **August 31, 2026 at 11:59pm Pacific**.

## Email

`aivadcohen123@gmail.com`

## Full name

`Aviad Cohen`

## Team description

Individual submission by Aviad Cohen (`aviad12g`), with OpenAI Codex used as an AI coding and research assistant for evidence auditing, packaging, and submission preparation. Aviad Cohen is the team leader.

## Public contribution URL

<https://github.com/aviad12g/tifxyz-doctor/tree/agent/freeze-fusion-aware-checkpoints/experiments/fail-closed-surface-to-ink>

## Short description of how the contribution substantially increases the probability of reading complete scrolls

Fail-Closed Surface-to-Ink is a modular validation and stopping framework for Vesuvius Challenge geometry and ink searches. It accepts standard OME-Zarr/Zarr volumes and TIFFXYZ quadmeshes, then produces hash-bound geometry, m7-support, full normal-offset sampling, raw-depth, model-consensus, and terminal-decision evidence. It prevents plausible but invalid surfaces or texture artifacts from silently advancing into expensive inference and apparent-letter claims.

We demonstrated the framework across all 13 scroll volumes eligible for the 2027 Grand Prize. Four scrolls reached clean ≥0.5 cm²-class raw surfaces; independent depth- and artifact-controlled review rejected every proposed text region. On PHerc0813, GrowPatch’s whole proposal grew from 2.13 to 10.10 cm², while the accepted component plateaued at 0.536 cm²: generations 60 and 80 contained exactly the same 1,520 accepted quads. The toolkit therefore proves when “grow farther” adds no valid physical evidence. A separate three-seed same-scroll experiment showed that a coarse model could memorize one region (AUROC 0.996) yet fail only 1.228 mm away (AUROC 0.512). In a later pinned canonical ResNet-152 PHerc1203 pilot, forward/reverse maps had correlation 0.0324, top-1% overlap 0.00336, and zero top-0.5% overlap; fixed raw review found no defensible text, so the framework skipped further inference.

The contribution includes native TIFFXYZ validation, exact self-intersection and incremental-area audits, calibrated surface gating, seam-safe Zarr rendering, model-independent raw-depth controls, symmetric forward/reverse consensus, reproducible provenance, focused tests, and a 13-scroll stopping/reopen policy. It does not claim text; it supplies actionable failure diagnostics and reusable infrastructure that make future unwrapping and ink-detection work faster, safer, and easier to reproduce.

## Terms

`Yes, I agree` — only the submitting team leader should select this in the official form.
