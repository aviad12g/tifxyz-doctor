# Discord launch note

Hi — I released TIFXYZ Doctor, a read-only preflight and
geometry-diagnostics CLI for Vesuvius TIFXYZ surfaces:
https://github.com/aviad12g/tifxyz-doctor

While building it, I audited a bounded set of registered public packages and
found three concrete collection/interoperability observations:

1. In a metadata census of 450 registered original/normalized roots, 277 use
   the literal UUID `output_tifxyz` (including all 264 normalized roots).
2. All 138 inspected PHerc1203 current/version metadata snapshots provide
   `area_vx2` and `area_cm2` but not `area`; at the Villa revision I inspected,
   the Python reader retains those fields in `extra` while `Tifxyz.area` is
   unset.
3. Three registered `tifxyz_normalized` packages resolve to only a 2×2
   `(-1,-1,-1)` sentinel grid, so they have zero valid vertices and zero
   renderable faces:
   - PHerc0332 `20240711124827-20240618142020`
   - PHerc0332 `20240828190516-20240716140050`
   - PHerc0500P2 `20250716055236-z_dbg_gen_00356_inp_hr`

The repository includes a hash-pinned 3/3 regression for the sentinel-only
packages, a machine-readable corpus record with exact URLs and hashes, 47
tests, and an executable five-case Python/C++ reader differential. I am
treating UUID reuse as a collection warning, not coordinate corruption, and I
am not inferring why the empty entries exist—especially from `z_dbg`.

Are the UUID reuse and area-key omission intentional compatibility choices, and
are the three sentinel-only registered entries—especially the `z_dbg`-named
entry—expected registry artifacts? Feedback on the checks, corpus evidence, or
usefulness in the segmentation workflow would be very welcome.
