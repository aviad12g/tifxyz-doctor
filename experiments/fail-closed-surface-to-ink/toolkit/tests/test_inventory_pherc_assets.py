from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


MODULE_PATH = Path(__file__).parents[1] / "inventory_pherc_assets.py"
SPEC = importlib.util.spec_from_file_location("inventory_pherc_assets", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
inventory = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = inventory
SPEC.loader.exec_module(inventory)


def object_xml(
    key: str,
    *,
    size: int,
    etag: str,
    modified: str = "2026-07-08T14:17:00.000Z",
) -> str:
    return f"""
    <Contents>
      <Key>{key}</Key>
      <LastModified>{modified}</LastModified>
      <ETag>&quot;{etag}&quot;</ETag>
      <ChecksumAlgorithm>CRC64NVME</ChecksumAlgorithm>
      <ChecksumType>FULL_OBJECT</ChecksumType>
      <Size>{size}</Size>
      <StorageClass>INTELLIGENT_TIERING</StorageClass>
    </Contents>"""


def page_xml(
    prefix: str,
    objects: list[str],
    *,
    truncated: bool = False,
    continuation: str | None = None,
    next_token: str | None = None,
    namespace: bool = True,
    delimiter: str | None = None,
    common_prefixes: list[str] | None = None,
) -> str:
    ns = ' xmlns="http://s3.amazonaws.com/doc/2006-03-01/"' if namespace else ""
    continuation_xml = (
        f"<ContinuationToken>{continuation}</ContinuationToken>"
        if continuation is not None
        else ""
    )
    next_xml = (
        f"<NextContinuationToken>{next_token}</NextContinuationToken>"
        if next_token is not None
        else ""
    )
    delimiter_xml = f"<Delimiter>{delimiter}</Delimiter>" if delimiter else ""
    common_prefix_xml = "".join(
        f"<CommonPrefixes><Prefix>{item}</Prefix></CommonPrefixes>"
        for item in (common_prefixes or [])
    )
    key_count = len(objects) + len(common_prefixes or [])
    return (
        f'<?xml version="1.0" encoding="UTF-8"?>\n<ListBucketResult{ns}>'
        f"<Name>vesuvius-challenge-open-data</Name><Prefix>{prefix}</Prefix>"
        f"{continuation_xml}{next_xml}<KeyCount>{key_count}</KeyCount>"
        f"<MaxKeys>1000</MaxKeys>{delimiter_xml}"
        f"<IsTruncated>{str(truncated).lower()}</IsTruncated>"
        + "".join(objects)
        + common_prefix_xml
        + "</ListBucketResult>"
    )


class ParseTests(unittest.TestCase):
    def test_parse_preserves_s3_metadata_with_or_without_namespace(self) -> None:
        key = "PHerc1447/volumes/raw.zarr/.zattrs"
        for namespace in (True, False):
            with self.subTest(namespace=namespace):
                page = inventory.parse_list_objects_v2_xml(
                    page_xml(
                        "PHerc1447/volumes/",
                        [object_xml(key, size=17, etag="abc")],
                        namespace=namespace,
                    )
                )
                self.assertEqual(page.prefix, "PHerc1447/volumes/")
                self.assertEqual(page.objects[0].key, key)
                self.assertEqual(page.objects[0].size, 17)
                self.assertEqual(page.objects[0].etag, '"abc"')
                self.assertEqual(
                    page.objects[0].last_modified, "2026-07-08T14:17:00.000Z"
                )
                self.assertEqual(
                    page.objects[0].checksum_algorithms, ("CRC64NVME",)
                )

    def test_truncated_page_without_token_is_rejected(self) -> None:
        page = inventory.parse_list_objects_v2_xml(
            page_xml("PHerc1447/segments/", [], truncated=True)
        )

        class Source:
            def fetch(self, prefix, continuation_token, delimiter=None):
                return page

        with self.assertRaisesRegex(inventory.InventoryError, "no NextContinuationToken"):
            inventory.list_all_objects(Source(), "PHerc1447/segments/")


class FixtureInventoryTests(unittest.TestCase):
    def _write_fixtures(self, directory: Path) -> None:
        volume_prefix = "PHerc1447/volumes/"
        prediction_prefix = "PHerc1447/representations/predictions/"
        segment_prefix = "PHerc1447/segments/"
        (directory / "01-volumes.xml").write_text(
            page_xml(
                volume_prefix,
                [
                    object_xml(
                        volume_prefix + "20250521151220-8.640um-masked.zarr/.zattrs",
                        size=101,
                        etag="volume-etag",
                    )
                ],
            ),
            encoding="utf-8",
        )
        (directory / "02-predictions.xml").write_text(
            page_xml(
                prediction_prefix,
                [
                    object_xml(
                        prediction_prefix
                        + "surfaces/surface-m7.normal-grids/0/0.0.0",
                        size=202,
                        etag="normal-etag",
                    ),
                    object_xml(
                        prediction_prefix + "lasagna/ink.zarr/zarr.json",
                        size=203,
                        etag="prediction-etag",
                    ),
                ],
            ),
            encoding="utf-8",
        )
        segment_a = segment_prefix + "segment-a/mesh/intermediate/tifxyz_original/"
        (directory / "03-segments-page-1.xml").write_text(
            page_xml(
                segment_prefix,
                [
                    object_xml(segment_a + "meta.json", size=1, etag="meta"),
                    object_xml(segment_a + "x.tif", size=11, etag="x"),
                    object_xml(segment_a + "y.tif", size=12, etag="y"),
                ],
                truncated=True,
                next_token="segments-page-2",
            ),
            encoding="utf-8",
        )
        segment_b = segment_prefix + "segment-b/"
        (directory / "04-segments-page-2.xml").write_text(
            page_xml(
                segment_prefix,
                [
                    object_xml(segment_a + "z.tif", size=13, etag="z"),
                    object_xml(
                        segment_b + "mesh/intermediate/segment-b_original.obj",
                        size=301,
                        etag="obj",
                    ),
                    object_xml(
                        segment_b + "surface-volumes/raw-surface.tifs/00.tif",
                        size=302,
                        etag="surface",
                    ),
                    object_xml(
                        segment_b + "ink-detection/prediction.tif",
                        size=303,
                        etag="other",
                    ),
                ],
                continuation="segments-page-2",
            ),
            encoding="utf-8",
        )

    def _write_scalable_fixtures(self, directory: Path) -> None:
        volume_prefix = "PHerc1447/volumes/"
        prediction_prefix = "PHerc1447/representations/predictions/"
        surfaces_prediction_prefix = prediction_prefix + "surfaces/"
        segments_prefix = "PHerc1447/segments/"
        segment_a = segments_prefix + "segment-a/"
        segment_b = segments_prefix + "segment-b/"
        raw_prefix = segments_prefix + "raw/"
        nested_segment = raw_prefix + "debug-1/"
        fixtures = {
            "01-volume-root.xml": page_xml(
                volume_prefix,
                [],
                delimiter="/",
                common_prefixes=[volume_prefix + "raw.ome.zarr/"],
            ),
            "02-prediction-root.xml": page_xml(
                prediction_prefix,
                [],
                delimiter="/",
                common_prefixes=[surfaces_prediction_prefix],
            ),
            "03-prediction-surfaces.xml": page_xml(
                surfaces_prediction_prefix,
                [],
                delimiter="/",
                common_prefixes=[
                    surfaces_prediction_prefix + "surface.normal-grids/",
                    surfaces_prediction_prefix + "surface.zarr/",
                ],
            ),
            "04-segment-root.xml": page_xml(
                segments_prefix,
                [],
                delimiter="/",
                common_prefixes=[segment_a, segment_b, raw_prefix],
            ),
            "04c-raw-root.xml": page_xml(
                raw_prefix,
                [],
                delimiter="/",
                common_prefixes=[nested_segment],
            ),
            "04d-debug-root.xml": page_xml(
                nested_segment,
                [
                    object_xml(
                        nested_segment + "meta.json",
                        size=7,
                        etag="debug-meta",
                    ),
                    object_xml(
                        nested_segment + "x.tif",
                        size=8,
                        etag="debug-x",
                    ),
                    object_xml(
                        nested_segment + "y.tif",
                        size=9,
                        etag="debug-y",
                    ),
                    object_xml(
                        nested_segment + "z.tif",
                        size=10,
                        etag="debug-z",
                    ),
                ],
                delimiter="/",
            ),
            "05-segment-a-mesh.xml": page_xml(
                segment_a + "mesh/",
                [
                    object_xml(
                        segment_a + "mesh/tifxyz_original/meta.json",
                        size=1,
                        etag="meta",
                    ),
                    object_xml(
                        segment_a + "mesh/tifxyz_original/x.tif",
                        size=2,
                        etag="x",
                    ),
                    object_xml(
                        segment_a + "mesh/tifxyz_original/y.tif",
                        size=3,
                        etag="y",
                    ),
                    object_xml(
                        segment_a + "mesh/tifxyz_original/z.tif",
                        size=4,
                        etag="z",
                    ),
                ],
            ),
            "04a-segment-a-root.xml": page_xml(
                segment_a,
                [],
                delimiter="/",
                common_prefixes=[
                    segment_a + "mesh/",
                    segment_a + "surface-volumes/",
                ],
            ),
            "04b-segment-b-root.xml": page_xml(
                segment_b,
                [],
                delimiter="/",
                common_prefixes=[
                    segment_b + "mesh/",
                    segment_b + "surface-volumes/",
                ],
            ),
            "06-segment-a-surfaces.xml": page_xml(
                segment_a + "surface-volumes/", [], delimiter="/"
            ),
            "07-segment-b-mesh.xml": page_xml(
                segment_b + "mesh/",
                [
                    object_xml(
                        segment_b + "mesh/segment-b_original.obj",
                        size=5,
                        etag="obj",
                    )
                ],
            ),
            "08-segment-b-surfaces.xml": page_xml(
                segment_b + "surface-volumes/",
                [],
                delimiter="/",
                common_prefixes=[
                    segment_b + "surface-volumes/surface.tifs/",
                    segment_b + "surface-volumes/surface.zarr/",
                ],
            ),
            "09-segment-b-tifs.xml": page_xml(
                segment_b + "surface-volumes/surface.tifs/",
                [
                    object_xml(
                        segment_b + "surface-volumes/surface.tifs/00.tif",
                        size=6,
                        etag="layer",
                    )
                ],
            ),
        }
        for name, payload in fixtures.items():
            (directory / name).write_text(payload, encoding="utf-8")

    def test_offline_fixtures_paginate_and_classify_every_asset(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            fixture_dir = Path(temp_dir)
            self._write_fixtures(fixture_dir)
            source = inventory.FixtureListObjectsV2Source([fixture_dir])
            report = inventory.inventory_scroll(
                "PHerc1447", source, scalable=False
            )

        summary = report["summary"]
        self.assertEqual(report["source"]["mode"], "xml-fixtures")
        self.assertEqual(
            report["source"]["pages_by_prefix"]["PHerc1447/segments/"], 2
        )
        self.assertEqual(summary["objects_total"], 10)
        self.assertEqual(summary["raw_volume_zarrs"], 1)
        self.assertEqual(summary["prediction_assets"], 2)
        self.assertEqual(summary["normal_grid_assets"], 1)
        self.assertEqual(summary["segments_total"], 2)
        self.assertEqual(summary["complete_tifxyz_variants"], 1)
        self.assertEqual(summary["surface_tiff_stacks"], 1)
        self.assertEqual(summary["unscreened_tifxyz_only_meshes"], 1)
        self.assertEqual(summary["other_objects_retained"], 1)

        candidate = report["unscreened_tifxyz_only_meshes"][0]
        self.assertEqual(candidate["segment_id"], "segment-a")
        self.assertEqual(candidate["transform"], "original")
        self.assertEqual(
            candidate["reason"], "complete_tifxyz_no_published_surface_volume"
        )
        segment_a = next(
            segment for segment in report["segments"]
            if segment["segment_id"] == "segment-a"
        )
        self.assertTrue(segment_a["capabilities"]["surface_extraction_required"])
        axes = segment_a["tifxyz_variants"][0]["coordinate_axes_present"]
        self.assertEqual(axes, ["x", "y", "z"])
        segment_b = next(
            segment for segment in report["segments"]
            if segment["segment_id"] == "segment-b"
        )
        self.assertTrue(
            segment_b["capabilities"]["direct_surface_screening_ready"]
        )
        self.assertEqual(len(segment_b["other_objects"]), 1)

        # Object metadata is neither dropped nor duplicated by classification.
        object_records: list[dict[str, object]] = []

        def visit(value: object) -> None:
            if isinstance(value, dict):
                if {
                    "key",
                    "size",
                    "etag",
                    "last_modified",
                    "storage_class",
                }.issubset(value):
                    object_records.append(value)
                else:
                    for child in value.values():
                        visit(child)
            elif isinstance(value, list):
                for child in value:
                    visit(child)

        visit(report)
        self.assertEqual(len(object_records), summary["objects_total"])
        self.assertEqual(
            len({str(record["key"]) for record in object_records}),
            summary["objects_total"],
        )

    def test_snapshot_is_deterministic_and_output_is_immutable_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fixtures = root / "fixtures"
            fixtures.mkdir()
            self._write_fixtures(fixtures)
            first = inventory.inventory_scroll(
                "PHerc1447",
                inventory.FixtureListObjectsV2Source([fixtures]),
                scalable=False,
            )
            second = inventory.inventory_scroll(
                "PHerc1447",
                inventory.FixtureListObjectsV2Source([fixtures]),
                scalable=False,
            )
            self.assertEqual(
                first["snapshot"]["objects_sha256"],
                second["snapshot"]["objects_sha256"],
            )
            output = root / "inventory.json"
            inventory.write_report(first, str(output))
            loaded = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(loaded["snapshot"], first["snapshot"])
            with self.assertRaisesRegex(inventory.InventoryError, "refusing to overwrite"):
                inventory.write_report(first, str(output))
            inventory.write_report(second, str(output), force=True)

    def test_scalable_discovery_stops_before_large_chunk_containers(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            fixture_dir = Path(temp_dir)
            self._write_scalable_fixtures(fixture_dir)
            report = inventory.inventory_scroll(
                "PHerc1447",
                inventory.FixtureListObjectsV2Source([fixture_dir]),
                workers=2,
            )

        summary = report["summary"]
        self.assertTrue(report["source"]["scalable_discovery"])
        self.assertFalse(
            report["source"]["large_container_descendants_enumerated"]
        )
        self.assertEqual(summary["objects_total"], 10)
        self.assertEqual(summary["raw_volume_zarrs"], 1)
        self.assertEqual(summary["prediction_assets"], 2)
        self.assertEqual(summary["normal_grid_assets"], 1)
        self.assertEqual(summary["surface_volume_assets"], 2)
        self.assertEqual(summary["surface_tiff_stacks"], 1)
        self.assertEqual(summary["surface_zarrs"], 1)
        self.assertEqual(summary["segments_total"], 2)
        self.assertEqual(summary["unscreened_tifxyz_only_meshes"], 1)
        self.assertEqual(summary["raw_tifxyz_variants"], 1)
        self.assertEqual(summary["unseen_raw_tifxyz_candidates"], 1)
        self.assertEqual(summary["asset_prefixes_total"], 8)
        self.assertEqual(summary["prefix_only_assets"], 4)
        self.assertEqual(summary["recursively_enumerated_assets"], 4)
        self.assertEqual(
            report["raw_volumes"][0]["object_inventory"]["mode"],
            "prefix_only",
        )
        self.assertEqual(report["raw_volumes"][0]["objects"], [])
        self.assertEqual(
            report["source"]["listing_audit"]["request_count"], 13
        )
        requested = {
            item["prefix"]
            for item in report["source"]["listing_audit"]["requests"]
        }
        self.assertNotIn("PHerc1447/volumes/raw.ome.zarr/", requested)
        self.assertNotIn(
            "PHerc1447/representations/predictions/surfaces/surface.zarr/",
            requested,
        )
        self.assertEqual(
            report["unseen_raw_tifxyz_candidates"][0]["variant_id"],
            "debug-1",
        )

        metadata_records: list[dict[str, object]] = []

        def collect_metadata(value: object) -> None:
            if isinstance(value, dict):
                if {
                    "key",
                    "size",
                    "etag",
                    "last_modified",
                    "storage_class",
                }.issubset(value):
                    metadata_records.append(value)
                else:
                    for child in value.values():
                        collect_metadata(child)
            elif isinstance(value, list):
                for child in value:
                    collect_metadata(child)

        collect_metadata(report)
        self.assertEqual(len(metadata_records), summary["indexed_objects_total"])
        self.assertEqual(
            len({str(record["key"]) for record in metadata_records}),
            summary["indexed_objects_total"],
        )


class PaginationSafetyTests(unittest.TestCase):
    def test_duplicate_object_across_pages_is_rejected(self) -> None:
        prefix = "PHerc1447/segments/"
        record = inventory.ObjectRecord(
            key=prefix + "segment/mesh/a.obj",
            size=1,
            etag='"a"',
            last_modified="2026-01-01T00:00:00.000Z",
        )
        pages = {
            None: inventory.ListPage(
                prefix, None, None, "next", True, (record,)
            ),
            "next": inventory.ListPage(
                prefix, None, "next", None, False, (record,)
            ),
        }

        class Source:
            def fetch(self, requested_prefix, token, delimiter=None):
                return pages[token]

        with self.assertRaisesRegex(inventory.InventoryError, "duplicate key"):
            inventory.list_all_objects(Source(), prefix)


if __name__ == "__main__":
    unittest.main()
