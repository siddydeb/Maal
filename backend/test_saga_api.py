"""Saga API: multi-output job payload, context/pair bundling, batch endpoint validation (DB and GPU mocked)."""

from __future__ import annotations

import asyncio
import json
import os
import unittest
from io import BytesIO
from unittest.mock import AsyncMock, patch

os.environ.setdefault("MONGODB_URI", "mongodb://localhost/test")

from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

from backend import main, saga  # noqa: E402


def _png(size=(64, 48), alpha=True) -> bytes:
    img = Image.new("RGBA" if alpha else "RGB", size, (200, 150, 120, 255) if alpha else (1, 2, 3))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class _Comfy:
    async def health(self, **_kw):
        return True

    async def ensure_root(self):
        return None


def _outputs(roles):
    return [
        (b"png", "image/png", {"model_label": "anima", "seed_used": 5 + i // 2,
                               "saga": {"prompt_used": "face focus", "denoise": 0.3, "role": role,
                                        "variant": i // 2 + 1, "variants": 2}})
        for i, role in enumerate(roles)
    ]


class ExecuteTests(unittest.TestCase):
    def test_variants_are_stored_and_reported(self):
        stored = []

        async def store(**kw):
            stored.append(kw)
            n = len(stored)
            return {"id": f"g{n}", "kind": kw["kind"], "prompt": kw["prompt"], "model": kw["model"],
                    "content_type": kw["content_type"], "size_bytes": 3, "created_at": "now"}

        with patch.object(main, "_comfy", return_value=_Comfy()), \
                patch.object(main.db, "store_media", side_effect=store), \
                patch.object(saga, "run_still_outputs", AsyncMock(return_value=_outputs(["main", "pair"] * 2))):
            payload = asyncio.run(main._execute_generation(
                mode="saga", prompt="", prompt_en="", image_bytes=saga.pack_inputs({"image": _png(), "pair": _png()}),
                negative=None, seed=5, resolved={"original": ""}, options={"layout": "part", "photo": True}))
        self.assertEqual(len(stored), 4)
        self.assertEqual((payload["prompt_used"], payload["seed_used"], payload["denoise"]), ("face focus", 5, 0.3))
        self.assertEqual([(v["id"], v["seed_used"], v["role"]) for v in payload["variants"]],
                         [("g1", 5, "main"), ("g2", 5, "pair"), ("g3", 6, "main"), ("g4", 6, "pair")])
        self.assertEqual(stored[3]["meta"]["saga"]["variant"], 2)

    def test_batch_zip_is_not_a_library_image(self):
        stored = []

        async def store(**kw):
            stored.append(kw)
            return {"id": "z1", "kind": kw["kind"], "prompt": "", "model": "m", "content_type": kw["content_type"],
                    "size_bytes": 3, "created_at": "now"}

        result = (b"PK", "application/zip", {"model_label": "anima", "seed_used": 9,
                                              "saga": {"kind": "batch", "prompt_used": "x", "denoise": 0.3}})
        with patch.object(main, "_comfy", return_value=_Comfy()), \
                patch.object(main.db, "store_media", side_effect=store), \
                patch.object(saga, "run_batch", AsyncMock(return_value=result)):
            payload = asyncio.run(main._execute_generation(
                mode="saga", prompt="", prompt_en="", image_bytes=b"PK\x03\x04", negative=None, seed=9,
                resolved={"original": ""}, options={"batch": True}))
        self.assertEqual((stored[0]["kind"], payload["thumb_url"], payload["seed_used"]), ("zip", None, 9))


class EndpointTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(main.app)
        self.patches = [
            patch.object(main, "_comfy", return_value=_Comfy()),
            patch.object(main, "_kick_job_dispatcher"),
            patch.object(main.db, "find_job_by_client_key", AsyncMock(return_value=None)),
        ]
        for p in self.patches:
            p.start()
        main.app.dependency_overrides[main.require_owner] = lambda: "usaga"
        self.created = []

        async def create_job(**kw):
            self.created.append(kw)
            return {"id": "j1", "queue_position": 1}

        self.patches.append(patch.object(main.db, "create_job", side_effect=create_job))
        self.patches[-1].start()
        self.patches.append(patch.object(main, "is_saga_owner", return_value=True))
        self.patches[-1].start()

    def tearDown(self):
        main.app.dependency_overrides.clear()
        for p in self.patches:
            p.stop()

    def test_part_job_bundles_context_and_pair(self):
        r = self.client.post("/api/jobs", data={"mode": "saga", "prompt": "", "layout": "part", "count": "4",
                                                "strength": "subtle", "keep_lines": "1"},
                             files={"image": ("f.png", _png()), "pair": ("t.png", _png()), "context": ("b.png", _png())})
        self.assertEqual(r.status_code, 200, r.text)
        job = self.created[0]
        self.assertEqual((job["options"]["count"], job["options"]["keep_lines"], job["options"]["strength"]),
                         (4, True, "subtle"))
        files, _ = saga.unpack_inputs(job["image_bytes"])
        self.assertEqual(sorted(files), ["context", "image", "pair"])

    def test_part_job_rejects_mismatched_pair(self):
        r = self.client.post("/api/jobs", data={"mode": "saga", "prompt": "", "layout": "part"},
                             files={"image": ("f.png", _png()), "pair": ("t.png", _png((32, 32)))})
        self.assertEqual(r.status_code, 400)
        self.assertIn("same size", r.json()["detail"])

    def test_batch(self):
        manifest = {"items": [{"image": "jenny/f_gross.png", "pair": "jenny/f_gross_talk.png", "context": "body.png"},
                              {"image": "jenny/f_laugh.png"}]}
        files = [("files", (name, _png())) for name in
                 ("jenny/f_gross.png", "jenny/f_gross_talk.png", "body.png", "jenny/f_laugh.png", "unused.png")]
        r = self.client.post("/api/saga/batch", data={"manifest": json.dumps(manifest), "strength": "subtle",
                                                      "match_skin": "1", "seed": "77"}, files=files)
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual((r.json()["items"], r.json()["seed"]), (2, 77))
        job = self.created[0]
        self.assertEqual((job["options"]["batch"], job["options"]["match_skin"], job["seed"]), (True, True, 77))
        files, stored_manifest = saga.unpack_inputs(job["image_bytes"])
        self.assertNotIn("unused.png", files)
        self.assertEqual(stored_manifest["items"][0]["pair"], ["jenny/f_gross_talk.png"])

    def test_rewrite_off_keeps_prompt_as_typed(self):
        r = self.client.post("/api/jobs", data={"mode": "saga", "prompt": "1boy, navy blue pants, long black hair",
                                                "rewrite": "0", "negative": " stubble,  beard "})
        self.assertEqual(r.status_code, 200, r.text)
        job = self.created[0]
        self.assertEqual(job["prompt_english"], "1boy, navy blue pants, long black hair")
        self.assertEqual(job["options"]["negative"], "stubble, beard")

    def test_rewrite_on_never_recolors(self):
        r = self.client.post("/api/jobs", data={"mode": "saga", "prompt": "man wearing navy blue pants, jacket black"})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertNotIn("change", self.created[0]["prompt_english"])

    def test_batch_negative_and_raw_prompt(self):
        r = self.client.post("/api/saga/batch", data={
            "manifest": json.dumps({"items": [{"image": "f.png"}]}), "prompt": "navy blue shirt",
            "rewrite": "0", "negative": "stubble, beard"}, files=[("files", ("f.png", _png()))])
        self.assertEqual(r.status_code, 200, r.text)
        job = self.created[0]
        self.assertEqual((job["prompt_english"], job["options"]["negative"]), ("navy blue shirt", "stubble, beard"))

    def test_batch_bad_manifest(self):
        r = self.client.post("/api/saga/batch", data={"manifest": json.dumps({"items": [{"image": "x.png"}]})},
                             files=[("files", ("y.png", _png()))])
        self.assertEqual(r.status_code, 400)
        self.assertIn("x.png", r.json()["detail"])


if __name__ == "__main__":
    unittest.main()
