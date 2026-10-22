"""Saga studio: adult-only prompt guard, options, and the Anima still graph."""

from __future__ import annotations

import asyncio
import unittest
from io import BytesIO

from backend import saga


class PromptGuardTests(unittest.TestCase):
    def test_adult_prompts_pass(self):
        for text in (
            "a 40 year old blonde woman in a kitchen",
            "young woman at the beach, 25 years old",
            "college professor in her office",
        ):
            self.assertEqual(saga.check_prompt(text), text)

    def test_minor_prompts_rejected(self):
        for text in (
            "a schoolgirl in uniform",
            "teen girl at the mall",
            "a 16 year old girl",
            "17yo cheerleader",
            "loli character",
            "",
        ):
            with self.assertRaises(saga.SagaError, msg=text):
                saga.check_prompt(text)


class RewriteTests(unittest.TestCase):
    def test_clean_tags(self):
        raw = "<think>\n\n</think>\n\nTags: 1girl, nurse, adult woman, nurse, adult woman, night\nextra line"
        self.assertEqual(saga.clean_tags(raw), "1girl, nurse, adult woman, night")

    def test_looks_like_tags(self):
        self.assertTrue(saga.looks_like_tags("1girl, solo, long hair, smile, red dress"))
        self.assertFalse(saga.looks_like_tags("a tired nurse sitting on a bench at night"))
        self.assertFalse(saga.looks_like_tags("face"))

    def test_rewrite_falls_back_to_original(self):
        class _Broken:
            async def run_text(self, graph):
                raise RuntimeError("comfy down")

        self.assertEqual(asyncio.run(saga.rewrite_prompt(_Broken(), "a woman reading")), "a woman reading")

    def test_rewrite_rejects_minor_tags(self):
        class _Bad:
            async def run_text(self, graph):
                return "1girl, loli"

        self.assertEqual(asyncio.run(saga.rewrite_prompt(_Bad(), "a small woman")), "a small woman")


class OptionTests(unittest.TestCase):
    def test_aspect(self):
        self.assertEqual(saga.parse_aspect(None), "landscape")
        self.assertEqual(saga.parse_aspect("Portrait"), "portrait")
        with self.assertRaises(saga.SagaError):
            saga.parse_aspect("panorama")

    def test_seconds(self):
        self.assertEqual(saga.parse_seconds(None), 5)
        self.assertEqual(saga.parse_seconds("3"), 3)
        self.assertEqual(saga.parse_seconds("10"), 10)
        with self.assertRaises(saga.SagaError):
            saga.parse_seconds("4")
        with self.assertRaises(saga.SagaError):
            saga.parse_seconds("abc")

    def test_still_options(self):
        opts = saga.parse_still_options()
        self.assertEqual(opts, {"aspect": "landscape", "character": None, "outfit": True, "quality": "hd",
                                "layout": "scene", "background": None, "rewrite": True, "count": 1,
                                "negative": None})
        self.assertFalse(saga.parse_still_options(rewrite="0")["rewrite"])
        opts = saga.parse_still_options(character="Debbie", outfit="0", quality="fast")
        self.assertEqual((opts["character"], opts["outfit"], opts["quality"]), ("debbie", False, "fast"))
        with self.assertRaises(saga.SagaError):
            saga.parse_still_options(character="jenny")
        with self.assertRaises(saga.SagaError):
            saga.parse_still_options(layout="sprite")
        with self.assertRaises(saga.SagaError):
            saga.parse_still_options(layout="sprite", background="a high school classroom full of teens")
        opts = saga.parse_still_options(layout="sprite", background="suburban kitchen")
        self.assertEqual(opts["background"], "suburban kitchen")

    def test_photo_options(self):
        opts = saga.parse_still_options(photo=True)
        self.assertEqual((opts["photo"], opts["strength"]), (True, "medium"))
        self.assertNotIn("photo", saga.parse_still_options())
        opts = saga.parse_still_options(photo=True, layout="sprite", strength="Strong")
        self.assertEqual((opts["layout"], opts["background"], opts["strength"]), ("sprite", None, "strong"))
        with self.assertRaises(saga.SagaError):
            saga.parse_still_options(photo=True, strength="max")

    def test_empty_prompt_only_with_photo(self):
        self.assertEqual(saga.check_prompt("  ", allow_empty=True), "")
        with self.assertRaises(saga.SagaError):
            saga.check_prompt("")
        with self.assertRaises(saga.SagaError):
            saga.check_prompt("a schoolgirl", allow_empty=True)

    def test_prepare_photo_snaps_to_16(self):
        from io import BytesIO

        from PIL import Image

        buf = BytesIO()
        Image.new("RGB", (1000, 1500), "red").save(buf, format="JPEG")
        png, w, h = saga.prepare_photo(buf.getvalue())
        self.assertEqual((w % 16, h % 16), (0, 0))
        self.assertLessEqual(max(w, h), saga.PHOTO_BOX)
        self.assertEqual(Image.open(BytesIO(png)).size, (w, h))

    def test_public_config_filters_installed(self):
        cfg = saga.public_config({saga.char_lora("debbie")})
        self.assertEqual([c["id"] for c in cfg["characters"]], ["debbie"])
        self.assertEqual(len(saga.public_config(None)["characters"]), len(saga.CHARACTERS))

    def test_segment_plan(self):
        self.assertEqual(saga.segment_plan(3), (1, 49))
        self.assertEqual(saga.segment_plan(5), (1, 81))
        self.assertEqual(saga.segment_plan(8), (2, 65))
        self.assertEqual(saga.segment_plan(10), (2, 81))


class GraphTests(unittest.TestCase):
    def test_still_graph_uses_anima_and_style_lora(self):
        g = saga.build_still_graph("a woman in a kitchen", seed=7, width=1344, height=768)
        self.assertEqual(g["1"]["inputs"]["unet_name"], saga.ANIMA_UNET)
        self.assertEqual(g["2"]["inputs"]["type"], "stable_diffusion")
        self.assertEqual(g["40"]["inputs"]["lora_name"], saga.STYLE_LORA)
        self.assertNotIn("41", g)
        self.assertIn(saga.STYLE_TRIGGER, g["5"]["inputs"]["text"])
        self.assertTrue(g["5"]["inputs"]["text"].endswith("a woman in a kitchen"))
        self.assertIn("child", g["6"]["inputs"]["text"])
        self.assertEqual((g["7"]["inputs"]["width"], g["7"]["inputs"]["height"]), (1344, 768))
        self.assertEqual(g["8"]["inputs"]["sampler_name"], "er_sde")
        self.assertEqual(g["10"]["inputs"]["images"], ["9", 0])

    def test_character_hd_graph(self):
        g = saga.build_still_graph("x", seed=7, width=1344, height=768, character="diane", hd=True)
        self.assertEqual(g["40"]["inputs"]["strength_model"], saga.STYLE_STRENGTH_WITH_CHAR)
        self.assertEqual(g["41"]["inputs"]["lora_name"], saga.char_lora("diane"))
        self.assertEqual(g["8"]["inputs"]["model"], ["41", 0])
        self.assertEqual((g["13"]["inputs"]["width"], g["13"]["inputs"]["height"]), (2016, 1152))
        self.assertEqual(g["15"]["inputs"]["denoise"], saga.HIRES_DENOISE)
        self.assertEqual(g["17"]["inputs"]["model_name"], saga.FACE_DETECTOR)
        self.assertEqual(g["19"]["inputs"]["model_name"], saga.HAND_DETECTOR)
        self.assertEqual(g["10"]["inputs"]["images"], ["21", 0])

    def test_photo_graph_encodes_source(self):
        g = saga.build_still_graph("x", seed=1, width=896, height=1344, source_image="in.png", denoise=0.65)
        self.assertEqual(g["50"]["inputs"]["image"], "in.png")
        self.assertEqual(g["7"]["class_type"], "VAEEncode")
        self.assertEqual(g["8"]["inputs"]["denoise"], 0.65)
        plain = saga.build_still_graph("x", seed=1, width=896, height=1344, denoise=0.65)
        self.assertEqual((plain["7"]["class_type"], plain["8"]["inputs"]["denoise"]), ("EmptyLatentImage", 1.0))

    def test_run_still_photo_restyle(self):
        from io import BytesIO

        from PIL import Image

        graphs, uploads = [], []

        class _Client:
            async def _upload_image(self, data):
                uploads.append(data)
                return "up.png"

            async def _run_and_fetch(self, graph, prefer, input_name):
                graphs.append((graph, input_name))
                out = BytesIO()
                Image.new("RGB", (64, 64)).save(out, format="PNG")
                return out.getvalue(), "image/png"

        src = BytesIO()
        Image.new("RGB", (900, 1350)).save(src, format="JPEG")
        _, _, meta = asyncio.run(saga.run_still(_Client(), "", seed=3, options={"strength": "strong"},
                                                image_bytes=src.getvalue()))
        graph, input_name = graphs[0]
        self.assertEqual((len(uploads), input_name), (1, "up.png"))
        self.assertEqual(graph["8"]["inputs"]["denoise"], saga.BLANK_STRENGTHS["strong"])
        self.assertEqual(graph["5"]["inputs"]["text"], saga.STYLE_PREFIX)
        self.assertEqual((meta["saga"]["kind"], meta["saga"]["strength"]), ("photo", "strong"))
        self.assertEqual((meta["saga"]["aspect"], meta["saga"]["width"], meta["saga"]["height"]), ("photo", 1344, 2016))

        asyncio.run(saga.run_still(_Client(), "woman laughing", seed=3, options={"strength": "strong"},
                                   image_bytes=src.getvalue()))
        self.assertEqual(graphs[1][0]["8"]["inputs"]["denoise"], saga.STRENGTHS["strong"])

    def test_part_options(self):
        with self.assertRaises(saga.SagaError):
            saga.parse_still_options(layout="part")
        opts = saga.parse_still_options(layout="part", photo=True)
        self.assertEqual((opts["strength"], opts["denoise"], opts["count"]), ("light", None, 1))
        self.assertEqual((opts["keep_lines"], opts["match_skin"]), (False, False))
        opts = saga.parse_still_options(layout="part", photo=True, strength="Subtle", denoise="0.35", count="4",
                                        keep_lines="1", match_skin="1", context=True, pair=True)
        self.assertEqual((opts["strength"], opts["denoise"], opts["count"]), ("subtle", 0.35, 4))
        self.assertEqual((opts["keep_lines"], opts["match_skin"]), (True, True))
        for bad in ({"denoise": "0.9"}, {"denoise": "x"}, {"count": "5"}, {"count": "0"}):
            with self.assertRaises(saga.SagaError, msg=bad):
                saga.parse_still_options(layout="part", photo=True, **bad)
        with self.assertRaises(saga.SagaError):
            saga.parse_still_options(photo=True, context=True)
        self.assertTrue(saga.parse_still_options(photo=True, keep_size="1")["keep_size"])

    def _face(self, size=(1024, 768), box=(691, 172, 786, 314), colour=(210, 160, 140)):
        from PIL import Image

        img = Image.new("RGBA", size, (0, 0, 0, 0))
        img.paste((*colour, 255), box)
        return img

    def test_prepare_part_sheet_has_picture_and_mask(self):
        from PIL import Image

        layer = self._face()
        png, w16, h16, plan = saga.prepare_part(layer)
        x0, y0, x1, y1 = plan["region"]
        self.assertTrue(x0 < 691 and y0 < 172 and x1 > 786 and y1 > 314 and x1 - x0 < 200)
        self.assertEqual((w16 % 16, h16 % 16, max(w16, h16)), (0, 0, saga.PART_BOX))
        sheet = Image.open(BytesIO(png))
        self.assertEqual((sheet.mode, sheet.size), ("RGB", (w16 * 2, h16)))
        left, top, w, h = plan["inner"]
        cx, cy = left + w // 2, top + h // 2
        self.assertTrue(all(abs(v - t) <= 3 for v, t in zip(sheet.getpixel((cx, cy)), (210, 160, 140))))
        self.assertEqual(sheet.getpixel((w16 + cx, cy)), (255, 255, 255))
        self.assertEqual(sheet.getpixel((w16 + 1, 1)), (0, 0, 0))

        rendered = sheet.crop((0, 0, w16, h16)).resize((round(w16 * 1.5), round(h16 * 1.5)))
        out = saga.finish_part(self._png(rendered), plan)
        self.assertEqual((out.mode, out.size), ("RGB", (1024, 768)))
        r, g, b = out.getpixel((738, 243))
        self.assertTrue(abs(r - 210) < 6 and abs(g - 160) < 6 and abs(b - 140) < 6, (r, g, b))
        cut = saga.with_alpha(out, layer)
        self.assertEqual(list(cut.getchannel("A").getdata()), list(layer.getchannel("A").getdata()))

    def test_prepare_part_uses_context_and_custom_repaint(self):
        from PIL import Image

        layer = self._face()
        context = Image.new("RGBA", layer.size, (20, 40, 200, 255))
        repaint = Image.new("L", layer.size, 0)
        repaint.paste(255, (720, 280, 760, 300))
        png, w16, h16, plan = saga.prepare_part(layer, context=context, repaint=repaint)
        sheet = Image.open(BytesIO(png))
        self.assertTrue(all(abs(v - t) <= 3 for v, t in zip(sheet.getpixel((2, 2)), (20, 40, 200))))
        left, top, w, h = plan["inner"]
        self.assertEqual(sheet.getpixel((w16 + left + w // 2, top + h // 2)), (0, 0, 0))

    def test_part_without_alpha_stays_opaque(self):
        from PIL import Image

        layer = Image.new("RGBA", (200, 120), (10, 20, 30, 255))
        _, w16, h16, plan = saga.prepare_part(layer)
        self.assertEqual(plan["region"], (0, 0, 200, 120))
        rendered = self._png(Image.new("RGB", (w16, h16)))
        self.assertEqual(saga.with_alpha(saga.finish_part(rendered, plan), layer).mode, "RGB")

    def test_masked_graph(self):
        g = saga.build_still_graph("x", seed=1, width=512, height=640, hd=True, detail=False,
                                   source_image="sheet.png", denoise=0.3, masked=True)
        self.assertEqual((g["51"]["inputs"]["x"], g["52"]["inputs"]["x"], g["52"]["inputs"]["width"]), (0, 512, 512))
        self.assertEqual(g["7"]["class_type"], "SetLatentNoiseMask")
        self.assertEqual(g["54"]["inputs"]["pixels"], ["51", 0])
        self.assertEqual(g["15"]["inputs"]["latent_image"], ["55", 0])
        self.assertEqual(g["55"]["inputs"]["mask"], ["53", 0])
        self.assertNotIn("17", g)

    class _PartClient:
        """Returns a flat colour per render (cycling through colours) and records graphs."""

        def __init__(self, colours=((90, 90, 90),), tags=None):
            self.colours, self.tags, self.graphs, self.asked = list(colours), tags, [], []

        async def run_text(self, graph):
            self.asked.append(graph["2"]["inputs"]["prompt"])
            if self.tags is None:
                raise AssertionError("rewrite should not run")
            return self.tags

        async def _upload_image(self, data):
            return "sheet.png"

        async def _run_and_fetch(self, graph, prefer, input_name):
            from PIL import Image

            colour = self.colours[len(self.graphs) % len(self.colours)]
            self.graphs.append(graph)
            out = BytesIO()
            Image.new("RGB", (graph["51"]["inputs"]["width"], graph["51"]["inputs"]["height"]), colour).save(out, "PNG")
            return out.getvalue(), "image/png"

    def _png(self, img):
        buf = BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def test_run_still_part(self):
        from PIL import Image

        client = self._PartClient()
        layer = self._face(size=(300, 500), box=(50, 60, 250, 400))
        data, _, meta = asyncio.run(saga.run_still(client, "left hand, open palm", seed=3,
                                                   options={"layout": "part"}, image_bytes=self._png(layer)))
        graph = client.graphs[0]
        self.assertEqual(graph["8"]["inputs"]["denoise"], saga.STRENGTHS["light"])
        self.assertEqual(graph["7"]["class_type"], "SetLatentNoiseMask")
        self.assertEqual(graph["6"]["inputs"]["text"], saga.PART_NEGATIVE)
        self.assertTrue(graph["5"]["inputs"]["text"].endswith(
            f"left hand, open palm, {saga.PART_STYLE}, {saga.PART_SUFFIX}"))
        out = Image.open(BytesIO(data))
        self.assertEqual((out.size, out.getpixel((150, 200))), ((300, 500), (90, 90, 90, 255)))
        self.assertEqual(list(out.getchannel("A").getdata()), list(layer.getchannel("A").getdata()))
        self.assertEqual((meta["saga"]["kind"], meta["saga"]["transparent"], meta["saga"]["denoise"]),
                         ("part", True, 0.5))

    def test_part_changes_keep_strength(self):
        client = self._PartClient(tags="<think>\n\n</think>\n\nTags: blue eyes, dark lipstick, blue eyes, smiling")
        _, _, meta = asyncio.run(saga.run_still(
            client, "", seed=3, image_bytes=self._png(self._face()),
            options={"layout": "part", "part": "face", "strength": "light",
                     "changes": "make her eyes blue and give her darker lipstick"}))
        self.assertEqual(len(client.asked), 1)
        self.assertIn(saga.PART_TYPES["face"] + ", blue eyes, dark lipstick, smiling", client.graphs[0]["5"]["inputs"]["text"])
        self.assertEqual(client.graphs[0]["8"]["inputs"]["denoise"], saga.STRENGTHS["light"])
        self.assertEqual((meta["saga"]["part"], meta["saga"]["strength"]), ("face", "light"))
        self.assertTrue(meta["saga"]["prompt_used"].startswith(saga.PART_TYPES["face"]))

    def test_exact_denoise_wins(self):
        client = self._PartClient()
        _, _, meta = asyncio.run(saga.run_still(client, "", seed=1, image_bytes=self._png(self._face()),
                                                options={"layout": "part", "strength": "strong", "denoise": 0.35}))
        self.assertEqual((client.graphs[0]["8"]["inputs"]["denoise"], meta["saga"]["denoise"]), (0.35, 0.35))

    def test_variants_and_talking_pair(self):
        from PIL import Image

        layer = self._face()
        talking = layer.copy()
        talking.paste((60, 20, 20, 255), (725, 280, 750, 295))
        client = self._PartClient(colours=((90, 90, 90), (200, 10, 10)))
        outputs = asyncio.run(saga.run_still_outputs(
            client, "", seed=7, options={"layout": "part", "part": "face", "count": 2},
            inputs={"image": self._png(layer), "pair": self._png(talking)}))
        self.assertEqual([(m["seed_used"], m["saga"]["role"], m["saga"]["variant"]) for _, _, m in outputs],
                         [(7, "main", 1), (7, "pair", 1), (8, "main", 2), (8, "pair", 2)])
        self.assertEqual([g["8"]["inputs"]["seed"] for g in client.graphs], [7, 7, 8, 8])
        main, pair = (Image.open(BytesIO(d)) for d, _, _ in outputs[:2])
        self.assertEqual(main.getpixel((700, 200)), pair.getpixel((700, 200)))
        self.assertEqual(pair.getpixel((737, 287)), (200, 10, 10, 255))
        self.assertEqual(main.getpixel((737, 287)), (90, 90, 90, 255))
        self.assertEqual(list(pair.getchannel("A").getdata()), list(talking.getchannel("A").getdata()))

    def test_context_and_pair_must_match_size(self):
        with self.assertRaises(saga.SagaError):
            asyncio.run(saga.run_still_outputs(
                self._PartClient(), "", seed=1, options={"layout": "part"},
                inputs={"image": self._png(self._face()), "context": self._png(self._face(size=(800, 600)))}))

    def test_keep_lines(self):
        from PIL import Image

        original = self._face()
        original.paste((40, 25, 15, 255), (700, 200, 780, 203))
        result = Image.new("RGB", original.size, (120, 200, 120))
        out = saga.keep_lines(result, original)
        self.assertEqual(out.getpixel((740, 201)), (40, 25, 15))
        self.assertEqual(out.getpixel((740, 250)), (120, 200, 120))
        self.assertEqual(out.getpixel((10, 10)), (120, 200, 120))

    def test_match_skin_moves_skin_to_reference(self):
        from PIL import Image

        layer = self._face(colour=(248, 192, 152))
        layer.paste((40, 25, 15, 255), (700, 200, 780, 205))
        result = Image.new("RGB", layer.size, (200, 120, 80))
        result.paste((40, 25, 15), (700, 200, 780, 205))
        result.paste((200, 80, 100), (730, 280, 750, 290))
        params = saga.skin_params(result, layer, layer)
        self.assertIsNotNone(params)
        out = saga.match_skin(result, layer.getchannel("A"), params)
        self.assertTrue(all(abs(v - t) <= 4 for v, t in zip(out.getpixel((740, 260)), (248, 192, 152))),
                        out.getpixel((740, 260)))
        self.assertEqual(out.getpixel((740, 202)), (40, 25, 15))
        self.assertEqual(out.getpixel((740, 285)), (200, 80, 100))
        self.assertIs(saga.match_skin(result, layer.getchannel("A"), None), result)

    def test_bundle_round_trip(self):
        blob = saga.pack_inputs({"image": b"a", "pair": b"b"})
        self.assertTrue(saga.is_bundle(blob))
        self.assertEqual(saga.unpack_inputs(blob), ({"image": b"a", "pair": b"b"}, None))
        self.assertEqual(saga.unpack_inputs(b"\x89PNG"), ({"image": b"\x89PNG"}, None))
        self.assertEqual(saga.bundle_preview(saga.pack_inputs({"x/f.png": b"f"}, {"items": [{"image": "x/f.png"}]})), b"f")
        self.assertEqual(saga.safe_name("jenny\\faces\\f_gross.png"), "jenny/faces/f_gross.png")
        for bad in ("../x.png", "", "a/../b.png"):
            with self.assertRaises(saga.SagaError):
                saga.safe_name(bad)

    def test_check_batch(self):
        face, talk = self._png(self._face()), self._png(self._face())
        small = self._png(self._face(size=(800, 600), box=(10, 10, 50, 50)))
        files = {"f.png": face, "f_talk.png": talk, "body.png": face, "small.png": small}
        clean = saga.check_batch(files, {"items": [{"image": "f.png", "pair": "f_talk.png", "context": "body.png"},
                                                   {"image": "small.png", "part": "Hand"}]})
        self.assertEqual(clean["items"], [{"image": "f.png", "pair": ["f_talk.png"], "context": "body.png"},
                                          {"image": "small.png", "pair": [], "part": "hand"}])
        for bad in ({}, {"items": []}, {"items": [{"image": "missing.png"}]},
                    {"items": [{"image": "f.png", "pair": "small.png"}]}, {"items": [{"pair": "f.png"}]}):
            with self.assertRaises(saga.SagaError, msg=bad):
                saga.check_batch(files, bad)

    def test_run_batch(self):
        import json
        import zipfile

        from PIL import Image

        layer = self._face()
        talking = layer.copy()
        talking.paste((60, 20, 20, 255), (725, 280, 750, 295))
        files = {"f.png": self._png(layer), "f_talk.png": self._png(talking), "h.png": self._png(self._face())}
        bundle = saga.pack_inputs(files, {"items": [{"image": "f.png", "pair": "f_talk.png"}, {"image": "h.png", "part": "hand"}]})
        client, seen = self._PartClient(), []
        data, ctype, meta = asyncio.run(saga.run_batch(
            client, "", seed=11, options={"part": "face", "strength": "subtle", "keep_lines": True},
            bundle=bundle, progress=lambda done, total: seen.append((done, total))))
        self.assertEqual((ctype, meta["seed_used"], meta["saga"]["done"], seen), ("application/zip", 11, 2, [(1, 2), (2, 2)]))
        self.assertEqual({g["8"]["inputs"]["seed"] for g in client.graphs}, {11})
        self.assertEqual(client.graphs[0]["8"]["inputs"]["denoise"], saga.STRENGTHS["subtle"])
        self.assertIn(saga.PART_TYPES["hand"], client.graphs[2]["5"]["inputs"]["text"])
        with zipfile.ZipFile(BytesIO(data)) as zf:
            self.assertEqual(sorted(zf.namelist()), ["f.png", "f_talk.png", "h.png", "report.json"])
            report = json.loads(zf.read("report.json"))
            self.assertEqual(Image.open(BytesIO(zf.read("f_talk.png"))).size, (1024, 768))
        self.assertEqual((report["seed_used"], report["done"], report["failed"]), (11, 2, 0))

    def test_run_batch_stops_after_three_failures(self):
        class _Down(self._PartClient):
            async def _upload_image(self, data):
                raise RuntimeError("tunnel down")

        face = self._png(self._face())
        files = {f"{i}.png": face for i in range(5)}
        bundle = saga.pack_inputs(files, {"items": [{"image": f"{i}.png"} for i in range(5)]})
        with self.assertRaises(saga.SagaError):
            asyncio.run(saga.run_batch(_Down(), "", seed=1, options={}, bundle=bundle))

    def test_keep_size_returns_photo_size(self):
        from PIL import Image

        class _Client:
            async def _upload_image(self, data):
                return "up.png"

            async def _run_and_fetch(self, graph, prefer, input_name):
                out = BytesIO()
                Image.new("RGB", (1344, 1008)).save(out, format="PNG")
                return out.getvalue(), "image/png"

        src = self._png(Image.new("RGB", (1024, 768)))
        data, _, meta = asyncio.run(saga.run_still(_Client(), "", seed=1, image_bytes=src,
                                                   options={"keep_size": True, "quality": "fast"}))
        self.assertEqual((Image.open(BytesIO(data)).size, meta["saga"]["width"]), ((1024, 768), 1024))

    def test_rewrite_off_keeps_words(self):
        from PIL import Image

        graphs = []

        class _Client:
            async def run_text(self, graph):
                raise AssertionError("rewrite should be off")

            async def _upload_image(self, data):
                return "p.png"

            async def _run_and_fetch(self, graph, prefer, input_name):
                graphs.append(graph)
                out = BytesIO()
                Image.new("RGB", (64, 64)).save(out, format="PNG")
                return out.getvalue(), "image/png"

        asyncio.run(saga.run_still(_Client(), "a woman reading in a park", seed=1, options={"rewrite": False}))
        self.assertTrue(graphs[0]["5"]["inputs"]["text"].endswith("a woman reading in a park"))
        self.assertEqual(graphs[0]["6"]["inputs"]["text"], saga.NEGATIVE)

    def test_avoid_words_join_the_negative(self):
        client = self._PartClient()
        _, _, meta = asyncio.run(saga.run_still(client, "", seed=1, image_bytes=self._png(self._face()),
                                                options={"layout": "part", "negative": " stubble,  beard ,"}))
        self.assertEqual(client.graphs[0]["6"]["inputs"]["text"], saga.PART_NEGATIVE + ", stubble, beard")
        self.assertEqual(meta["saga"]["negative"], "stubble, beard")

    def test_avoid_is_limited(self):
        self.assertIsNone(saga.parse_still_options(negative="  ")["negative"])
        with self.assertRaises(saga.SagaError):
            saga.parse_still_options(negative="x" * (saga.NEGATIVE_MAX_CHARS + 1))
        self.assertEqual(saga.parse_motion_options(negative="beard")["negative"], "beard")

    def test_cutout_graph(self):
        g = saga.build_still_graph("x", seed=1, width=768, height=1344, cutout=True)
        self.assertEqual(g["33"]["class_type"], "JoinImageWithAlpha")
        self.assertEqual(g["10"]["inputs"]["images"], ["33", 0])

    def test_character_prompt(self):
        text = saga.character_prompt("in the kitchen", "debbie", True)
        self.assertTrue(text.startswith("ssdebbie"))
        self.assertIn(saga.CHARACTERS["debbie"]["outfit"], text)
        self.assertNotIn(saga.CHARACTERS["debbie"]["outfit"], saga.character_prompt("x", "debbie", False))
        self.assertEqual(saga.character_prompt("x", None, True), "x")

    def test_compose_sprite(self):
        from io import BytesIO

        from PIL import Image

        bg = BytesIO()
        Image.new("RGB", (200, 100), "blue").save(bg, format="PNG")
        sp = Image.new("RGBA", (50, 100), (0, 0, 0, 0))
        sp.paste(Image.new("RGBA", (20, 60), (255, 0, 0, 255)), (15, 20))
        sb = BytesIO()
        sp.save(sb, format="PNG")
        out = Image.open(BytesIO(saga.compose_sprite(bg.getvalue(), sb.getvalue()))).convert("RGB")
        self.assertEqual(out.size, (200, 100))
        self.assertEqual(out.getpixel((100, 99)), (255, 0, 0))
        self.assertEqual(out.getpixel((5, 50)), (0, 0, 255))

    def _motion(self, **opts):
        calls = {}

        class _Client:
            async def generate_video(self, image_bytes, prompt, **kw):
                calls.update(kw, prompt=prompt)
                return b"mp4", "video/mp4"

        result = asyncio.run(saga.run_motion(_Client(), b"png", "she waves", seed=1, options=opts))
        return calls, result

    def test_motion_options(self):
        self.assertEqual(saga.parse_motion_options(), {"seconds": 5, "quality": "smooth", "negative": None})
        self.assertEqual(saga.parse_motion_options(seconds="5", quality="hd"),
                         {"seconds": 5, "quality": "hd", "negative": None})
        with self.assertRaises(saga.SagaError):
            saga.parse_motion_options(seconds=10, quality="hd")
        with self.assertRaises(saga.SagaError):
            saga.parse_motion_options(quality="4k")

    def test_motion_uses_full_anisora(self):
        calls, (data, ctype, meta) = self._motion(seconds="3", quality="hd")
        self.assertEqual((data, ctype), (b"mp4", "video/mp4"))
        self.assertEqual((calls["unet_high"], calls["unet_low"]), (saga.ANISORA_FULL_HIGH, saga.ANISORA_FULL_LOW))
        self.assertEqual((calls["loras_high"], calls["loras_low"]), ([], []))
        self.assertEqual(calls["length"], 3 * saga.VIDEO_FPS + 1)
        self.assertEqual((calls["steps"], calls["cfg"], calls["width"]), (8, 1.0, 1280))
        self.assertTrue(calls["prompt"].endswith("she waves"))
        self.assertEqual(meta["saga"], {"kind": "motion", "seconds": 3, "quality": "hd", "fps": 32, "segments": 1,
                                        "negative": None})
        self.assertEqual(calls["negative"], saga.MOTION_NEGATIVE)

    def test_motion_long_graph(self):
        calls, (_, _, meta) = self._motion(seconds=10)
        self.assertEqual((calls["width"], calls["length"]), (832, 81))
        g = {
            "10": {"inputs": {"start_image": ["7", 0]}},
            "11": {"inputs": {"noise_seed": 1, "positive": ["10", 0]}},
            "12": {"inputs": {"latent_image": ["11", 0]}},
            "14": {"inputs": {"images": ["13", 0], "fps": 16}},
        }
        g = calls["graph_hook"](g)
        self.assertEqual(g["101"]["inputs"]["batch_index"], 80)
        self.assertEqual(g["110"]["inputs"]["start_image"], ["101", 0])
        self.assertEqual(g["111"]["inputs"]["noise_seed"], 2)
        self.assertEqual(g["112"]["inputs"]["latent_image"], ["111", 0])
        self.assertEqual(g["114"]["inputs"]["length"], 80)
        self.assertEqual(g["121"]["inputs"]["images"], ["115", 0])
        self.assertEqual(g["14"]["inputs"], {"images": ["121", 0], "fps": 32})
        self.assertEqual(g["10"]["inputs"]["start_image"], ["7", 0])
        self.assertEqual((meta["saga"]["fps"], meta["saga"]["segments"]), (32, 2))


ALL_LLLITE = {saga.LLLITE_INPAINT, saga.LLLITE_ANY, saga.LLLITE_LINEART}


class ControlTests(unittest.TestCase):
    """Anima LLLite structure control on part redraws."""

    def _part(self, **kw):
        return saga.parse_still_options(layout="part", photo=True, **kw)

    def test_no_control_changes_nothing(self):
        opts = self._part()
        self.assertEqual((opts["control"], opts["debug"]), ([], False))
        self.assertNotIn("control_strength", opts)
        self.assertEqual(saga.control_specs(opts), [])
        with self.assertRaises(saga.SagaError) as err:
            self._part(denoise="0.9")
        self.assertIn("0.95 with control", str(err.exception))

    def test_control_options(self):
        opts = self._part(control="gray, INPAINT", control_strength="inpaint:1.0,gray:0.7",
                          control_start="0", control_end="0.8", denoise="0.9")
        self.assertEqual(opts["control"], ["inpaint", "gray"])
        self.assertEqual(opts["control_strength"], {"inpaint": 1.0, "gray": 0.7})
        self.assertEqual((opts["control_start"], opts["control_end"], opts["denoise"]), (0.0, 0.8, 0.9))
        self.assertEqual(self._part(control="lines,gray", control_strength="0.8")["control_strength"],
                         {"lines": 0.8, "gray": 0.8})
        self.assertEqual(self._part(control=["lines"], control_model="lineart1")["control_model"], "lineart1")
        self.assertEqual(self._part(control="inpaint", denoise="0.95")["denoise"], 0.95)

    def test_control_errors(self):
        bad = [
            dict(control="depth"),
            dict(control="inpaint", control_strength="3"),
            dict(control="inpaint", control_strength="lines:0.5"),
            dict(control="inpaint", control_strength="inpaint=1"),
            dict(control="inpaint", control_start="0.8", control_end="0.5"),
            dict(control="inpaint", control_end="1.5"),
            dict(control="inpaint", denoise="0.96"),
            dict(control="lines", control_model="canny"),
            dict(control="inpaint", uploads=["control_image"]),
            dict(control="gray", uploads=["control_image_lines"]),
            dict(control="lines", uploads=["mystery"]),
        ]
        for kw in bad:
            with self.subTest(kw=kw), self.assertRaises(saga.SagaError):
                self._part(**kw)
        with self.assertRaises(saga.SagaError) as err:
            saga.parse_still_options(photo=True, control="inpaint")
        self.assertIn("Part mode", str(err.exception))
        self.assertEqual(self._part(control="gray", uploads=["control_image"])["control"], ["gray"])

    def test_specs_and_weights(self):
        specs = saga.control_specs(self._part(control="gray,lines,inpaint", control_model="lineart1",
                                              control_strength="lines:0.6", control_end="0.9"))
        self.assertEqual([(s["kind"], s["weights"], s["strength"], s.get("panel")) for s in specs], [
            ("inpaint", saga.LLLITE_INPAINT, 1.0, None),
            ("lines", saga.LLLITE_LINEART, 0.6, 2),
            ("gray", saga.LLLITE_ANY, 1.0, 3),
        ])
        self.assertTrue(all((s["start"], s["end"]) == (0.0, 0.9) for s in specs))

    def test_graph_chains_lllite_on_both_samplers(self):
        specs = saga.control_specs(self._part(control="inpaint,lines,gray"))
        g = saga.build_still_graph("x", seed=1, width=512, height=640, hd=True, detail=False,
                                   source_image="sheet.png", denoise=0.8, masked=True, controls=specs)
        self.assertEqual([g[n]["class_type"] for n in ("62", "63", "64")], [saga.LLLITE_NODE] * 3)
        self.assertEqual((g["62"]["inputs"]["model"], g["62"]["inputs"]["image"], g["62"]["inputs"]["mask"]),
                         (["40", 0], ["51", 0], ["53", 0]))
        self.assertEqual((g["63"]["inputs"]["model"], g["63"]["inputs"]["image"]), (["62", 0], ["60", 0]))
        self.assertEqual((g["64"]["inputs"]["model"], g["64"]["inputs"]["image"]), (["63", 0], ["61", 0]))
        self.assertEqual([(g[c]["inputs"]["x"], g[c]["inputs"]["width"]) for c in ("51", "52", "60", "61")],
                         [(0, 512), (512, 512), (1024, 512), (1536, 512)])
        self.assertEqual((g["8"]["inputs"]["model"], g["15"]["inputs"]["model"]), (["64", 0], ["64", 0]))
        self.assertTrue(all(g[n]["inputs"]["preserve_wrapper"] for n in ("62", "63", "64")))
        ids = [int(k) for k in g]
        self.assertEqual(len(ids), len(set(ids)))

    def test_graph_without_control_is_unchanged(self):
        g = saga.build_still_graph("x", seed=1, width=512, height=640, hd=True, detail=False,
                                   source_image="sheet.png", denoise=0.5, masked=True, controls=[])
        self.assertFalse(any(n["class_type"] == saga.LLLITE_NODE for n in g.values()))
        self.assertEqual((g["8"]["inputs"]["model"], g["15"]["inputs"]["model"]), (["40", 0], ["40", 0]))
        with self.assertRaises(ValueError):
            saga.build_still_graph("x", seed=1, width=512, height=640,
                                   controls=saga.control_specs(self._part(control="gray")))

    def _layer(self):
        from PIL import Image

        layer = Image.new("RGBA", (900, 700), (0, 0, 0, 0))
        layer.paste((220, 170, 150, 255), (300, 200, 600, 500))
        layer.paste((20, 20, 20, 255), (400, 300, 410, 310))
        return layer

    @staticmethod
    def _dark_centre(img, x0, x1):
        import numpy as np

        a = np.asarray(img.convert("L"), dtype=np.float32)[:, x0:x1]
        ys, xs = np.nonzero(a < 60)
        return float(xs.mean()), float(ys.mean())

    def test_sheet_panels_line_up(self):
        from PIL import Image

        layer = self._layer()
        upload = Image.new("RGBA", layer.size, (0, 0, 0, 0))
        upload.paste((0, 0, 0, 255), (400, 300, 410, 310))
        png, w, h, plan = saga.prepare_part(layer, controls=[("lines", None), ("gray", upload)])
        sheet = Image.open(BytesIO(png))
        self.assertEqual(sheet.size, (w * 4, h))
        spots = [self._dark_centre(sheet, w * i, w * (i + 1)) for i in (0, 2, 3)]
        for x, y in spots[1:]:
            self.assertLessEqual(abs(x - spots[0][0]) + abs(y - spots[0][1]), 1.0)
        plain, w2, h2, plan2 = saga.prepare_part(layer)
        self.assertEqual((Image.open(BytesIO(plain)).size, plan2), ((w * 2, h), plan))
        self.assertEqual(Image.open(BytesIO(plain)).tobytes(), sheet.crop((0, 0, w * 2, h)).tobytes())

    def test_derived_lines_and_gray(self):
        from PIL import Image

        img = Image.new("RGBA", (4, 1))
        img.putdata([(50, 50, 50, 255), (200, 200, 200, 255), (90, 90, 90, 255), (10, 10, 10, 0)])
        self.assertEqual([p[0] for p in saga.control_lines(img).getdata()], [0, 255, 128, 255])
        gray = saga.control_gray(Image.new("RGB", (1, 1), (200, 100, 50))).getpixel((0, 0))
        self.assertEqual(gray, (124, 124, 124))

    def test_uploads_and_available(self):
        self.assertEqual(saga.control_uploads({"control_image": "both", "control_image_gray": "g"}),
                         {"lines": "both", "gray": "g"})
        self.assertEqual(saga.available_controls(None), list(saga.CONTROLS))
        self.assertEqual(saga.available_controls(set()), [])
        self.assertEqual(saga.available_controls({saga.LLLITE_INPAINT, saga.LLLITE_LINEART}), ["inpaint", "lines"])
        cfg = saga.public_config(set(), {saga.LLLITE_INPAINT})
        self.assertEqual((cfg["controls"], cfg["control_denoise_max"]), (["inpaint"], 0.95))

    class _Client(GraphTests._PartClient):
        def __init__(self, lllite=frozenset(ALL_LLLITE), **kw):
            super().__init__(**kw)
            self.lllite = lllite

        async def node_choices(self, node, field):
            assert (node, field) == (saga.LLLITE_NODE, "lllite_name")
            return None if self.lllite is None else set(self.lllite)

    def _png(self, img):
        return GraphTests._png(None, img)

    def test_run_with_control_pair_and_debug(self):
        from PIL import Image

        layer = self._layer()
        talking = layer.copy()
        talking.paste((90, 10, 10, 255), (420, 400, 480, 420))
        client = self._Client(colours=((90, 90, 90), (200, 10, 10)))
        outputs = asyncio.run(saga.run_still_outputs(
            client, "", seed=5, inputs={"image": self._png(layer), "pair": self._png(talking),
                                        "control_image_lines": self._png(layer)},
            options={"layout": "part", "control": "inpaint,lines,gray", "control_strength": "lines:0.8",
                     "control_end": "0.9", "denoise": "0.85", "debug": "1", "rewrite": "0"}))
        self.assertEqual([m["saga"]["role"] for _, _, m in outputs], ["main", "pair", "debug"])
        self.assertEqual(len(client.graphs), 2)
        for g in client.graphs:
            self.assertEqual(g["8"]["inputs"]["model"], ["64", 0])
            self.assertEqual(g["8"]["inputs"]["denoise"], 0.85)
            self.assertEqual(g["60"]["inputs"]["x"], g["51"]["inputs"]["width"] * 2)
        meta = outputs[0][2]["saga"]
        self.assertEqual((meta["control"], meta["control_strength"], meta["control_start"], meta["control_end"]),
                         (["inpaint", "lines", "gray"], {"inpaint": 1.0, "lines": 0.8, "gray": 1.0}, 0.0, 0.9))
        self.assertEqual((meta["control_model"], meta["control_images"]), ("any", ["control_image_lines"]))
        debug, _, dmeta = outputs[2]
        sheet = Image.open(BytesIO(debug))
        self.assertEqual(dmeta["saga"]["panels"], ["picture", "mask", "lines", "gray"])
        self.assertEqual(sheet.size, (dmeta["saga"]["width"], dmeta["saga"]["height"]))

    def test_no_control_needs_no_lllite(self):
        client = GraphTests._PartClient()
        outputs = asyncio.run(saga.run_still_outputs(
            client, "", seed=5, inputs={"image": self._png(self._layer())}, options={"layout": "part"}))
        self.assertEqual(len(outputs), 1)
        self.assertEqual(outputs[0][2]["saga"]["control"], [])
        self.assertFalse(any(n["class_type"] == saga.LLLITE_NODE for n in client.graphs[0].values()))

    def test_missing_lllite_fails_the_job(self):
        cases = [(None, "AnimaLLLiteApply_sdscripts node"),
                 ({saga.LLLITE_INPAINT}, saga.LLLITE_ANY)]
        for lllite, missing in cases:
            client = self._Client(lllite=lllite)
            with self.subTest(lllite=lllite), self.assertRaises(saga.SagaError) as err:
                asyncio.run(saga.run_still_outputs(
                    client, "", seed=1, inputs={"image": self._png(self._layer())},
                    options={"layout": "part", "control": "inpaint,gray"}))
            self.assertEqual(str(err.exception), f"Anima LLLite not installed on the GPU: {missing}"
                             + (" (custom_nodes/ComfyUI-Anima-LLLite)" if lllite is None else ""))
            self.assertEqual(client.graphs, [])

    def test_control_image_must_match_size(self):
        from PIL import Image

        with self.assertRaises(saga.SagaError) as err:
            asyncio.run(saga.run_still_outputs(
                self._Client(), "", seed=1, options={"layout": "part", "control": "lines"},
                inputs={"image": self._png(self._layer()),
                        "control_image": self._png(Image.new("RGB", (100, 100), "white"))}))
        self.assertIn("control_image", str(err.exception))

    def test_batch_controls(self):
        from PIL import Image

        layer = self._layer()
        files = {"a.png": self._png(layer), "a_lines.png": self._png(Image.new("RGB", layer.size, "white")),
                 "b.png": self._png(layer)}
        manifest = {"items": [{"image": "a.png", "control_image": "a_lines.png"}, {"image": "b.png"}]}
        client = self._Client()
        data, _, meta = asyncio.run(saga.run_batch(
            client, "", seed=4, bundle=saga.pack_inputs(files, manifest),
            options={"control": "inpaint,lines", "denoise": "0.8", "rewrite": "0"}))
        self.assertEqual((meta["saga"]["control"], meta["saga"]["done"]), (["inpaint", "lines"], 2))
        self.assertEqual(meta["saga"]["control_images"], ["control_image"])
        self.assertEqual([g["63"]["class_type"] for g in client.graphs], [saga.LLLITE_NODE] * 2)
        with self.assertRaises(saga.SagaError):
            saga.check_batch({"a.png": files["a.png"], "c.png": self._png(Image.new("RGB", (10, 10)))},
                             {"items": [{"image": "a.png", "control_image": "c.png"}]})


if __name__ == "__main__":
    unittest.main()
