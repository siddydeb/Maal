"""Saga studio: adult-only prompt guard, options, and the Anima still graph."""

from __future__ import annotations

import asyncio
import unittest

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
                                "layout": "scene", "background": None})
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
        self.assertEqual(saga.parse_motion_options(), {"seconds": 5, "quality": "smooth"})
        self.assertEqual(saga.parse_motion_options(seconds="5", quality="hd"), {"seconds": 5, "quality": "hd"})
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
        self.assertEqual(meta["saga"], {"kind": "motion", "seconds": 3, "quality": "hd", "fps": 32, "segments": 1})

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


if __name__ == "__main__":
    unittest.main()
