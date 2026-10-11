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
        with self.assertRaises(saga.SagaError):
            saga.parse_seconds("4")
        with self.assertRaises(saga.SagaError):
            saga.parse_seconds("abc")


class GraphTests(unittest.TestCase):
    def test_still_graph_uses_anima_and_style_lora(self):
        g = saga.build_still_graph("a woman in a kitchen", seed=7, width=1344, height=768)
        self.assertEqual(g["1"]["inputs"]["unet_name"], saga.ANIMA_UNET)
        self.assertEqual(g["2"]["inputs"]["type"], "stable_diffusion")
        self.assertEqual(g["4"]["inputs"]["lora_name"], saga.STYLE_LORA)
        self.assertIn(saga.STYLE_TRIGGER, g["5"]["inputs"]["text"])
        self.assertTrue(g["5"]["inputs"]["text"].endswith("a woman in a kitchen"))
        self.assertIn("child", g["6"]["inputs"]["text"])
        self.assertEqual((g["7"]["inputs"]["width"], g["7"]["inputs"]["height"]), (1344, 768))
        self.assertEqual(g["8"]["inputs"]["sampler_name"], "er_sde")

    def test_motion_uses_anisora_on_high_noise(self):
        calls = {}

        class _Client:
            async def generate_video(self, image_bytes, prompt, **kw):
                calls.update(kw, prompt=prompt)
                return b"mp4", "video/mp4"

        data, ctype, meta = asyncio.run(
            saga.run_motion(_Client(), b"png", "she waves", seed=1, seconds="3")
        )
        self.assertEqual((data, ctype), (b"mp4", "video/mp4"))
        self.assertEqual(calls["loras_high"], [(saga.ANISORA_HIGH, 1.0)])
        self.assertEqual(calls["length"], 3 * saga.VIDEO_FPS + 1)
        self.assertEqual((calls["steps"], calls["cfg"]), (8, 1.0))
        self.assertTrue(calls["prompt"].endswith("she waves"))
        self.assertEqual(meta["saga"], {"kind": "motion", "seconds": 3})


if __name__ == "__main__":
    unittest.main()
