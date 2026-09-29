"""Unit tests for text-to-image planning and graph."""

from __future__ import annotations

import unittest

from backend import t2i
from backend.workflows_wan import build_t2i_prompt

CFG = {
    **t2i._DEFAULTS,
    "aspects": {
        "portrait": {"label": "2:3", "size": [832, 1216]},
        "square": {"label": "1:1", "size": [1024, 1024]},
    },
    "default_aspect": "portrait",
    "prompt_suffix": "Phone photo.",
    "loras": [
        {"file": "real.safetensors", "strength": 0.8, "trigger": "amateurish photo"},
        {"file": "sameface.safetensors", "strength": -0.8, "only_without_character": True},
        {"file": "unlock.safetensors", "strength": 0.7, "trigger": "unlockword", "when": r"\bnude\b"},
    ],
    "characters": [
        {"id": "zara", "name": "Zara", "lora": "char_zara_v1.safetensors", "trigger": "zara_v1 woman"},
    ],
    "outfits": [{"id": "red", "name": "Red dress", "text": "She wears a red dress."}],
}


class T2IPlanTests(unittest.TestCase):
    def test_random_girl_uses_sameface_and_default_aspect(self) -> None:
        p = t2i.plan_t2i("woman in a cafe", cfg=CFG, installed=None)
        self.assertEqual((p.width, p.height), (832, 1216))
        self.assertEqual(
            [fn for fn, _, _ in p.loras], ["real.safetensors", "sameface.safetensors"]
        )
        self.assertEqual(p.loras[1][1], -0.8)
        self.assertTrue(p.prompt.startswith("amateurish photo."))
        self.assertIn("woman in a cafe.", p.prompt)
        self.assertTrue(p.prompt.endswith("Phone photo."))

    def test_character_drops_sameface_and_leads_stack(self) -> None:
        p = t2i.plan_t2i(
            "on a beach", cfg=CFG, installed=None, character_id="zara", outfit_id="red", aspect="square"
        )
        self.assertEqual(
            [fn for fn, _, _ in p.loras], ["char_zara_v1.safetensors", "real.safetensors"]
        )
        self.assertIn("zara_v1 woman,", p.prompt)
        self.assertIn("She wears a red dress.", p.prompt)
        self.assertEqual((p.width, p.height), (1024, 1024))

    def test_conditional_lora_and_missing_files(self) -> None:
        p = t2i.plan_t2i("nude on a bed", cfg=CFG, installed={"unlock.safetensors"})
        self.assertEqual([fn for fn, _, _ in p.loras], ["unlock.safetensors"])
        self.assertIn("unlockword", p.prompt)
        self.assertEqual(len(p.warnings), 2)

    def test_uninstalled_character_is_an_error(self) -> None:
        with self.assertRaises(t2i.T2IError):
            t2i.plan_t2i("x", cfg=CFG, installed={"real.safetensors"}, character_id="zara")
        with self.assertRaises(t2i.T2IError):
            t2i.plan_t2i("x", cfg=CFG, installed=None, character_id="nobody")

    def test_public_config_hides_filenames(self) -> None:
        pub = t2i.public_config(CFG, {"real.safetensors", "sameface.safetensors"})
        self.assertEqual(pub["characters"], [{"id": "zara", "name": "Zara", "installed": False}])
        self.assertTrue(pub["realism_ready"])
        self.assertNotIn("safetensors", str(pub))


class T2IGraphTests(unittest.TestCase):
    def test_graph_is_empty_latent_full_denoise_with_negative_lora(self) -> None:
        g = build_t2i_prompt(
            positive="a woman",
            flux_unet="flux1-dev-fp8.safetensors",
            flux_clip_l="clip_l.safetensors",
            flux_t5="t5xxl_fp8_e4m3fn.safetensors",
            flux_vae="ae.safetensors",
            width=830,
            height=1216,
            steps=30,
            guidance=3.0,
            seed=7,
            loras=[("real.safetensors", 0.8, 0.8), ("sameface.safetensors", -0.8, -0.8)],
        )
        types = [n["class_type"] for n in g.values()]
        self.assertIn("EmptySD3LatentImage", types)
        self.assertNotIn("LoadImage", types)
        self.assertEqual(types.count("LoraLoader"), 2)
        latent = next(n for n in g.values() if n["class_type"] == "EmptySD3LatentImage")
        self.assertEqual(latent["inputs"]["width"], 816)
        ks = next(n for n in g.values() if n["class_type"] == "KSampler")
        self.assertEqual(ks["inputs"]["denoise"], 1.0)
        self.assertEqual(ks["inputs"]["scheduler"], "beta")
        neg = [n for n in g.values() if n["class_type"] == "LoraLoader"][1]
        self.assertEqual(neg["inputs"]["strength_model"], -0.8)


if __name__ == "__main__":
    unittest.main()
