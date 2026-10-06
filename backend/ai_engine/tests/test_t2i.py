"""Unit tests for text-to-image planning and graph."""

from __future__ import annotations

import unittest

from backend import t2i
from backend.workflows_klein import build_klein_prompt, edit_size
from backend.workflows_wan import build_t2i_prompt

CFG = {
    **t2i._DEFAULTS,
    "engine": "flux",
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
        self.assertEqual(
            pub["characters"],
            [{"id": "zara", "name": "Zara", "installed": False, "engine": "flux", "face_fix": False}],
        )
        self.assertTrue(pub["realism_ready"])
        self.assertNotIn("safetensors", str(pub))
        self.assertEqual(pub["default_engine"], "flux")
        self.assertFalse(pub["klein_ready"])


class KleinPlanTests(unittest.TestCase):
    KCFG = {**CFG, "engine": "klein"}

    def test_klein_skips_flux_realism_loras(self) -> None:
        p = t2i.plan_t2i("woman in a cafe", cfg=self.KCFG, installed=None)
        self.assertEqual(p.engine, "klein")
        self.assertEqual(p.loras, [])
        self.assertNotIn("amateurish photo", p.prompt)
        self.assertTrue(p.prompt.endswith("Phone photo."))

    def test_engine_override_and_character_forces_flux(self) -> None:
        self.assertEqual(t2i.plan_t2i("x", cfg=CFG, installed=None, engine="klein").engine, "klein")
        p = t2i.plan_t2i("x", cfg=self.KCFG, installed=None, character_id="zara")
        self.assertEqual(p.engine, "flux")
        self.assertEqual(p.loras[0][0], "char_zara_v1.safetensors")
        self.assertEqual(t2i.resolve_engine(self.KCFG, "bogus", None), "flux")

    KAVYA = {
        "id": "kavya", "name": "Kavya", "lora": "kavya_klein.safetensors", "strength": 0.8,
        "trigger": "kavya woman", "engine": "klein", "ref_gridfs_id": "abc123", "face_fix": True,
        "loras": [{"file": "klein_real.safetensors", "strength": 0.6}], "suffix": "Candid phone photo.",
    }

    def test_klein_character_uses_klein_with_ref_and_face_fix(self) -> None:
        cfg = {**CFG, "characters": CFG["characters"] + [self.KAVYA]}
        p = t2i.plan_t2i("walking on a beach", cfg=cfg, installed=None, character_id="kavya")
        self.assertEqual(p.engine, "klein")
        self.assertEqual([l[0] for l in p.loras], ["kavya_klein.safetensors", "klein_real.safetensors"])
        self.assertEqual(p.loras[0][1], 0.8)
        self.assertTrue(p.prompt.startswith("kavya woman, walking on a beach. Candid phone photo."))
        self.assertEqual(p.ref_gridfs_id, "abc123")
        self.assertIn("kavya woman", p.face_fix["prompt"])
        off = t2i.plan_t2i("x", cfg=cfg, installed=None, character_id="kavya", face_fix=False)
        self.assertIsNone(off.face_fix)
        self.assertEqual(off.ref_gridfs_id, "abc123")
        pub = t2i.public_config(cfg, None)["characters"][1]
        self.assertEqual((pub["engine"], pub["face_fix"]), ("klein", True))
        self.assertNotIn("abc123", str(t2i.public_config(cfg, None)))

    def test_edit_prompt_appends_keep_face_once(self) -> None:
        out = t2i.edit_prompt("change her top to a red saree", self.KCFG)
        self.assertTrue(out.startswith("change her top to a red saree."))
        self.assertIn("identity exactly the same", out)
        kept = "Beach dress, keep her face the same"
        self.assertEqual(t2i.edit_prompt(kept, self.KCFG), kept)


class KleinGraphTests(unittest.TestCase):
    def test_edit_graph_has_reference_latent_and_gguf(self) -> None:
        g = build_klein_prompt(
            positive="a woman", width=832, height=1216, seed=3,
            loras=[("klein_consistency_v2.safetensors", 0.8)], ref_image_name="in.png",
        )
        types = [n["class_type"] for n in g.values()]
        self.assertEqual(g["1"]["class_type"], "UnetLoaderGGUF")
        self.assertIn("ReferenceLatent", types)
        self.assertIn("LoraLoaderModelOnly", types)
        self.assertEqual(g["7"]["inputs"]["positive"], ["33", 0])
        self.assertEqual(g["7"]["inputs"]["model"], ["20", 0])
        self.assertEqual(g["9"]["inputs"]["steps"], 4)

    def test_t2i_graph_has_no_image(self) -> None:
        g = build_klein_prompt(positive="a woman", width=832, height=1216, seed=3)
        types = [n["class_type"] for n in g.values()]
        self.assertNotIn("LoadImage", types)
        self.assertEqual(g["7"]["inputs"]["positive"], ["5", 0])

    def test_face_fix_details_only_largest_face_with_reference(self) -> None:
        g = build_klein_prompt(
            positive="a woman", width=832, height=1216, seed=3, ref_image_name="in.png",
            face_fix={"prompt": "her face", "denoise": 0.4, "steps": 8},
        )
        self.assertEqual(g["54"]["inputs"]["take_count"], 1)
        self.assertEqual(g["55"]["class_type"], "DetailerForEach")
        self.assertEqual(g["55"]["inputs"]["positive"], ["52", 0])
        self.assertEqual(g["52"]["inputs"]["latent"], ["32", 0])
        self.assertEqual(g["14"]["inputs"]["images"], ["55", 0])
        plain = build_klein_prompt(positive="a woman", width=832, height=1216, seed=3)
        self.assertEqual(plain["14"]["inputs"]["images"], ["13", 0])

    def test_edit_size_keeps_aspect_at_1mp(self) -> None:
        w, h = edit_size(832, 1216)
        self.assertEqual((w % 16, h % 16), (0, 0))
        self.assertAlmostEqual(w / h, 832 / 1216, places=1)
        self.assertLessEqual(w * h, 1024 * 1024)


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
