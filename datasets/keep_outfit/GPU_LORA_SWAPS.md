# GPU LoRA swaps

Image wet/sheer and image act LoRAs were removed from the product.

- Clothed reshape: `private/lora_files.py` → `flux_kontext_figure_reshape_v1`
- Fluid/facial (img2img on Kontext UNET): `private/lora_files.py` → `flux_kontext_fluid_v1` (Cumifier Kontext; tag `fluid_i2i`)
- Video oral/sex: `private/lora_stack.py` (Wan 2.2 I2V)
