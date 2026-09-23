/** Generic fallbacks. Real action list is loaded after unlock from GET /api/presets. */

export const ACTION_PRESETS = [
  {
    id: "nude",
    label: "Nude",
    hint: "undress · keep her face",
    mode: "img",
    prompt:
      "Photorealistic edit of the exact woman in the start image. Keep her exact face and identity. Remove all clothing so she is fully nude. Same pose, framing, lighting, and background.",
  },
  {
    id: "cumshot_clothes",
    label: "Cumshot",
    hint: "face · keep clothes",
    mode: "img",
    prompt:
      "Photorealistic edit of the exact woman in the start image. Keep her exact face, clothes, pose, and background. Only add a heavy visible facial cumshot.",
  },
  {
    id: "cumshot_nude",
    label: "Cumshot Nude",
    hint: "naked · body drenched",
    mode: "img",
    prompt:
      "Photorealistic edit of the exact woman in the start image. Keep her exact face. Remove all clothing. Cover her nude body in a heavy cumshot of realistic semen.",
  },
  {
    id: "enhance_boobs",
    label: "Boobs",
    hint: "cleavage · clothed",
    mode: "img",
    prompt:
      "Photorealistic edit of the exact woman in the start image. Keep face, identity, and the same opaque outfit. Make breasts clearly much larger under the same clothes.",
  },
  {
    id: "enhance_ass",
    label: "Ass",
    hint: "hips · clothed",
    mode: "img",
    prompt:
      "Photorealistic edit of the exact woman in the start image. Keep face and the same clothes. Make her ass and hips clearly larger under the same outfit.",
  },
  {
    id: "enhance",
    label: "Enhance",
    hint: "boobs + ass · clothed",
    mode: "img",
    prompt:
      "Photorealistic edit of the exact woman in the start image. Keep face and the same outfit. Make breasts and ass/hips clearly larger under the same clothes.",
  },
  {
    id: "oral",
    label: "Oral",
    hint: "full BJ sequence",
    mode: "vid",
    prompt:
      "bl0wj0b, PENISLORA. Photorealistic video of the exact woman in the start image giving a complete blowjob and deepthroat to one man in this same scene: a man with visible torso and hips is with her, exactly one erect penis attached to his body (never floating or detached), she takes that connected penis fully into her mouth and deepthroats him, then keeps giving a full continuous blowjob for the entire clip — lips sealed on the shaft, rhythmic head bobbing, repeated deep in-and-out strokes with visible full-shaft travel again and again, not tip-only and not frozen. Keep her exact same face, hair, expression, clothes, and background; same camera angle and framing; no jumpcut, no kneeling teleport, no pose swap; sharp face every frame; one continuous shot.",
  },
  {
    id: "jingle",
    label: "Jingle",
    hint: "breast bounce · clothed",
    mode: "vid",
    prompt:
      "Photorealistic video of the exact woman in the start image: her breasts bounce, shake, jiggle and sway continuously for the whole clip — clear soft heavy bounce under the same clothes, natural physics, not frozen and not a tiny micro-wiggle. Keep her exact same face, hair, expression, clothes, body proportions, and background; same camera angle and framing; no jumpcut, no pose swap, no outfit change; sharp face every frame; one continuous shot.",
  },
  {
    id: "twerk",
    label: "Twerk",
    hint: "ass shake · clothed",
    mode: "vid",
    prompt:
      "Photorealistic video of the exact woman in the start image: twerking and ass shaking — twerks causing her ass to jiggle and shake continuously for the whole clip, clear heavy booty bounce under the same clothes, natural physics, not frozen. Keep her exact same face, hair, expression, clothes, body proportions, and background; same camera angle and framing; no jumpcut, no pose swap, no outfit change; sharp face every frame; one continuous shot.",
  },
  {
    id: "sex",
    label: "Sex",
    hint: "missionary",
    mode: "vid",
    prompt:
      "m15510n4ry, PENISLORA. Photorealistic video of the exact woman in the start image having missionary sex with one man in this same scene: a man with visible torso and hips is with her, exactly one erect penis attached to his body (never floating or detached), he is on top, that connected penis enters her vagina, then continuous in-and-out thrusting for the entire clip — full-shaft travel again and again, not frozen and not a tiny wiggle. Keep her exact same face, hair, expression, and background; same camera angle and framing; no jumpcut, no pose swap; sharp face every frame; one continuous shot.",
  },
];

export function setActionPresets(list) {
  if (!Array.isArray(list) || !list.length) return;
  ACTION_PRESETS.splice(0, ACTION_PRESETS.length, ...list);
}

export function presetsForMode(mode) {
  return ACTION_PRESETS.filter((p) => p.mode === mode);
}

export function presetById(id) {
  return ACTION_PRESETS.find((p) => p.id === id) || null;
}
