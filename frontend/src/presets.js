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
    hint: "with a man",
    mode: "vid",
    prompt:
      "Only change: she is already mid continuous blowjob on one man. Keep her exact same face. One continuous shot. PENISLORA.",
  },
  {
    id: "sex",
    label: "Sex",
    hint: "missionary",
    mode: "vid",
    prompt:
      "Only change: missionary sex; man on top; erect penis entering her; continuous thrusting. One continuous shot, no cuts. PENISLORA.",
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
