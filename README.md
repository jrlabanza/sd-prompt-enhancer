# sd-forge-prompt-enhancer

A **guided prompt builder** for Illustrious-XL and NoobAI XL on Forge. You pick
from human-readable dropdowns (Pose: "sitting", Shot: "upper body", Style:
"soft watercolor") and the extension translates each choice into the booru tags
those models were trained on. You never need to know what "cowboy shot" or
"rim lighting" means as a tag.

## How to use

1. Open the **Prompt Enhancer** tab in Forge.
2. (Optional) Pick a **Scenario template** at the top — "Beach vacation",
   "School day", "Cyberpunk night" — and click *Apply scenario*. Every section
   below fills with a sensible starting point.
3. Tweak any section: subject, pose, framing, outfit, scene, mood, art style.
4. Click **Build prompt →**.
5. Click **Send to txt2img** (or img2img) to paste both prompts into that tab.

You can also design from scratch by leaving the scenario as
"(none — design from scratch)" and filling each section yourself.

## What's inside

### 9 sections (each is a dropdown):

| # | Section | Examples |
|---|---|---|
| 1 | Subject / Character | Solo / 2girls / group · 1girl / 1boy / mature / child · character name |
| 2 | Pose & Action | Standing / Sitting on chair / Lying on back · Looking at viewer / Holding cup / Peace sign |
| 3 | Framing | Portrait / Upper body / Cowboy shot / Full body · From above / Dutch angle |
| 4 | Outfit | 30+ presets: school uniform, kimono, swimsuit, fantasy armor, cyberpunk... |
| 5 | Scene / Time / Lighting | 30+ locations · 9 time-of-day options · 10 lighting moods |
| 6 | Expression & Mood | Smile / Smug / Sad / Determined / Sleepy ... |
| 7 | Art Style (model-aware) | Detailed anime / Watercolor / Cinematic / Manga / Cel-shaded / Ghibli-esque ... |
| 8 | Quality, Negative, Booru meta | Best/Standard/Minimal tiers · year/source/rating tags |
| 9 | Custom extras | Free-text positive/negative additions (LoRA tags, artist tags, etc.) |

### Scenario templates (one-click presets)

Character portrait, Full-body reference sheet, School day, Cafe outing,
Beach vacation, Combat scene, Magical girl, Idol performance, Summer festival,
Cozy bedroom, Office worker, Cyberpunk night, Sci-fi space, Fantasy adventurer,
Sakura, Rainy day, Winter snow, Goth portrait.

### Canonical Illustrious tag order (what we emit)

```
subject -> character_name -> outfit -> pose/action -> framing
       -> location -> time/lighting -> mood -> art-style
       -> year/source/rating -> quality -> custom extras
```

This ordering matches the booru caption ordering Illustrious / NoobAI were
trained on, which tends to give the most predictable results.

## Supported model families

- **Illustrious-XL** (and any merge with `illustrious` in the filename)
- **NoobAI XL** (v-pred and eps)

Detection reads `shared.opts.sd_model_checkpoint` and matches against the
filename. Override in the dropdown if the auto-detect picks wrong.

## Install

Already installed at `extensions/sd-forge-prompt-enhancer`. Restart Forge to
register the tab.

## Files

- `install.py` — stub (no external deps; gradio is provided by Forge).
- `scripts/prompt_enhancer.py` — guided builder UI + tag dictionaries +
  scenario templates + paste-params wiring.
