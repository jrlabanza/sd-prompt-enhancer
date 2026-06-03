"""
sd-forge-prompt-enhancer
========================

Guided prompt builder for Illustrious-XL / NoobAI XL on Forge.

The user picks from human-readable dropdowns (Pose: "sitting", Shot:
"upper body", Style: "soft watercolor") and we translate each choice into
the booru tags those models were trained on. Output is assembled in
Illustrious canonical tag order:
  subject -> outfit -> pose/action -> framing -> location -> time/lighting
  -> mood -> art-style -> year/source/rating -> quality -> custom extras

Send-to-txt2img/img2img uses Forge's paste-params API.
"""
from __future__ import annotations

import logging
import re
from typing import List, Tuple, Dict

import gradio as gr

try:
    from modules import script_callbacks, shared
except ImportError:
    script_callbacks = None  # type: ignore
    shared = None  # type: ignore

logger = logging.getLogger(__name__)
TAG = "[prompt-enhancer]"

NONE = "(skip)"

# ===========================================================================
# 1. Subject — single merged dropdown so count/type can never conflict.
# ===========================================================================

SUBJECTS = {
    "Solo - 1girl (young adult female)": "1girl, solo",
    "Solo - 1boy (young adult male)":    "1boy, solo",
    "Solo - mature woman":               "1girl, solo, mature_female",
    "Solo - elderly":                    "1other, solo, old",
    "Solo - female child":               "1girl, solo, child",
    "Solo - male child":                 "1boy, solo, child",
    "Solo - non-human creature":         "1other, solo",
    "Pair - 2girls":                     "2girls",
    "Pair - 2boys":                      "2boys",
    "Pair - 1girl + 1boy":               "1girl, 1boy",
    "Trio - 3girls":                     "3girls",
    "Trio - 3boys":                      "3boys",
    "Trio - 2girls + 1boy":              "2girls, 1boy",
    "Trio - 1girl + 2boys":              "1girl, 2boys",
    "Group - multiple girls":            "multiple_girls",
    "Group - multiple boys":             "multiple_boys",
    "Group - mixed":                     "multiple_girls, multiple_boys",
    NONE:                                "",
}

# ===========================================================================
# 2. Pose / Action
# ===========================================================================

POSES = {
    "Standing":                   "standing",
    "Standing, hands on hips":    "standing, hands_on_hips",
    "Standing, arms crossed":     "standing, crossed_arms",
    "Standing, hands behind back":"standing, hands_behind_back",
    "Standing, leaning forward":  "standing, leaning_forward",
    "Standing, hands in pockets": "standing, hand_in_pocket",
    "Walking":                    "walking",
    "Running":                    "running",
    "Sitting on chair":           "sitting, on_chair",
    "Sitting on floor":           "sitting, on_floor",
    "Sitting cross-legged":       "sitting, indian_style",
    "Sitting on knees (seiza)":   "sitting, seiza",
    "Kneeling":                   "kneeling",
    "Lying on back":              "lying, on_back",
    "Lying on stomach":           "lying, on_stomach",
    "Lying on side":              "lying, on_side",
    "Jumping":                    "jumping",
    "Falling":                    "falling",
    "Flying":                     "flying",
    "Dancing":                    "dancing",
    "Crouching":                  "squatting",
    "Hugging knees":              "knees_to_chest, hugging_own_legs",
    "Combat / fighting stance":   "fighting_stance",
    "Spell-casting":              "casting_spell, magical_aura",
    "Holding weapon":             "holding_weapon",
    "Leaning against wall":       "leaning, against_wall",
    "Stretching arms above head": "arms_up, stretching",
    "Twisting / looking back":    "twisting, looking_back",
    # --- [NSFW] poses ---
    "[NSFW] Bent over (suggestive)":     "bent_over, looking_at_viewer",
    "[NSFW] All fours / on hands & knees":"all_fours, on_hands_and_knees",
    "[NSFW] Spreading legs":             "spread_legs, presenting",
    "[NSFW] Top-down bottom-up":         "top-down_bottom-up",
    "[NSFW] Reclining on bed":           "lying, on_back, on_bed, pillow",
    "[NSFW] M-shaped legs":              "spread_legs, m_legs",
    "[NSFW] Embracing partner":          "hug, embracing",
    "[NSFW] Kissing":                    "kissing, kiss",
    "[NSFW] Cowgirl position":           "cowgirl_position",
    "[NSFW] Touching self":              "masturbation, fingering",
    "[NSFW] Squatting (revealing)":      "squatting, spread_legs",
    # --- More safe poses ---
    "Standing on tip-toes":              "standing, tiptoe",
    "Standing, one leg up":              "standing, leg_up",
    "Standing, one knee up":             "standing, knee_up",
    "Sitting at desk":                   "sitting, on_chair, desk",
    "Sitting on bed":                    "sitting, on_bed",
    "Sitting on window":                 "sitting, on_windowsill",
    "Sitting, arms back":                "sitting, arms_behind_back",
    "Crouching, hands on ground":        "crouching, hands_on_ground",
    "Hands clasped (prayer)":            "own_hands_together, praying",
    "Yoga pose":                         "yoga, balance",
    "Stretching back / arching":         "arched_back, stretching",
    "Curled up":                         "curled_up, fetal_position",
    "Star pose (arms and legs out)":     "spread_arms, spread_legs, star_pose",
    "Heroic / superhero pose":           "standing, heroic_pose, hands_on_hips",
    "Defensive / guard stance":          "fighting_stance, defensive",
    "Riding (horse / vehicle)":          "riding",
    "Skating / ice skating":             "skating, ice_skating",
    "Skateboarding":                     "skateboard, riding",
    "Bicycling":                         "bicycle, riding",
    "Surfing":                           "surfboard, surfing",
    "Diving":                            "diving, underwater",
    "Carrying someone (princess)":       "carrying, princess_carry",
    "Piggyback":                         "carrying, piggyback",
    "Holding hands with partner":        "holding_hands",
    NONE:                         "",
}

ACTIONS = {
    "Looking at viewer":     "looking_at_viewer",
    "Looking away":          "looking_away",
    "Looking back":          "looking_back",
    "Looking up":            "looking_up",
    "Looking down":          "looking_down",
    "Looking to the side":   "looking_to_the_side",
    "Closed eyes":           "closed_eyes",
    "Eyes closed, smiling":  "closed_eyes, smile",
    "Holding something":     "holding",
    "Holding cup / drink":   "holding_cup",
    "Holding phone":         "holding_phone",
    "Holding book":          "holding_book",
    "Reading book":          "reading, book",
    "Eating":                "eating",
    "Drinking":              "drinking",
    "Sleeping":              "sleeping, closed_eyes",
    "Waving":                "waving",
    "Pointing at viewer":    "pointing_at_viewer",
    "Peace sign":            "v",
    "Double peace sign":     "double_v",
    "Hand on cheek":         "hand_on_own_cheek",
    "Hand on chin":          "hand_on_own_chin",
    "Hand in hair":          "hand_in_own_hair",
    "Arms up / stretching":  "arms_up, stretching",
    "Shy / blushing":        "blush, embarrassed",
    "Crying":                "tears, crying",
    "Laughing":              "laughing, open_mouth",
    "Yawning":               "yawn, sleepy",
    "Tilting head":          "head_tilt",
    "Adjusting glasses":     "adjusting_glasses",
    "Hair flip":             "hair_flip",
    # --- [NSFW] actions ---
    "[NSFW] Lifting skirt":          "skirt_lift, panties_visible",
    "[NSFW] Pulling clothes aside":  "clothes_pull, exposing",
    "[NSFW] Covering breasts":       "covering_breasts, embarrassed",
    "[NSFW] Covering crotch":        "covering_crotch",
    "[NSFW] Squeezing own breast":   "grabbing_own_breast",
    "[NSFW] Tongue out (sexy)":      "tongue_out, seductive_smile",
    "[NSFW] Biting lip":             "biting_lip, seductive",
    "[NSFW] Undressing":             "undressing, unbuttoning",
    "[NSFW] Self-cupping panties":   "hand_in_panties",
    "[NSFW] Pulling panties down":   "panty_pull, downblouse",
    # --- More safe actions ---
    "Brushing hair":                  "brushing_hair, hairbrush",
    "Applying makeup":                "applying_makeup",
    "Tying hair":                     "tying_hair, ponytail",
    "Putting on glasses":             "adjusting_glasses, glasses",
    "Holding flower":                 "holding_flower, flower",
    "Holding sword":                  "holding_sword, sword",
    "Holding gun":                    "holding_gun, weapon",
    "Holding umbrella":               "holding_umbrella, umbrella",
    "Holding bag":                    "holding_bag, bag",
    "Cooking":                        "cooking, frying_pan",
    "Studying":                       "studying, book, pen",
    "Writing":                        "writing, pen, notebook",
    "Drawing / painting":             "drawing, painting, brush",
    "Singing":                        "singing, microphone, open_mouth",
    "Playing instrument":             "playing_instrument, music",
    "Whispering":                     "whispering, hand_to_mouth",
    "Throwing":                       "throwing, arm_back",
    "Catching":                       "catching, arms_out",
    "Cheering":                       "cheering, arms_up, open_mouth",
    "Saluting":                       "salute, formal",
    "Bowing":                         "bowing, formal",
    "Curtsy":                         "curtsy, holding_dress",
    "Pinching cheek":                 "pinching, cheek_pinching",
    "Petting cat / dog":              "petting, cat" if 1 else "",
    NONE:                    "",
}

# ===========================================================================
# 3. Framing
# ===========================================================================

SHOT_TYPES = {
    "Portrait (head/shoulders)":     "portrait",
    "Close-up (face)":                "close-up",
    "Extreme close-up":               "extreme_close-up",
    "Upper body":                     "upper_body",
    "Cowboy shot (knees up)":         "cowboy_shot",
    "Full body":                      "full_body",
    "Wide shot (small subject)":      "wide_shot",
    "Scenery (no/tiny subject)":      "scenery",
    "POV (first person)":             "pov",
    "Selfie":                         "selfie",
    "Side profile":                   "profile, from_side",
    "Three-quarter view":             "three-quarter_view",
    "Over the shoulder":              "over_the_shoulder",
    "Establishing wide":              "scenery, very_wide_shot",
    "Macro / detail":                 "extreme_close-up, macro",
    "Action / dynamic":               "dynamic_pose, motion_lines",
    NONE:                             "",
}

CAMERA_ANGLES = {
    "Eye level (default)":  "",
    "From above":           "from_above",
    "From below":           "from_below",
    "From side":            "from_side",
    "From behind":          "from_behind",
    "Dutch angle":          "dutch_angle",
    "Bird's eye view":      "bird's-eye_view",
    "Worm's eye view":      "worm's-eye_view",
}

# ===========================================================================
# 4. Outfit
# ===========================================================================

OUTFIT_PRESETS = {
    "Casual everyday":            "casual, t-shirt, jeans",
    "Casual streetwear":          "streetwear, hoodie, jeans, sneakers",
    "School uniform (sailor)":    "school_uniform, sailor_collar, pleated_skirt",
    "School uniform (blazer)":    "school_uniform, blazer, necktie, pleated_skirt",
    "Office / business":          "business_suit, pencil_skirt, blouse",
    "Office (male)":              "business_suit, necktie, dress_shirt",
    "Formal dress":               "evening_gown, long_dress",
    "Wedding":                    "wedding_dress, veil",
    "Kimono":                     "kimono",
    "Yukata (summer festival)":   "yukata, obi",
    "Hanbok":                     "hanbok",
    "Cheongsam / qipao":          "china_dress",
    "Maid":                       "maid, maid_apron, maid_headdress",
    "Nurse":                      "nurse, nurse_cap",
    "Waitress":                   "waitress, apron",
    "Police officer":             "police_uniform, peaked_cap",
    "Soldier / military":         "military_uniform, beret",
    "Fantasy armor":              "armor, plate_armor, gauntlets",
    "Fantasy mage robes":         "wizard_robe, hood_up, witch_hat",
    "Cyberpunk":                  "cyberpunk, leather_jacket, neon_trim",
    "Sci-fi bodysuit":            "bodysuit, sci-fi, futuristic",
    "Magical girl":               "magical_girl, frills, hair_ribbon, gloves",
    "Idol stage costume":         "idol, frilled_dress, hair_bow, gloves",
    "Goth / lolita":              "gothic_lolita, frills, black_dress",
    "Sportswear (gym)":           "sportswear, sports_bra, shorts",
    "Sportswear (track)":         "track_suit, jacket",
    "Swimsuit (bikini)":          "bikini",
    "Swimsuit (one-piece)":       "one-piece_swimsuit",
    "Pajamas":                    "pajamas, long_sleeves",
    "Bathrobe / towel":           "bathrobe",
    "Winter coat":                "winter_clothes, coat, scarf",
    "Summer dress":               "summer_dress, sundress",
    "Hoodie + shorts":            "hoodie, shorts",
    "Naked (no clothes)":         "nude",
    # --- Additional safe outfits ---
    "Oversized shirt":            "oversized_shirt, off-shoulder",
    "Tank top + shorts":          "tank_top, shorts",
    "Knitted sweater":            "sweater, long_sleeves",
    "Hoodie + thighhighs":        "hoodie, thighhighs",
    "Dress shirt only":           "dress_shirt, no_pants",
    "Bodysuit (sport)":           "bodysuit, sportswear",
    "Tracksuit":                  "tracksuit, jacket",
    "Sari":                       "sari, indian_clothes",
    "Bridal lingerie":            "bridal_lingerie, lace, veil",
    # --- [NSFW] options ---
    "[NSFW] Lingerie (basic)":           "lingerie, bra, panties",
    "[NSFW] Lingerie (lacy + garter)":   "lingerie, lace, garter_belt, thighhighs",
    "[NSFW] Playboy bunny suit":         "playboy_bunny, bunny_ears, fishnet_pantyhose",
    "[NSFW] Micro bikini":               "micro_bikini, very_revealing",
    "[NSFW] Sling bikini":               "sling_bikini",
    "[NSFW] Naked apron":                "naked_apron, kitchen_apron",
    "[NSFW] Wet shirt (see-through)":    "wet_clothes, see-through, transparent",
    "[NSFW] Open shirt":                 "open_shirt, cleavage, exposed_breasts",
    "[NSFW] Crop top + miniskirt":       "crop_top, miniskirt, midriff",
    "[NSFW] Stockings + garter only":    "stockings, garter_belt, thighhighs, nude",
    "[NSFW] Topless":                    "topless, bare_breasts",
    "[NSFW] Bottomless":                 "bottomless, no_panties",
    "[NSFW] Nude + jewelry only":        "completely_nude, jewelry, necklace",
    "[NSFW] Revealing maid":             "maid, revealing_clothes, garter_belt, cleavage",
    "[NSFW] Gym uniform / bloomers":     "gym_uniform, bloomers",
    "[NSFW] Schoolgirl (undressing)":    "school_uniform, undressing, unbuttoned_shirt",
    "[NSFW] Torn clothes":               "torn_clothes, torn_shirt, exposed_skin",
    "[NSFW] Latex catsuit":              "latex, bodysuit, shiny_clothes",
    "[NSFW] Bondage outfit":             "bondage_outfit, harness, leather",
    "[NSFW] Cum on clothes":             "cum_on_clothes, dirty_clothes",
    # --- More fashion / themed outfits ---
    "Steampunk":                  "steampunk, brown_clothing, goggles",
    "Punk (leather + chains)":    "punk, leather_jacket, spiked_collar",
    "Y2K fashion":                "y2k, low_rise_pants, crop_top",
    "Cottagecore":                "cottagecore, floral_dress, sun_hat",
    "Dark academia":              "dark_academia, blazer, plaid_skirt",
    "Mori girl":                  "mori_girl, layered_clothes, earth_tones",
    "Lolita (sweet)":             "sweet_lolita, frills, bonnet, pink_dress",
    "Lolita (classic)":           "classic_lolita, lace, modest_dress",
    "Athleisure":                 "athleisure, leggings, sports_bra",
    "Sundress + sun hat":         "sundress, sun_hat, summer_outfit",
    "Trench coat":                "trench_coat, belt, long_coat",
    "Leather jacket + jeans":     "leather_jacket, jeans, casual_cool",
    "Cowgirl":                    "cowboy_hat, denim, cowgirl",
    "Pirate":                     "pirate, tricorne, eyepatch, sword",
    "Ninja":                      "ninja, mask, dark_clothes, weapon",
    "Samurai":                    "samurai_armor, katana, hakama",
    "Greek goddess":              "toga, gold_jewelry, sandals",
    "Egyptian":                   "egyptian_clothes, gold_jewelry, eye_makeup",
    "Holiday: Santa":             "santa_costume, santa_hat, red_clothes",
    "Holiday: bunny":             "bunny_ears, bunny_tail, easter",
    "Holiday: witch":             "witch_hat, witch_robe, halloween",
    "Holiday: angel":             "angel_wings, white_dress, halo",
    "Holiday: devil":             "demon_horns, demon_tail, red_clothes",
    "Cosplay (general)":          "cosplay, costume",
    "Hospital gown":              "hospital_gown, medical",
    "Apron only (cooking)":       "apron, casual",
    "Oversized hoodie only":      "hoodie, no_pants, oversized_clothes",
    NONE:                         "",
}

# ===========================================================================
# 5. Scene / Location
# ===========================================================================

LOCATIONS = {
    "Simple white background":   "white_background, simple_background",
    "Simple black background":   "black_background, simple_background",
    "Gradient background":       "gradient_background, simple_background",
    "Bedroom":                   "bedroom, indoors",
    "Living room":               "living_room, indoors, sofa",
    "Kitchen":                   "kitchen, indoors",
    "Bathroom":                  "bathroom, indoors, tile_wall",
    "Classroom":                 "classroom, indoors, school_desk",
    "School hallway":            "hallway, indoors, lockers",
    "Library":                   "library, indoors, bookshelf",
    "Cafe / coffee shop":        "cafe, indoors, table",
    "Restaurant":                "restaurant, indoors, table",
    "Office":                    "office, indoors, desk, computer",
    "Shop / store":              "shop, indoors, shelves",
    "Bar / nightclub":           "bar_(place), indoors, neon_lights",
    "Train interior":            "train_interior, indoors",
    "Park (outdoor)":            "park, outdoors, tree, grass",
    "Forest":                    "forest, outdoors, tree",
    "Beach":                     "beach, outdoors, ocean, sand",
    "Mountain landscape":        "mountain, outdoors, sky",
    "City street":               "street, outdoors, building, city",
    "Rooftop":                   "rooftop, outdoors, city",
    "School courtyard":          "courtyard, outdoors, school",
    "Garden":                    "garden, outdoors, flower",
    "Snowy outdoors":            "snow, outdoors, winter",
    "Rainy street":              "rain, outdoors, wet, reflection",
    "Cherry blossom":            "outdoors, cherry_blossoms, petals",
    "Festival night":            "festival, night, paper_lantern, outdoors",
    "Stage / concert":           "stage, indoors, spotlight, audience",
    "Castle interior":           "castle, indoors, stone_wall",
    "Dungeon":                   "dungeon, indoors, dark, torch",
    "Magical forest":            "fantasy, forest, glowing, mystical",
    "Cyberpunk city":            "cyberpunk, city, neon, night",
    "Space station":             "spaceship_interior, sci-fi",
    "Starry sky":                "sky, night_sky, stars, outdoors",
    "Pool / poolside":           "pool, poolside, water",
    "Underwater":                "underwater, bubbles",
    "Rooftop garden":            "rooftop, garden, outdoors, plants",
    "Train platform":            "train_station, platform, outdoors",
    "Locker room":               "locker_room, indoors",
    # --- Intimate locations ---
    "Bedroom (bed close-up)":    "bedroom, on_bed, sheets, pillow, indoors",
    "Hot spring / onsen":        "onsen, hot_spring, steam, water",
    "Shower":                    "shower, bathroom, water, wet",
    "Bathtub / bath":            "bathtub, bath, indoors, water",
    "[NSFW] Love hotel":         "love_hotel, indoors, neon",
    "[NSFW] Massage parlor":     "massage_parlor, indoors, oil",
    # --- More locations ---
    "Bookstore":                 "bookstore, indoors, bookshelf",
    "Toy store":                 "toy_store, indoors",
    "Convenience store":         "convenience_store, indoors",
    "Movie theater":             "movie_theater, indoors, dark",
    "Arcade":                    "arcade, indoors, neon_lights",
    "Bowling alley":             "bowling_alley, indoors",
    "Karaoke room":              "karaoke, indoors, microphone, neon",
    "Bus stop":                  "bus_stop, outdoors",
    "Subway / train interior":   "train_interior, indoors, commute",
    "Airport":                   "airport, indoors",
    "Hotel room":                "hotel_room, indoors, bed",
    "Hotel lobby":               "hotel_lobby, indoors",
    "Bridge over water":         "bridge, outdoors, water",
    "Pier / dock":               "pier, outdoors, ocean",
    "Waterfall":                 "waterfall, outdoors, nature",
    "Cliff edge":                "cliff, outdoors, sky",
    "Desert":                    "desert, outdoors, sand, sky",
    "Tundra / snowy plain":      "tundra, snow, outdoors",
    "Jungle":                    "jungle, outdoors, green, dense_foliage",
    "Tropical island":           "island, tropical, palm_tree, outdoors",
    "Cave":                      "cave, indoors, dark, rocks",
    "Aquarium":                  "aquarium, indoors, fish, blue_lighting",
    "Theme park / carnival":     "amusement_park, outdoors, ferris_wheel",
    "Ice skating rink":          "ice_rink, indoors, ice",
    "Greenhouse":                "greenhouse, indoors, plants, glass_ceiling",
    "Vineyard":                  "vineyard, outdoors, grape_vine",
    "Farm / barn":               "farm, outdoors, hay, barn",
    "Train platform":            "train_station, platform, outdoors",
    "Skyscraper office":         "office, indoors, large_windows, cityscape",
    "Penthouse":                 "penthouse, indoors, modern, view",
    "Underground / sewer":       "sewer, indoors, dark, pipes",
    "Spaceship cockpit":         "spaceship_interior, cockpit, sci-fi, consoles",
    "Throne room":               "throne_room, indoors, royal",
    "Library, antique":          "library, indoors, antique, bookshelf",
    "Train, scenic route":       "train_interior, indoors, window_view",
    NONE:                        "",
}

# ===========================================================================
# 6. Time of day + Lighting
# ===========================================================================

TIME_OF_DAY = {
    "Morning":          "morning, sunlight",
    "Noon / daytime":   "day, daylight",
    "Afternoon":        "afternoon, sunlight",
    "Golden hour":      "golden_hour, sunlight, warm_lighting",
    "Sunset":           "sunset, orange_sky",
    "Twilight":         "twilight, evening",
    "Night":            "night, dark",
    "Late night":       "night, moonlight",
    "Dawn":             "dawn, soft_lighting",
    NONE:               "",
}

LIGHTING = {
    "Soft natural":           "soft_lighting, natural_lighting",
    "Bright / sunny":         "bright, sunny",
    "Cinematic":              "cinematic_lighting, dramatic_lighting",
    "Backlit":                "backlighting, rim_lighting",
    "Rim light":              "rim_lighting",
    "Neon glow":              "neon_lights, glowing",
    "Candle / firelight":     "firelight, warm_lighting",
    "Studio lighting":        "studio_lighting",
    "Volumetric / god rays":  "volumetric_lighting, light_rays",
    "Moody / low light":      "low_light, moody, shadows",
    "Spotlight (single)":     "spotlight, single_light, dark_background",
    "Side lighting":          "side_lighting, half_shadow",
    "Top lighting (dramatic)":"top_lighting, dramatic_shadows",
    "Underwater glow":        "underwater_lighting, caustics, blue_tint",
    "Christmas / fairy lights":"fairy_lights, warm_lighting, bokeh",
    "Lantern light":          "lantern, warm_lighting, japanese",
    "Fluorescent (cold)":     "fluorescent_lighting, cold, sterile",
    "Sunset glow":            "warm_lighting, sunset_glow, orange",
    "Aurora borealis":        "aurora, northern_lights, magical",
    "Lightning strike":       "lightning, thunderstorm, dramatic",
    NONE:                     "",
}

# ===========================================================================
# 7. Mood / Expression
# ===========================================================================

EXPRESSIONS = {
    "Smiling":              "smile",
    "Big grin":             "grin, open_mouth",
    "Soft smile":           "smile, soft_expression",
    "Neutral":              "expressionless",
    "Serious":              "serious",
    "Sad":                  "sad, frown",
    "Crying":               "tears, crying",
    "Surprised":            "surprised, open_mouth",
    "Angry":                "angry",
    "Annoyed":              "annoyed",
    "Smug":                 "smug, smirk",
    "Embarrassed / blush":  "blush, embarrassed",
    "Shy":                  "shy, blush",
    "Sleepy":               "sleepy, half-closed_eyes",
    "Excited":              "excited, open_mouth",
    "Bored":                "bored",
    "Confident":            "confident",
    "Determined":           "determined",
    "Seductive":            "seductive_smile",
    "Yandere":              "yandere, crazed_eyes",
    "Smirk":                "smirk",
    "Pout":                 "pout, pouty_lips",
    "Wink":                 "wink, one_eye_closed",
    "Smiling, eyes closed": "smile, closed_eyes",
    # --- [NSFW] expressions ---
    "[NSFW] Ahegao":                "ahegao, rolling_eyes, tongue_out",
    "[NSFW] Erotic blush":          "blush, lustful_eyes, parted_lips",
    "[NSFW] Moaning":               "open_mouth, moaning, blush",
    "[NSFW] Half-lidded (sultry)":  "half-closed_eyes, sultry, bedroom_eyes",
    "[NSFW] Crying (pleasure)":     "tears, blush, open_mouth, pleasure",
    "[NSFW] Drooling":              "drooling, blush, open_mouth",
    "[NSFW] Naughty smile":         "evil_smile, smirk, lustful",
    "[NSFW] Orgasm face":           "orgasm_face, ahegao, rolling_eyes",
    # --- More expressions ---
    "Sticking tongue out (playful)":"tongue_out, playful",
    "Heart eyes":                    "heart-shaped_pupils, hearts, love",
    "Star eyes":                     "+_+, star-shaped_pupils, excited",
    "Eyes wide with awe":            "starry_eyes, wide_eyed, awe",
    "Hopeful":                       "hopeful, sparkling_eyes",
    "Worried":                       "worried, furrowed_brow",
    "Suspicious":                    "suspicious, narrowed_eyes",
    "Grumpy":                        "grumpy, pout, frown",
    "Content / peaceful":            "content_smile, peaceful",
    "Flirty":                        "flirty, half-closed_eyes",
    "Concentrating":                 "concentrated, focused, serious",
    "Daydreaming":                   "dreamy, looking_up, soft_expression",
    "Mysterious":                    "mysterious, half_smile",
    "Cocky / smug grin":             "smug, grin, raised_eyebrow",
    NONE:                   "",
}

# ===========================================================================
# 8. Art Style — model-aware
# ===========================================================================

ART_STYLES_ILLUSTRIOUS = {
    "Detailed anime (default)":    "anime_style, detailed, sharp_focus",
    "Soft pastel anime":           "pastel_colors, soft_lighting, soft_anime_style",
    "Vibrant anime":               "vibrant_colors, saturated, anime_style",
    "Cinematic / film":            "cinematic, depth_of_field, film_grain, dramatic_lighting",
    "Watercolor":                  "watercolor_(medium), traditional_media, soft_brushstrokes",
    "Oil painting":                "oil_painting_(medium), painterly, traditional_media",
    "Pencil sketch":               "sketch, pencil_drawing, monochrome, traditional_media",
    "Lineart only":                "lineart, monochrome, simple_shading",
    "Cel-shaded":                  "cel_shading, flat_colors, bold_outlines",
    "Manga (B&W)":                 "monochrome, manga, screentones, comic",
    "Painterly fantasy":           "painterly, fantasy_art, detailed_background",
    "3D / CG render":              "3d, cg, render, glossy",
    "Realistic / semi-real":       "semi-realistic, realistic, detailed_skin",
    "Chibi / SD":                  "chibi, sd_character, cute",
    "Gothic / dark":               "dark, gothic, moody, dramatic_shadows",
    "Retro 90s anime":             "1990s_(style), retro_anime, vhs_aesthetic",
    "Vaporwave":                   "vaporwave, pink_and_purple, retro_aesthetic",
    "Storybook / children's book": "storybook_illustration, soft_lines, warm_colors",
    "Studio Ghibli-esque":         "ghibli_inspired, soft_painted_background, warm_colors",
    "Konachan / wallpaper":        "wallpaper, scenic, detailed_background, official_art",
    "Western comic book":          "comic_book_style, bold_outlines, halftone",
    "Disney-esque":                "disney_style, smooth, expressive_face",
    "Pixar-esque":                 "3d, pixar_style, smooth, cute",
    "Manga ink wash":              "ink_wash, manga, monochrome",
    "Acrylic painting":            "acrylic_painting, textured, traditional",
    "Pop art":                     "pop_art, bold_colors, halftone_dots",
    "Art nouveau":                 "art_nouveau, ornate, decorative_frame",
    "Art deco":                    "art_deco, geometric, gold_accents",
    "Impressionist":               "impressionist_style, painterly, soft_focus",
    "Ukiyo-e (Japanese woodblock)":"ukiyo-e, japanese_traditional, flat_colors",
    "Surrealist":                  "surreal, dreamlike, abstract",
    "Pixel art":                   "pixel_art, retro_game",
    NONE:                          "",
}

ART_STYLES_NOOBAI = dict(ART_STYLES_ILLUSTRIOUS)

# ===========================================================================
# Artist styles (model-aware, curated)
# ===========================================================================
# Each entry: label -> (artist_tag, short_description, danbooru_url)
# Labels are prefixed with a [Category] tag so users can scan by mood.
# Selecting an artist emits either `artist_tag` (full strength) or
# `(artist_tag:strength)` if the user dials the strength slider away from 1.0.

ARTIST_STYLES = {
    "(none - use only my visual style preset)": (
        "", "No artist tag added. Output uses only the Visual style preset.", ""
    ),

    # --- Modern / clean anime ---
    "[Modern] kantoku - polished light-novel illustration": (
        "kantoku",
        "Clean modern anime, polished shading, light-novel illustrator aesthetic. "
        "Works great for character portraits and slice-of-life scenes.",
        "https://danbooru.donmai.us/posts?tags=kantoku",
    ),
    "[Modern] hiten_(hitenkei) - clean anime, soft palette": (
        "hiten_(hitenkei)",
        "Clean lines, soft palette, modern anime style. Great for cute "
        "everyday scenes.",
        "https://danbooru.donmai.us/posts?tags=hiten_(hitenkei)",
    ),
    "[Modern] ningen_mame - polished character art": (
        "ningen_mame",
        "Detailed character art with crisp shading and clean rendering. "
        "Strong faces, balanced compositions.",
        "https://danbooru.donmai.us/posts?tags=ningen_mame",
    ),
    "[Modern] ask_(askzy) - clean modern anime": (
        "ask_(askzy)",
        "Smooth modern anime, polished skin, soft lighting. Reliable allrounder.",
        "https://danbooru.donmai.us/posts?tags=ask_(askzy)",
    ),
    "[Modern] agawa_ryou - clean polished anime": (
        "agawa_ryou",
        "Clean anime aesthetic with polished lighting and balanced colors.",
        "https://danbooru.donmai.us/posts?tags=agawa_ryou",
    ),

    # --- Soft / pastel / dreamy ---
    "[Soft] ciloranko - soft pastel anime, dreamy": (
        "ciloranko",
        "Dreamy pastel colors, soft lighting, gentle expressions. "
        "Perfect for comfy / wholesome / romantic scenes.",
        "https://danbooru.donmai.us/posts?tags=ciloranko",
    ),
    "[Soft] mochizuki_kei - soft pastel": (
        "mochizuki_kei",
        "Soft pastel palette, dreamy atmosphere, gentle character faces.",
        "https://danbooru.donmai.us/posts?tags=mochizuki_kei",
    ),
    "[Soft] shal.e - cute soft anime": (
        "shal.e",
        "Cute anime aesthetic with soft colors and rounded shapes.",
        "https://danbooru.donmai.us/posts?tags=shal.e",
    ),
    "[Soft] as109 - smooth painterly soft": (
        "as109",
        "Smooth painterly anime with soft brushwork, dreamy mood.",
        "https://danbooru.donmai.us/posts?tags=as109",
    ),
    "[Soft] nardack - soft polished portraits": (
        "nardack",
        "Soft polished anime portraits with delicate skin and warm tones.",
        "https://danbooru.donmai.us/posts?tags=nardack",
    ),

    # --- Cinematic / dark / dramatic ---
    "[Cinematic] wlop - painterly dark fantasy": (
        "wlop",
        "Painterly, dramatic lighting, somber atmospheres. Highly detailed faces, "
        "moody dark backgrounds, signature ethereal feel.",
        "https://danbooru.donmai.us/posts?tags=wlop",
    ),
    "[Cinematic] ke-ta - cinematic anime, dramatic": (
        "ke-ta",
        "Cinematic anime with strong lighting contrast and dramatic compositions.",
        "https://danbooru.donmai.us/posts?tags=ke-ta",
    ),
    "[Cinematic] guweiz - detailed fantasy atmospheric": (
        "guweiz",
        "Detailed fantasy art with atmospheric lighting, deep colors. "
        "Strong for fantasy adventurers and dramatic portraits.",
        "https://danbooru.donmai.us/posts?tags=guweiz",
    ),
    "[Cinematic] nakkar - cinematic painterly": (
        "nakkar",
        "Cinematic painterly anime with strong mood lighting.",
        "https://danbooru.donmai.us/posts?tags=nakkar",
    ),
    "[Cinematic] redum4 - detailed painterly portraits": (
        "redum4",
        "Detailed painterly character art with rich textures and depth.",
        "https://danbooru.donmai.us/posts?tags=redum4",
    ),

    # --- Vibrant / pop / neon ---
    "[Vibrant] mika_pikazo - vibrant neon pop": (
        "mika_pikazo",
        "Vibrant saturated colors, neon-pop aesthetic, modern J-pop / idol feel.",
        "https://danbooru.donmai.us/posts?tags=mika_pikazo",
    ),
    "[Vibrant] redjuice - cyberpunk neon detailed": (
        "redjuice",
        "Cyberpunk-tinged detailed art with neon palette and modern UI feel.",
        "https://danbooru.donmai.us/posts?tags=redjuice",
    ),
    "[Vibrant] jaco - flat colors, clean cute": (
        "jaco",
        "Flat-color clean anime, cute compositions, easy on the eye.",
        "https://danbooru.donmai.us/posts?tags=jaco",
    ),

    # --- Watercolor / traditional ---
    "[Traditional] yoneyama_mai - watercolor traditional": (
        "yoneyama_mai",
        "Traditional watercolor-style anime with soft brushstrokes and "
        "muted natural palette.",
        "https://danbooru.donmai.us/posts?tags=yoneyama_mai",
    ),
    "[Traditional] aoi_ogata - soft watercolor": (
        "aoi_ogata",
        "Soft watercolor traditional style with delicate color washes.",
        "https://danbooru.donmai.us/posts?tags=aoi_ogata",
    ),

    # --- Retro / vintage ---
    "[Retro] lack - retro vintage anime": (
        "lack",
        "Retro vintage anime aesthetic, slightly grainy palette, 1990s flavor.",
        "https://danbooru.donmai.us/posts?tags=lack",
    ),
    "[Retro] bkub - simple cute retro": (
        "bkub_(bkub.exe)",
        "Simple cute retro style, bold outlines, flat colors. Comedic flavor.",
        "https://danbooru.donmai.us/posts?tags=bkub_(bkub.exe)",
    ),

    # --- Painterly / fantasy ---
    "[Painterly] huanxiang_heitu - Chinese painterly": (
        "huanxiang_heitu",
        "Chinese painterly anime with rich detail and atmospheric environments.",
        "https://danbooru.donmai.us/posts?tags=huanxiang_heitu",
    ),
    "[Painterly] kawacy - illustrative painterly": (
        "kawacy",
        "Illustrative painterly anime with dramatic compositions and warm colors.",
        "https://danbooru.donmai.us/posts?tags=kawacy",
    ),
}


def _load_full_artist_list():
    """Load the 33k Illustrious-NoobAI compatible artist list bundled with
    this extension. Names come space-separated with escaped parens; we
    normalize to booru form (spaces -> underscores, unescape parens) for
    emission, while keeping the human-readable name for the dropdown label.
    Returns a list of (display_label, booru_tag) tuples and a {label: tag}
    dict for fast lookup.
    """
    import os
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(here, "artists.txt")
    items = []
    seen = set()
    try:
        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                name = raw.strip()
                if not name or name in seen:
                    continue
                seen.add(name)
                # Booru tag: spaces -> underscores, unescape backslash-parens
                tag = name.replace("\\(", "(").replace("\\)", ")").replace(" ", "_")
                items.append((name, tag))
    except FileNotFoundError:
        items = []
    return items


# Loaded once at import.
FULL_ARTIST_ITEMS = _load_full_artist_list()
FULL_ARTIST_LABEL_TO_TAG = {label: tag for label, tag in FULL_ARTIST_ITEMS}
FULL_ARTIST_LABELS = ["(none)"] + [lbl for lbl, _ in FULL_ARTIST_ITEMS]
logger.info("{} loaded {} full-list artists".format(TAG, len(FULL_ARTIST_ITEMS)))


def _danbooru_preview_url(booru_tag, rating="g,s"):
    """Server-side fetch of one preview image URL from Danbooru. Returns the
    image URL string, or None on any failure. Anonymous endpoint, no auth."""
    if not booru_tag:
        return None
    try:
        import urllib.request, urllib.parse, json, ssl
        q = urllib.parse.quote(booru_tag)
        url = "https://danbooru.donmai.us/posts.json?tags={}&limit=1&random=true".format(q)
        req = urllib.request.Request(
            url, headers={"User-Agent": "sd-forge-prompt-enhancer/1.0"}
        )
        ctx = ssl.create_default_context()
        with urllib.request.urlopen(req, timeout=6, context=ctx) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        if not data:
            return None
        post = data[0] if isinstance(data, list) else data
        # Prefer the smaller preview, fall back to sample/file
        return (post.get("preview_file_url")
                or post.get("large_file_url")
                or post.get("file_url"))
    except Exception as e:
        logger.warning("{} danbooru preview fetch failed: {}".format(TAG, e))
        return None


def _artist_emit(label: str, strength: float = 1.0) -> str:
    """Return the artist tag formatted for the prompt.
    Strength of 1.0 emits the bare tag; other strengths wrap as (tag:strength)."""
    entry = ARTIST_STYLES.get(label)
    if not entry: return ""
    tag = entry[0]
    if not tag: return ""
    if abs(strength - 1.0) < 0.01:
        return tag
    return "({}:{:.2f})".format(tag, strength)


def _artist_preview_md(label: str) -> str:
    """Build the markdown preview shown under the dropdown."""
    entry = ARTIST_STYLES.get(label)
    if not entry:
        return "*Pick an artist to see the description.*"
    tag, desc, url = entry
    if not tag:
        return "*{}*".format(desc)
    md = "**Tag**: `{}`  \n**Style**: {}".format(tag, desc)
    if url:
        md += "  \n[View sample images on Danbooru]({})".format(url)
    return md


# ===========================================================================
# 9. Body / Physique
# ===========================================================================
# Mix of safe (slim, fit, athletic, hourglass...) and [NSFW] bust / hip
# specifics. Picked separately so the user can mix any body type with any
# outfit/pose/scene.

BODY_TYPES = {
    "Slim / petite":           "slim, petite, slender",
    "Average / fit":           "fit, average_build",
    "Athletic / toned":        "athletic, toned, fit",
    "Curvy":                   "curvy, voluptuous",
    "Hourglass":               "hourglass_figure, narrow_waist, wide_hips",
    "Muscular / abs":          "muscular, abs, toned_stomach",
    "Plump / thicc":           "thick_thighs, wide_hips, plump",
    "Tall":                    "tall",
    "Short":                   "short, petite",
    "[NSFW] Small bust":       "small_breasts",
    "[NSFW] Medium bust":      "medium_breasts",
    "[NSFW] Large bust":       "large_breasts",
    "[NSFW] Huge bust":        "huge_breasts",
    "[NSFW] Gigantic bust":    "gigantic_breasts",
    "[NSFW] Flat chest":       "flat_chest",
    "[NSFW] Big ass":          "huge_ass, wide_hips",
    "[NSFW] Thick thighs":     "thick_thighs",
    "Lean / wiry":             "lean, wiry",
    "Soft / round":            "soft_body, round",
    "Tomboyish (boyish)":      "tomboy, boyish_figure",
    "Statuesque (very tall)":  "tall, statuesque, long_legs",
    "Athletic with curves":    "fit, hourglass_figure, athletic",
    "Pixie / fairy small":     "small, petite, tiny_frame",
    "Pear shaped":             "wide_hips, narrow_shoulders",
    "Apple shaped":            "wide_shoulders, broad_chest",
    NONE:                      "",
}

# ===========================================================================
# 10. Intimacy / NSFW level
# ===========================================================================
# Tier of explicitness. Adds rating + content tags that nudge the model into
# the right zone. Stays at "(safe — none)" by default.

INTIMACY_LEVELS = {
    "(safe - none)":              "",
    "Romantic / sensual":          "sensual, intimate, romantic",
    "Suggestive (mildly sexy)":    "suggestive, sexy",
    "[NSFW] Softcore":             "nsfw, sensual, intimate, sexy, partially_nude",
    "[NSFW] Erotic":               "nsfw, erotic, seductive, naked, undressed",
    "[NSFW] Explicit (sex acts)":  "nsfw, explicit, sex, intercourse",
    "[NSFW] Hardcore":             "nsfw, explicit, hardcore, sex, intercourse",
    "[NSFW] Fetish - bondage":     "nsfw, bondage, restrained, rope",
    "[NSFW] Fetish - femdom":      "nsfw, femdom, dominant_female",
    "[NSFW] Aftermath":            "nsfw, after_sex, cum, sweat, disheveled",
}


# ===========================================================================
# Quality / Negative
# ===========================================================================

ILLUSTRIOUS_QUALITY_TIERS = {
    "Best (most opinionated)":      "masterpiece, best quality, amazing quality, very aesthetic, absurdres",
    "Standard (safe default)":      "masterpiece, best quality, very aesthetic, absurdres",
    "Minimal (just say it's good)": "best quality, very aesthetic",
    "None (subject only)":          "",
}

NOOBAI_QUALITY_TIERS = {
    "Best (most opinionated)":      "masterpiece, best quality, very aesthetic, absurdres, newest",
    "Standard (safe default)":      "masterpiece, best quality, very aesthetic",
    "Minimal (just say it's good)": "best quality",
    "None (subject only)":          "",
}

ILLUSTRIOUS_NEGATIVE_TIERS = {
    "Strong (thorough)": (
        "worst quality, low quality, normal quality, lowres, "
        "bad anatomy, bad hands, bad fingers, extra fingers, missing fingers, "
        "extra arms, extra legs, missing arms, missing legs, "
        "mutated hands, fused fingers, "
        "watermark, signature, artist name, text, error, "
        "jpeg artifacts, blurry, sketch, "
        "censored, mosaic censoring, bar censor"
    ),
    "Standard (safe default)": (
        "worst quality, low quality, normal quality, lowres, "
        "bad anatomy, bad hands, watermark, signature, "
        "jpeg artifacts, blurry, sketch"
    ),
    "Minimal (just block junk)": "worst quality, low quality, lowres, jpeg artifacts",
    "None (no negative)": "",
}

NOOBAI_NEGATIVE_TIERS = ILLUSTRIOUS_NEGATIVE_TIERS

YEAR_TAGS = ["(no year tag)", "newest", "recent", "mid", "old", "oldest"]
SOURCE_TAGS = [
    "(no source tag)", "source_anime", "source_cartoon",
    "source_pony", "source_furry", "source_3d", "source_real",
    "official_art",
]
RATING_TAGS = [
    "(no rating tag)", "rating:general", "rating:sensitive",
    "rating:questionable", "rating:explicit",
]

MODELS: Dict[str, Dict] = {
    "Illustrious-XL (any variant)": {
        "keywords": ["illustrious", "illustrius", "waiillustrious", "wai_illustrious"],
        "positive_tiers": ILLUSTRIOUS_QUALITY_TIERS,
        "negative_tiers": ILLUSTRIOUS_NEGATIVE_TIERS,
        "art_styles": ART_STYLES_ILLUSTRIOUS,
    },
    "NoobAI XL (v-pred / eps)": {
        "keywords": ["noobai", "noob_ai", "noob-ai"],
        "positive_tiers": NOOBAI_QUALITY_TIERS,
        "negative_tiers": NOOBAI_NEGATIVE_TIERS,
        "art_styles": ART_STYLES_NOOBAI,
    },
    "(detect from loaded checkpoint)": {
        "keywords": [],
        "positive_tiers": ILLUSTRIOUS_QUALITY_TIERS,
        "negative_tiers": ILLUSTRIOUS_NEGATIVE_TIERS,
        "art_styles": ART_STYLES_ILLUSTRIOUS,
    },
}

# ===========================================================================
# Scenario templates
# ===========================================================================

SCENARIOS: Dict[str, Dict] = {
    "(none — design from scratch)": {},

    "Character portrait (clean)": {
        "subject": "Solo - 1girl (young adult female)",
        "pose":    "Standing",
        "action":  "Looking at viewer",
        "shot":    "Portrait (head/shoulders)",
        "angle":   "Eye level (default)",
        "outfit":  "Casual everyday",
        "location":"Simple white background",
        "time":    NONE,
        "lighting":"Soft natural",
        "expr":    "Soft smile",
        "style":   "Detailed anime (default)",
    },
    "Full-body reference sheet": {
        "subject": "Solo - 1girl (young adult female)",
        "pose":    "Standing, arms crossed",
        "action":  "Looking at viewer",
        "shot":    "Full body",
        "angle":   "Eye level (default)",
        "outfit":  "Casual everyday",
        "location":"Simple white background",
        "time":    NONE,
        "lighting":"Studio lighting",
        "expr":    "Neutral",
        "style":   "Detailed anime (default)",
    },
    "School day (classroom)": {
        "subject": "Solo - 1girl (young adult female)",
        "pose":    "Sitting on chair",
        "action":  "Looking at viewer",
        "shot":    "Upper body",
        "angle":   "Eye level (default)",
        "outfit":  "School uniform (sailor)",
        "location":"Classroom",
        "time":    "Afternoon",
        "lighting":"Soft natural",
        "expr":    "Smiling",
        "style":   "Detailed anime (default)",
    },
    "Cafe outing (casual)": {
        "subject": "Solo - 1girl (young adult female)",
        "pose":    "Sitting on chair",
        "action":  "Holding cup / drink",
        "shot":    "Upper body",
        "angle":   "Eye level (default)",
        "outfit":  "Casual everyday",
        "location":"Cafe / coffee shop",
        "time":    "Afternoon",
        "lighting":"Soft natural",
        "expr":    "Soft smile",
        "style":   "Soft pastel anime",
    },
    "Beach vacation": {
        "subject": "Solo - 1girl (young adult female)",
        "pose":    "Standing",
        "action":  "Looking at viewer",
        "shot":    "Cowboy shot (knees up)",
        "angle":   "Eye level (default)",
        "outfit":  "Swimsuit (bikini)",
        "location":"Beach",
        "time":    "Noon / daytime",
        "lighting":"Bright / sunny",
        "expr":    "Big grin",
        "style":   "Vibrant anime",
    },
    "Combat / action scene": {
        "subject": "Solo - 1girl (young adult female)",
        "pose":    "Combat / fighting stance",
        "action":  "Looking at viewer",
        "shot":    "Full body",
        "angle":   "Dutch angle",
        "outfit":  "Fantasy armor",
        "location":"Mountain landscape",
        "time":    "Sunset",
        "lighting":"Cinematic",
        "expr":    "Determined",
        "style":   "Cinematic / film",
    },
    "Magical girl transformation": {
        "subject": "Solo - 1girl (young adult female)",
        "pose":    "Spell-casting",
        "action":  "Looking up",
        "shot":    "Full body",
        "angle":   "From below",
        "outfit":  "Magical girl",
        "location":"Starry sky",
        "time":    "Late night",
        "lighting":"Volumetric / god rays",
        "expr":    "Confident",
        "style":   "Vibrant anime",
    },
    "Idol performance": {
        "subject": "Solo - 1girl (young adult female)",
        "pose":    "Dancing",
        "action":  "Looking at viewer",
        "shot":    "Cowboy shot (knees up)",
        "angle":   "From below",
        "outfit":  "Idol stage costume",
        "location":"Stage / concert",
        "time":    NONE,
        "lighting":"Neon glow",
        "expr":    "Big grin",
        "style":   "Vibrant anime",
    },
    "Summer festival (yukata)": {
        "subject": "Solo - 1girl (young adult female)",
        "pose":    "Walking",
        "action":  "Looking back",
        "shot":    "Cowboy shot (knees up)",
        "angle":   "Eye level (default)",
        "outfit":  "Yukata (summer festival)",
        "location":"Festival night",
        "time":    "Night",
        "lighting":"Candle / firelight",
        "expr":    "Soft smile",
        "style":   "Soft pastel anime",
    },
    "Cozy bedroom (pajamas)": {
        "subject": "Solo - 1girl (young adult female)",
        "pose":    "Sitting on floor",
        "action":  "Holding cup / drink",
        "shot":    "Upper body",
        "angle":   "Eye level (default)",
        "outfit":  "Pajamas",
        "location":"Bedroom",
        "time":    "Late night",
        "lighting":"Candle / firelight",
        "expr":    "Sleepy",
        "style":   "Soft pastel anime",
    },
    "Office worker": {
        "subject": "Solo - 1girl (young adult female)",
        "pose":    "Standing",
        "action":  "Holding phone",
        "shot":    "Upper body",
        "angle":   "Eye level (default)",
        "outfit":  "Office / business",
        "location":"Office",
        "time":    "Morning",
        "lighting":"Bright / sunny",
        "expr":    "Confident",
        "style":   "Detailed anime (default)",
    },
    "Cyberpunk night": {
        "subject": "Solo - 1girl (young adult female)",
        "pose":    "Standing, hands in pockets",
        "action":  "Looking at viewer",
        "shot":    "Cowboy shot (knees up)",
        "angle":   "From below",
        "outfit":  "Cyberpunk",
        "location":"Cyberpunk city",
        "time":    "Late night",
        "lighting":"Neon glow",
        "expr":    "Smug",
        "style":   "Cinematic / film",
    },
    "Sci-fi space": {
        "subject": "Solo - 1girl (young adult female)",
        "pose":    "Standing",
        "action":  "Looking at viewer",
        "shot":    "Cowboy shot (knees up)",
        "angle":   "Eye level (default)",
        "outfit":  "Sci-fi bodysuit",
        "location":"Space station",
        "time":    NONE,
        "lighting":"Studio lighting",
        "expr":    "Confident",
        "style":   "3D / CG render",
    },
    "Fantasy adventurer": {
        "subject": "Solo - 1girl (young adult female)",
        "pose":    "Standing, hands on hips",
        "action":  "Looking at viewer",
        "shot":    "Full body",
        "angle":   "Eye level (default)",
        "outfit":  "Fantasy armor",
        "location":"Forest",
        "time":    "Golden hour",
        "lighting":"Volumetric / god rays",
        "expr":    "Determined",
        "style":   "Painterly fantasy",
    },
    "Sakura / cherry blossoms": {
        "subject": "Solo - 1girl (young adult female)",
        "pose":    "Standing",
        "action":  "Looking at viewer",
        "shot":    "Upper body",
        "angle":   "Eye level (default)",
        "outfit":  "School uniform (blazer)",
        "location":"Cherry blossom",
        "time":    "Afternoon",
        "lighting":"Soft natural",
        "expr":    "Soft smile",
        "style":   "Soft pastel anime",
    },
    "Rainy day (umbrella)": {
        "subject": "Solo - 1girl (young adult female)",
        "pose":    "Standing",
        "action":  "Holding something",
        "shot":    "Cowboy shot (knees up)",
        "angle":   "Eye level (default)",
        "outfit":  "Casual streetwear",
        "location":"Rainy street",
        "time":    "Twilight",
        "lighting":"Moody / low light",
        "expr":    "Soft smile",
        "style":   "Cinematic / film",
    },
    "Winter snow": {
        "subject": "Solo - 1girl (young adult female)",
        "pose":    "Standing",
        "action":  "Looking at viewer",
        "shot":    "Upper body",
        "angle":   "Eye level (default)",
        "outfit":  "Winter coat",
        "location":"Snowy outdoors",
        "time":    "Afternoon",
        "lighting":"Soft natural",
        "expr":    "Soft smile",
        "style":   "Soft pastel anime",
    },
    "Goth / dark portrait": {
        "subject": "Solo - 1girl (young adult female)",
        "pose":    "Standing",
        "action":  "Looking at viewer",
        "shot":    "Upper body",
        "angle":   "Eye level (default)",
        "outfit":  "Goth / lolita",
        "location":"Simple black background",
        "time":    NONE,
        "lighting":"Moody / low light",
        "expr":    "Serious",
        "style":   "Gothic / dark",
    },
    # Multi-character scenarios (test that 1girl doesn't show up)
    "Two girls hanging out": {
        "subject": "Pair - 2girls",
        "pose":    "Sitting on chair",
        "action":  "Looking at viewer",
        "shot":    "Upper body",
        "angle":   "Eye level (default)",
        "outfit":  "Casual everyday",
        "location":"Cafe / coffee shop",
        "time":    "Afternoon",
        "lighting":"Soft natural",
        "expr":    "Big grin",
        "style":   "Detailed anime (default)",
    },
    "Mixed pair (1girl + 1boy)": {
        "subject": "Pair - 1girl + 1boy",
        "pose":    "Standing",
        "action":  "Looking at viewer",
        "shot":    "Upper body",
        "angle":   "Eye level (default)",
        "outfit":  "Casual everyday",
        "location":"Park (outdoor)",
        "time":    "Afternoon",
        "lighting":"Soft natural",
        "expr":    "Soft smile",
        "style":   "Detailed anime (default)",
    },
    # --- [NSFW] scenario presets ---
    "[NSFW] Bedroom seduction": {
        "subject": "Solo - 1girl (young adult female)",
        "pose":    "[NSFW] Reclining on bed",
        "action":  "Looking at viewer",
        "shot":    "Cowboy shot (knees up)",
        "angle":   "Eye level (default)",
        "outfit":  "[NSFW] Lingerie (lacy + garter)",
        "location":"Bedroom (bed close-up)",
        "time":    "Late night",
        "lighting":"Candle / firelight",
        "expr":    "[NSFW] Half-lidded (sultry)",
        "style":   "Cinematic / film",
    },
    "[NSFW] Shower scene": {
        "subject": "Solo - 1girl (young adult female)",
        "pose":    "Standing",
        "action":  "Looking at viewer",
        "shot":    "Upper body",
        "angle":   "Eye level (default)",
        "outfit":  "Naked (no clothes)",
        "location":"Shower",
        "time":    NONE,
        "lighting":"Soft natural",
        "expr":    "[NSFW] Erotic blush",
        "style":   "Cinematic / film",
    },
    "[NSFW] Hot spring relax": {
        "subject": "Solo - 1girl (young adult female)",
        "pose":    "Sitting on knees (seiza)",
        "action":  "Closed eyes",
        "shot":    "Upper body",
        "angle":   "Eye level (default)",
        "outfit":  "Naked (no clothes)",
        "location":"Hot spring / onsen",
        "time":    "Twilight",
        "lighting":"Volumetric / god rays",
        "expr":    "Soft smile",
        "style":   "Painterly fantasy",
    },
    "[NSFW] Lingerie photoshoot": {
        "subject": "Solo - 1girl (young adult female)",
        "pose":    "Standing, hands on hips",
        "action":  "Looking at viewer",
        "shot":    "Full body",
        "angle":   "Eye level (default)",
        "outfit":  "[NSFW] Lingerie (lacy + garter)",
        "location":"Simple white background",
        "time":    NONE,
        "lighting":"Studio lighting",
        "expr":    "Smug",
        "style":   "Cinematic / film",
    },
    "[NSFW] Naked apron (kitchen)": {
        "subject": "Solo - 1girl (young adult female)",
        "pose":    "Standing",
        "action":  "Looking back",
        "shot":    "Cowboy shot (knees up)",
        "angle":   "From behind",
        "outfit":  "[NSFW] Naked apron",
        "location":"Kitchen",
        "time":    "Morning",
        "lighting":"Soft natural",
        "expr":    "[NSFW] Half-lidded (sultry)",
        "style":   "Detailed anime (default)",
    },
    "[NSFW] Bunny suit casino": {
        "subject": "Solo - 1girl (young adult female)",
        "pose":    "Standing, hands on hips",
        "action":  "Wink",
        "shot":    "Cowboy shot (knees up)",
        "angle":   "Eye level (default)",
        "outfit":  "[NSFW] Playboy bunny suit",
        "location":"Bar / nightclub",
        "time":    "Late night",
        "lighting":"Neon glow",
        "expr":    "Smug",
        "style":   "Vibrant anime",
    },
    "[NSFW] Beach micro bikini": {
        "subject": "Solo - 1girl (young adult female)",
        "pose":    "Standing, hands behind back",
        "action":  "Looking at viewer",
        "shot":    "Full body",
        "angle":   "Eye level (default)",
        "outfit":  "[NSFW] Micro bikini",
        "location":"Beach",
        "time":    "Noon / daytime",
        "lighting":"Bright / sunny",
        "expr":    "[NSFW] Half-lidded (sultry)",
        "style":   "Vibrant anime",
    },
    "Library study session": {
        "subject": "Solo - 1girl (young adult female)",
        "pose":    "Sitting at desk",
        "action":  "Studying",
        "shot":    "Upper body",
        "angle":   "Eye level (default)",
        "outfit":  "Dark academia",
        "location":"Library, antique",
        "time":    "Afternoon",
        "lighting":"Soft natural",
        "expr":    "Concentrating",
        "style":   "Cinematic / film",
    },
    "Park picnic": {
        "subject": "Solo - 1girl (young adult female)",
        "pose":    "Sitting on floor",
        "action":  "Holding cup / drink",
        "shot":    "Full body",
        "angle":   "Eye level (default)",
        "outfit":  "Sundress + sun hat",
        "location":"Park (outdoor)",
        "time":    "Noon / daytime",
        "lighting":"Bright / sunny",
        "expr":    "Soft smile",
        "style":   "Soft pastel anime",
    },
    "Movie night couch": {
        "subject": "Solo - 1girl (young adult female)",
        "pose":    "Sitting on floor",
        "action":  "Holding cup / drink",
        "shot":    "Upper body",
        "angle":   "Eye level (default)",
        "outfit":  "Pajamas",
        "location":"Living room",
        "time":    "Late night",
        "lighting":"Spotlight (single)",
        "expr":    "Content / peaceful",
        "style":   "Cinematic / film",
    },
    "Karaoke night": {
        "subject": "Solo - 1girl (young adult female)",
        "pose":    "Standing",
        "action":  "Singing",
        "shot":    "Upper body",
        "angle":   "Eye level (default)",
        "outfit":  "Casual streetwear",
        "location":"Karaoke room",
        "time":    "Night",
        "lighting":"Neon glow",
        "expr":    "Excited",
        "style":   "Vibrant anime",
    },
    "Sushi date": {
        "subject": "Pair - 1girl + 1boy",
        "pose":    "Sitting on chair",
        "action":  "Eating",
        "shot":    "Upper body",
        "angle":   "Eye level (default)",
        "outfit":  "Casual everyday",
        "location":"Restaurant",
        "time":    "Twilight",
        "lighting":"Lantern light",
        "expr":    "Soft smile",
        "style":   "Soft pastel anime",
    },
    "Rainy day cafe": {
        "subject": "Solo - 1girl (young adult female)",
        "pose":    "Sitting on chair",
        "action":  "Holding cup / drink",
        "shot":    "Upper body",
        "angle":   "Eye level (default)",
        "outfit":  "Trench coat",
        "location":"Cafe / coffee shop",
        "time":    "Afternoon",
        "lighting":"Moody / low light",
        "expr":    "Daydreaming",
        "style":   "Cinematic / film",
    },
    "Sunset rooftop": {
        "subject": "Solo - 1girl (young adult female)",
        "pose":    "Standing",
        "action":  "Looking away",
        "shot":    "Full body",
        "angle":   "From side",
        "outfit":  "Casual streetwear",
        "location":"Rooftop",
        "time":    "Sunset",
        "lighting":"Sunset glow",
        "expr":    "Daydreaming",
        "style":   "Cinematic / film",
    },
    "Camping fireside": {
        "subject": "Solo - 1girl (young adult female)",
        "pose":    "Sitting cross-legged",
        "action":  "Holding cup / drink",
        "shot":    "Upper body",
        "angle":   "Eye level (default)",
        "outfit":  "Casual streetwear",
        "location":"Forest",
        "time":    "Night",
        "lighting":"Candle / firelight",
        "expr":    "Content / peaceful",
        "style":   "Painterly fantasy",
    },
    "Concert front row": {
        "subject": "Solo - 1girl (young adult female)",
        "pose":    "Standing",
        "action":  "Cheering",
        "shot":    "Upper body",
        "angle":   "From below",
        "outfit":  "Casual streetwear",
        "location":"Stage / concert",
        "time":    NONE,
        "lighting":"Neon glow",
        "expr":    "Excited",
        "style":   "Cinematic / film",
    },
    "New Year fireworks": {
        "subject": "Solo - 1girl (young adult female)",
        "pose":    "Standing",
        "action":  "Looking up",
        "shot":    "Cowboy shot (knees up)",
        "angle":   "From below",
        "outfit":  "Yukata (summer festival)",
        "location":"Festival night",
        "time":    "Late night",
        "lighting":"Neon glow",
        "expr":    "Eyes wide with awe",
        "style":   "Vibrant anime",
    },
    "Group scene (multiple girls)": {
        "subject": "Group - multiple girls",
        "pose":    "Standing",
        "action":  "Looking at viewer",
        "shot":    "Full body",
        "angle":   "Eye level (default)",
        "outfit":  "School uniform (blazer)",
        "location":"School courtyard",
        "time":    "Afternoon",
        "lighting":"Soft natural",
        "expr":    "Smiling",
        "style":   "Detailed anime (default)",
    },
}

# ===========================================================================
# Detection
# ===========================================================================

def _get_loaded_checkpoint() -> str:
    try:
        if shared is not None:
            ckpt = getattr(shared.opts, "sd_model_checkpoint", "") or ""
            return str(ckpt)
    except Exception:
        pass
    return ""


def _detect_model_family(checkpoint_name: str) -> str:
    name_low = (checkpoint_name or "").lower()
    if not name_low:
        return "(detect from loaded checkpoint)"
    for family_label, family in MODELS.items():
        for kw in family["keywords"]:
            if kw in name_low:
                return family_label
    return "Illustrious-XL (any variant)"


def _resolve_family(family_label: str):
    if family_label == "(detect from loaded checkpoint)" or family_label not in MODELS:
        family_label = _detect_model_family(_get_loaded_checkpoint())
    if family_label not in MODELS or family_label == "(detect from loaded checkpoint)":
        family_label = "Illustrious-XL (any variant)"
    return family_label, MODELS[family_label]


def _count_from_subject(label: str) -> int:
    """How many per-character trait slots are relevant.
    Solo / (skip) = 1, Pair = 2, Trio = 3, Group = up to 4."""
    if label.startswith("Group"): return 4
    if label.startswith("Trio"):  return 3
    if label.startswith("Pair"):  return 2
    return 1


# Subjects that are all-female (no males implied). The female-only negative
# helper kicks in when subject is in this set and the user has the checkbox on.
ALL_FEMALE_SUBJECTS = {
    "Solo - 1girl (young adult female)",
    "Solo - mature woman",
    "Solo - female child",
    "Pair - 2girls",
    "Trio - 3girls",
    "Group - multiple girls",
}

# Subjects with two or more females (eligible for the yuri tag).
MULTI_FEMALE_SUBJECTS = {
    "Pair - 2girls",
    "Trio - 3girls",
    "Group - multiple girls",
}

# Tag string we add to the NEGATIVE prompt when 'Force female-only' is on.
# Comprehensive male-exclusion so Illustrious can't slip a boy in.
FEMALE_ONLY_NEGATIVE = (
    "1boy, 2boys, 3boys, multiple_boys, male, male_focus, penis, testicles, "
    "mustache, beard, facial_hair"
)

# Tag string we add to the POSITIVE prompt when 'Yuri' is on.
YURI_POSITIVE = "yuri"


def _is_all_female(subject_label):
    return subject_label in ALL_FEMALE_SUBJECTS


def _is_multi_female(subject_label):
    return subject_label in MULTI_FEMALE_SUBJECTS


def _split_names(s: str):
    """Comma-split that ignores commas inside parens.
    Needed so series names like 'kafka_(honkai:_star_rail)' survive intact."""
    parts, depth, buf = [], 0, []
    for c in s:
        if c == '(':
            depth += 1; buf.append(c)
        elif c == ')':
            depth = max(0, depth - 1); buf.append(c)
        elif c == ',' and depth == 0:
            parts.append(''.join(buf)); buf = []
        else:
            buf.append(c)
    if buf: parts.append(''.join(buf))
    return parts


def _parse_char_names(s: str):
    """Parse one or more character names into normalized booru-format strings.
    Returns a list (possibly empty). Spaces -> underscores. Keeps parens, colons,
    hyphens, underscores."""
    if not s: return []
    out = []
    for part in _split_names(s):
        n = part.strip().lower().replace(" ", "_")
        n = re.sub(r"[^\w\(\):_\-]", "", n)
        if n: out.append(n)
    return out


# ===========================================================================
# Assembly
# ===========================================================================

def _tag(d: Dict[str, str], label: str) -> str:
    if not label or label == NONE:
        return ""
    return d.get(label, "")


def _join(*parts: str) -> str:
    """Comma-join parts, splitting each on commas, deduping case-insensitively,
    preserving first-seen order."""
    seen = []
    seen_set = set()
    for p in parts:
        if not p:
            continue
        for t in [t.strip() for t in p.split(",")]:
            if t and t.lower() not in seen_set:
                seen.append(t)
                seen_set.add(t.lower())
    return ", ".join(seen)


def build_prompt(
    family_label,
    subject_label,
    character_name,
    custom_subject,
    pose_label,
    action_label,
    shot_label,
    angle_label,
    outfit_label,
    location_label,
    time_label,
    lighting_label,
    expression_label,
    style_label,
    quality_tier,
    negative_tier,
    year_tag,
    source_tag,
    rating_tag,
    extra_positive,
    extra_negative,
    char1_traits="",
    char2_traits="",
    char3_traits="",
    char4_traits="",
    use_break=False,
    body_label=NONE,
    intimacy_label="(safe - none)",
    artist_label="(none - use only my visual style preset)",
    artist_strength=1.0,
    custom_artist_tag="",
    force_female_only=True,
    yuri=False,
) -> Tuple[str, str, str]:
    family_label, fam = _resolve_family(family_label)

    subject_t   = _tag(SUBJECTS, subject_label)
    n_chars     = _count_from_subject(subject_label)
    # Yuri tag goes into the subject head when applicable (multi-female).
    if yuri and _is_multi_female(subject_label) and subject_t:
        subject_t = subject_t + ", " + YURI_POSITIVE
    names_list  = _parse_char_names(character_name or "")
    custom_subj = (custom_subject or "").strip().strip(",")

    per_char_all = [
        (char1_traits or "").strip().strip(","),
        (char2_traits or "").strip().strip(","),
        (char3_traits or "").strip().strip(","),
        (char4_traits or "").strip().strip(","),
    ]
    relevant_traits = [t for t in per_char_all[:max(n_chars, 1)] if t]

    pose_t   = _tag(POSES, pose_label)
    action_t = _tag(ACTIONS, action_label)
    shot_t   = _tag(SHOT_TYPES, shot_label)
    angle_t  = _tag(CAMERA_ANGLES, angle_label)
    outfit_t = _tag(OUTFIT_PRESETS, outfit_label)
    loc_t    = _tag(LOCATIONS, location_label)
    time_t   = _tag(TIME_OF_DAY, time_label)
    light_t  = _tag(LIGHTING, lighting_label)
    expr_t     = _tag(EXPRESSIONS, expression_label)
    body_t     = _tag(BODY_TYPES, body_label)
    # Custom pasted tag (from the Explorer) takes priority over the dropdown.
    _custom_t = (custom_artist_tag or "").strip().strip(",")
    if _custom_t:
        _s = float(artist_strength or 1.0)
        if abs(_s - 1.0) < 0.01:
            artist_t = _custom_t
        else:
            artist_t = "({}:{:.2f})".format(_custom_t, _s)
    else:
        artist_t = _artist_emit(artist_label, float(artist_strength or 1.0))
    style_t    = _tag(fam["art_styles"], style_label)
    intimacy_t = INTIMACY_LEVELS.get(intimacy_label, "") if intimacy_label else ""

    meta_parts = []
    if year_tag and year_tag != "(no year tag)": meta_parts.append(year_tag)
    if source_tag and source_tag != "(no source tag)": meta_parts.append(source_tag)
    if rating_tag and rating_tag != "(no rating tag)": meta_parts.append(rating_tag)
    meta_t = ", ".join(meta_parts)

    quality_t   = fam["positive_tiers"].get(quality_tier, "")
    extra_pos_t = (extra_positive or "").strip().strip(",")

    # The "rest" of the prompt (everything after the subject/character clusters)
    # is always deduped via _join.
    rest = _join(
        outfit_t,
        pose_t, action_t,
        shot_t, angle_t,
        loc_t, time_t, light_t,
        expr_t,
        body_t,
        artist_t,
        style_t,
        intimacy_t,
        meta_t, quality_t,
        extra_pos_t,
    )

    # Build per-character clusters: each cluster = name_i + traits_i (either may be
    # empty). Cluster i exists if EITHER the name OR the traits slot has content.
    char_clusters = []
    for i in range(n_chars):
        nm = names_list[i] if i < len(names_list) else ""
        tr = per_char_all[i] if i < len(per_char_all) else ""
        parts = []
        if nm: parts.append(nm)
        if tr: parts.append(tr)
        if parts: char_clusters.append(", ".join(parts))

    if n_chars >= 2 and char_clusters:
        # Multi-character path: each character gets their own cluster so the
        # model can associate that character's name + traits to one person.
        head_parts = [subject_t]
        if custom_subj: head_parts.append(custom_subj)
        head = ", ".join(p for p in head_parts if p)

        if use_break:
            # ' BREAK ' marker chunks the prompt for Forge's attention engine,
            # giving Illustrious / NoobAI cleaner trait separation per character.
            positive = " BREAK ".join([head] + char_clusters)
            if rest:
                positive = positive + ", " + rest
        else:
            # No-BREAK: just concat clusters with commas, no cross-cluster dedup
            all_parts = [head] + char_clusters + ([rest] if rest else [])
            positive  = ", ".join(p for p in all_parts if p)
    else:
        # Solo / no per-char content: collapse all names into the head, dedup
        # everything together.
        solo_name = ", ".join(names_list) if names_list else ""
        positive = _join(subject_t, solo_name, custom_subj, rest)

    neg_t       = fam["negative_tiers"].get(negative_tier, "")
    extra_neg_t = (extra_negative or "").strip().strip(",")
    # Female-only enforcement: if subject is all-female and the checkbox is on,
    # add male-exclusion tags to the negative so Illustrious can't slip a boy in.
    female_only_t = (FEMALE_ONLY_NEGATIVE
                     if (force_female_only and _is_all_female(subject_label))
                     else "")
    negative    = _join(neg_t, female_only_t, extra_neg_t)

    if not positive:
        return ("", "", "Fill in a Subject, Outfit, or Scenario above, then click Build.")
    msg = "Built for {} ({} character{}). Quality: {}. Negative: {}.".format(
        family_label, n_chars, "s" if n_chars != 1 else "",
        quality_tier, negative_tier)
    return positive, negative, msg


# ===========================================================================
# Gradio UI
# ===========================================================================

def _on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as ui:
        gr.Markdown(
            "## Prompt Enhancer — guided builder\n"
            "Pick from the dropdowns below. Each option maps to the booru tags "
            "Illustrious and NoobAI XL were trained on — you don't need to know "
            "what 'cowboy shot' or 'rim lighting' means as a tag. Start with a "
            "**Scenario template** below to fill in many sections at once, then tweak."
        )

        with gr.Row():
            with gr.Column(scale=2):
                detected_label = gr.Markdown(
                    "**Detected**: (pick a scenario or click Detect)")
                family = gr.Dropdown(
                    label="Model family",
                    choices=list(MODELS.keys()),
                    value="(detect from loaded checkpoint)",
                    interactive=True,
                )
            with gr.Column(scale=1):
                detect_btn = gr.Button(
                    "\U0001F50D Detect from current checkpoint")

        with gr.Row():
            scenario = gr.Dropdown(
                label="Scenario template (one-click preset — fills the sections below)",
                choices=list(SCENARIOS.keys()),
                value="(none — design from scratch)",
            )
            apply_scenario_btn = gr.Button("Apply scenario", variant="secondary")

        with gr.Accordion("1. Subject / Character", open=True):
            subject_dd = gr.Dropdown(
                label="Subject (count + gender combined — no conflicts possible)",
                choices=list(SUBJECTS.keys()),
                value="Solo - 1girl (young adult female)",
            )
            character_name = gr.Textbox(
                label="Character name(s) — for multi-character, separate with commas; "
                      "one name per character (paren-aware so series tags survive)",
                placeholder="Solo: seele_vollerei  |  Trio: "
                            "kafka_(honkai:_star_rail), seele_vollerei, bronya_rand",
            )
            custom_subject = gr.Textbox(
                label="Extra subject details — SHARED across all characters "
                      "(hair, eyes, accessories that apply to everyone)",
                placeholder="e.g. blue hair, blue eyes, long hair, ahoge, glasses",
                lines=2,
            )

            with gr.Row():
                force_female_only_cb = gr.Checkbox(
                    label="🚫 Force female-only (adds male-exclusion to negative "
                          "— recommended for any all-female subject so the model "
                          "can't slip a boy in)",
                    value=True,
                )
                yuri_cb = gr.Checkbox(
                    label="💕 Add 'yuri' tag (only meaningful with 2+ females — "
                          "use for romantic / intimate same-sex context)",
                    value=False,
                )

        with gr.Accordion("1b. Per-character traits (only used when 2+ characters)",
                          open=False):
            gr.Markdown(
                "Each box describes ONE character's unique traits. Only the first "
                "N boxes are used based on your Subject choice (Pair=2, Trio=3, "
                "Group=up to 4). Leave blank for any character that should share "
                "the defaults. Solo subjects ignore this section."
            )
            with gr.Row():
                char1_traits = gr.Textbox(
                    label="Character 1 traits",
                    placeholder="e.g. blue hair, blue eyes, school uniform, twintails",
                    lines=2,
                )
                char2_traits = gr.Textbox(
                    label="Character 2 traits",
                    placeholder="e.g. red hair, red eyes, casual clothes, short hair",
                    lines=2,
                )
            with gr.Row():
                char3_traits = gr.Textbox(
                    label="Character 3 traits",
                    placeholder="e.g. black hair, brown eyes, business suit",
                    lines=2,
                )
                char4_traits = gr.Textbox(
                    label="Character 4 traits (Group only)",
                    placeholder="e.g. blonde hair, green eyes, sundress",
                    lines=2,
                )
            use_break = gr.Checkbox(
                label="Use BREAK separator between characters "
                      "(off by default — plain commas tend to work better)",
                value=False,
            )

        with gr.Accordion("2. Pose & Action", open=True):
            with gr.Row():
                pose_dd = gr.Dropdown(
                    label="Pose", choices=list(POSES.keys()), value="Standing")
                action_dd = gr.Dropdown(
                    label="Action / gaze", choices=list(ACTIONS.keys()),
                    value="Looking at viewer")

        with gr.Accordion("3. Framing", open=True):
            with gr.Row():
                shot_dd = gr.Dropdown(
                    label="Shot type", choices=list(SHOT_TYPES.keys()),
                    value="Upper body")
                angle_dd = gr.Dropdown(
                    label="Camera angle", choices=list(CAMERA_ANGLES.keys()),
                    value="Eye level (default)")

        with gr.Accordion("4. Outfit", open=True):
            outfit_dd = gr.Dropdown(
                label="Outfit style", choices=list(OUTFIT_PRESETS.keys()),
                value="Casual everyday")

        with gr.Accordion("5. Scene, Time & Lighting", open=True):
            location_dd = gr.Dropdown(
                label="Location", choices=list(LOCATIONS.keys()),
                value="Simple white background")
            with gr.Row():
                time_dd = gr.Dropdown(
                    label="Time of day", choices=list(TIME_OF_DAY.keys()),
                    value="Afternoon")
                lighting_dd = gr.Dropdown(
                    label="Lighting", choices=list(LIGHTING.keys()),
                    value="Soft natural")

        with gr.Accordion("6. Expression & Mood", open=False):
            expression_dd = gr.Dropdown(
                label="Expression", choices=list(EXPRESSIONS.keys()),
                value="Soft smile")

        with gr.Accordion("7. Art Style (model-aware)", open=True):
            style_dd = gr.Dropdown(
                label="Visual style",
                choices=list(ART_STYLES_ILLUSTRIOUS.keys()),
                value="Detailed anime (default)",
            )

        with gr.Accordion("8. Quality, Negative & Booru meta", open=False):
            quality = gr.Radio(
                label="Quality tier (positive baseline)",
                choices=list(ILLUSTRIOUS_QUALITY_TIERS.keys()),
                value="Standard (safe default)",
            )
            negative = gr.Radio(
                label="Negative tier",
                choices=list(ILLUSTRIOUS_NEGATIVE_TIERS.keys()),
                value="Standard (safe default)",
            )
            with gr.Row():
                year   = gr.Dropdown(label="Year tag", choices=YEAR_TAGS, value="newest")
                source = gr.Dropdown(label="Source tag", choices=SOURCE_TAGS, value="(no source tag)")
                rating = gr.Dropdown(label="Rating tag", choices=RATING_TAGS, value="(no rating tag)")

        with gr.Accordion("9. Body / Physique (optional)", open=False):
            body_dd = gr.Dropdown(
                label="Body type / physique - mix safe and [NSFW] freely",
                choices=list(BODY_TYPES.keys()),
                value=NONE,
            )

        with gr.Accordion("9b. Artist style — searchable 33k artist gallery",
                          open=False):
            gr.Markdown(
                "### 🎨 Pick an artist from the full Illustrious / NoobAI catalogue\n"
                "Type any part of an artist's name in the **Search** dropdown — "
                "filtering happens as you type. Pick one and a real Danbooru "
                "sample appears below so you can see exactly what their style "
                "looks like before you commit. Use the strength slider to dial "
                "the influence up or down."
            )

            # Searchable dropdown over all 33,719 artists. Gradio's
            # filterable=True gives instant in-place filtering as the user types.
            full_artist_dd = gr.Dropdown(
                label="🔎 Search artists ({:,} total) — type any part of a name".format(
                    len(FULL_ARTIST_ITEMS)),
                choices=FULL_ARTIST_LABELS,
                value="(none)",
                filterable=True,
                interactive=True,
            )

            with gr.Row():
                preview_btn = gr.Button(
                    "🖼️  Fetch sample image from Danbooru",
                    variant="secondary",
                    scale=1,
                )
                use_artist_btn = gr.Button(
                    "✅  Use this artist in my prompt",
                    variant="primary",
                    scale=1,
                )

            preview_image = gr.Image(
                label="Sample (random pick from Danbooru — refresh button = new pick)",
                interactive=False,
                show_download_button=False,
                height=380,
            )
            preview_info = gr.Markdown(
                value="*Pick an artist above, then click "
                      "**Fetch sample image from Danbooru** to see what their "
                      "style looks like.*",
            )

            # The "currently selected" textbox is what build_prompt actually
            # reads. The dropdown writes here when the user clicks
            # "Use this artist", and the user can also paste/edit freely.
            custom_artist_tag = gr.Textbox(
                label="Selected artist tag (used in the prompt — edit freely)",
                placeholder="e.g. wlop, ciloranko, kantoku — or type any "
                            "Danbooru artist tag",
                lines=1,
            )



            gr.Markdown(
                "---\n**Backup quick-picks** (the small curated list, kept "
                "as a fallback). Overridden by the search/textbox above:"
            )
            artist_dd = gr.Dropdown(
                label="Quick-pick artist (curated list)",
                choices=list(ARTIST_STYLES.keys()),
                value="(none - use only my visual style preset)",
            )
            artist_strength = gr.Slider(
                label="Artist strength (1.0 = full, lower = subtler, "
                      "higher = more dominant). Wraps as (artist:strength) "
                      "when not 1.0.",
                minimum=0.3, maximum=1.5, step=0.05, value=1.0,
            )
            artist_preview = gr.Markdown(
                value=_artist_preview_md("(none - use only my visual style preset)"),
            )

            def _refresh_artist_preview(lbl):
                return _artist_preview_md(lbl)

            artist_dd.change(fn=_refresh_artist_preview,
                             inputs=[artist_dd],
                             outputs=[artist_preview])

            # ---- searchable dropdown wiring ----

            def _on_pick_full_artist(label):
                if not label or label == "(none)":
                    return ("*Pick an artist above, then click "
                            "**Fetch sample image from Danbooru** to see what "
                            "their style looks like.*")
                tag = FULL_ARTIST_LABEL_TO_TAG.get(label, "")
                if not tag:
                    return "*Unknown artist label.*"
                url = "https://danbooru.donmai.us/posts?tags={}".format(tag)
                return ("**Selected**: `{}`  \n"
                        "[View all samples on Danbooru]({}) — or click "
                        "**Fetch sample image** to load one inline.").format(tag, url)

            full_artist_dd.change(
                fn=_on_pick_full_artist,
                inputs=[full_artist_dd],
                outputs=[preview_info],
            )

            def _do_fetch_preview(label):
                if not label or label == "(none)":
                    return None, ("*Pick an artist first.*")
                tag = FULL_ARTIST_LABEL_TO_TAG.get(label, "")
                if not tag:
                    return None, "*Unknown artist.*"
                url = _danbooru_preview_url(tag)
                if not url:
                    return None, ("**Selected**: `{}`  \n"
                                  "(Danbooru returned no samples for this tag — "
                                  "the artist might exist only under a slightly "
                                  "different name. Open the link in your "
                                  "browser to confirm.)\n\n"
                                  "[Search Danbooru for {}]"
                                  "(https://danbooru.donmai.us/posts?tags={})"
                                  ).format(tag, tag, tag)
                view = "https://danbooru.donmai.us/posts?tags={}".format(tag)
                return url, ("**Selected**: `{}`  \n"
                             "Sample loaded above. Click "
                             "**Fetch sample image** again for a different "
                             "random pick.  \n"
                             "[See all samples on Danbooru]({})").format(tag, view)

            preview_btn.click(
                fn=_do_fetch_preview,
                inputs=[full_artist_dd],
                outputs=[preview_image, preview_info],
            )

            def _do_use_artist(label, current):
                if not label or label == "(none)":
                    return current
                return FULL_ARTIST_LABEL_TO_TAG.get(label, current)

            use_artist_btn.click(
                fn=_do_use_artist,
                inputs=[full_artist_dd, custom_artist_tag],
                outputs=[custom_artist_tag],
            )

        with gr.Accordion("10. Intimacy / NSFW level", open=False):
            gr.Markdown(
                "Default is **(safe - none)**. Pick a higher tier only if you "
                "want the model to lean into intimate, suggestive, or explicit "
                "content. [NSFW] options require a permissive checkpoint "
                "(Illustrious / NoobAI are fine)."
            )
            intimacy_dd = gr.Dropdown(
                label="Intimacy / explicitness",
                choices=list(INTIMACY_LEVELS.keys()),
                value="(safe - none)",
            )

        with gr.Accordion("11. Custom extras (optional)", open=False):
            extra_positive = gr.Textbox(
                label="Append to POSITIVE",
                placeholder="e.g. cinematic lighting, <lora:my_style:0.7>",
                lines=2,
            )
            extra_negative = gr.Textbox(
                label="Append to NEGATIVE",
                placeholder="e.g. monochrome, censored, multiple views",
                lines=2,
            )

        build_btn = gr.Button("Build prompt →", variant="primary", size="lg")

        positive_out = gr.Textbox(
            label="Generated POSITIVE prompt", lines=8,
            show_copy_button=True,
            placeholder="Click 'Build prompt' to assemble.",
        )
        negative_out = gr.Textbox(
            label="Generated NEGATIVE prompt", lines=5,
            show_copy_button=True,
        )
        status = gr.Textbox(label="Status", interactive=False, lines=2,
                            value="Pick a scenario or fill in the sections, "
                                  "then click 'Build prompt'.")
        with gr.Row():
            send_t2i_btn = gr.Button("Send to txt2img", variant="primary")
            send_i2i_btn = gr.Button("Send to img2img")

        params_state = gr.Textbox(visible=False)

        # ---- Wiring ----

        def _do_detect():
            ckpt = _get_loaded_checkpoint()
            fam_label = _detect_model_family(ckpt)
            md = "**Detected**: `{}` → **{}**".format(
                ckpt or "(no checkpoint loaded)", fam_label)
            styles = list(MODELS[fam_label]["art_styles"].keys())
            return md, gr.update(value=fam_label), gr.update(
                choices=styles, value=styles[0] if styles else None)

        detect_btn.click(fn=_do_detect,
                         outputs=[detected_label, family, style_dd])

        def _on_family_change(family_label):
            _, fam = _resolve_family(family_label)
            styles = list(fam["art_styles"].keys())
            return gr.update(choices=styles, value=styles[0] if styles else None)

        family.change(fn=_on_family_change, inputs=[family], outputs=[style_dd])

        def _apply_scenario(scenario_label):
            tmpl = SCENARIOS.get(scenario_label, {})
            if not tmpl:
                return [gr.update()] * 11 + [
                    "Cleared — fill the sections manually."
                ]
            return [
                gr.update(value=tmpl.get("subject", "Solo - 1girl (young adult female)")),
                gr.update(value=tmpl.get("pose",     "Standing")),
                gr.update(value=tmpl.get("action",   "Looking at viewer")),
                gr.update(value=tmpl.get("shot",     "Upper body")),
                gr.update(value=tmpl.get("angle",    "Eye level (default)")),
                gr.update(value=tmpl.get("outfit",   "Casual everyday")),
                gr.update(value=tmpl.get("location", "Simple white background")),
                gr.update(value=tmpl.get("time",     "Afternoon")),
                gr.update(value=tmpl.get("lighting", "Soft natural")),
                gr.update(value=tmpl.get("expr",     "Soft smile")),
                gr.update(value=tmpl.get("style",    "Detailed anime (default)")),
                "Scenario applied: {} - you can still tweak any section.".format(scenario_label),
            ]

        apply_scenario_btn.click(
            fn=_apply_scenario, inputs=[scenario],
            outputs=[subject_dd, pose_dd, action_dd, shot_dd, angle_dd,
                     outfit_dd, location_dd, time_dd, lighting_dd,
                     expression_dd, style_dd, status],
        )

        def _format_params(positive: str, negative: str) -> str:
            if not positive and not negative:
                return ""
            lines = [positive or ""]
            if negative:
                lines.append("Negative prompt: {}".format(negative))
            lines.append("Steps: 32, Sampler: Euler a, CFG scale: 6, Size: 1024x1344")
            return "\n".join(lines)

        def _do_build(family_label,
                      subject_l, char_name, custom_subj,
                      pose_l, action_l,
                      shot_l, angle_l,
                      outfit_l,
                      loc_l, time_l, light_l,
                      expr_l, style_l,
                      quality_t, negative_t, year_t, source_t, rating_t,
                      extra_pos, extra_neg,
                      c1, c2, c3, c4, ub,
                      body_l, intimacy_l,
                      artist_l, artist_s, custom_artist,
                      force_female, yuri_on):
            pos, neg, msg = build_prompt(
                family_label, subject_l, char_name, custom_subj,
                pose_l, action_l, shot_l, angle_l, outfit_l,
                loc_l, time_l, light_l, expr_l, style_l,
                quality_t, negative_t, year_t, source_t, rating_t,
                extra_pos, extra_neg,
                char1_traits=c1, char2_traits=c2,
                char3_traits=c3, char4_traits=c4,
                use_break=ub,
                body_label=body_l,
                intimacy_label=intimacy_l,
                artist_label=artist_l,
                artist_strength=artist_s,
                custom_artist_tag=custom_artist,
                force_female_only=force_female,
                yuri=yuri_on,
            )
            return pos, neg, msg, _format_params(pos, neg)

        build_btn.click(
            fn=_do_build,
            inputs=[family,
                    subject_dd, character_name, custom_subject,
                    pose_dd, action_dd,
                    shot_dd, angle_dd,
                    outfit_dd,
                    location_dd, time_dd, lighting_dd,
                    expression_dd, style_dd,
                    quality, negative, year, source, rating,
                    extra_positive, extra_negative,
                    char1_traits, char2_traits, char3_traits, char4_traits,
                    use_break,
                    body_dd, intimacy_dd,
                    artist_dd, artist_strength, custom_artist_tag,
                    force_female_only_cb, yuri_cb],
            outputs=[positive_out, negative_out, status, params_state],
        )

        # Paste-params wiring (Forge infotext API)
        _paste_mod = None
        for mod_name in ("modules.infotext_utils",
                         "modules.generation_parameters_copypaste"):
            try:
                import importlib
                _paste_mod = importlib.import_module(mod_name)
                break
            except ImportError:
                continue

        if _paste_mod is not None and hasattr(_paste_mod, "register_paste_params_button"):
            ParamBinding = _paste_mod.ParamBinding
            register = _paste_mod.register_paste_params_button
            register(ParamBinding(
                paste_button=send_t2i_btn,
                tabname="txt2img",
                source_text_component=params_state,
            ))
            register(ParamBinding(
                paste_button=send_i2i_btn,
                tabname="img2img",
                source_text_component=params_state,
            ))
        else:
            send_t2i_btn.click(
                fn=lambda: "[!] Send-to-txt2img not wired. Copy manually.",
                outputs=[status],
            )
            send_i2i_btn.click(
                fn=lambda: "[!] Send-to-img2img not wired. Copy manually.",
                outputs=[status],
            )

    return [(ui, "Prompt Enhancer", "prompt_enhancer")]


if script_callbacks is not None:
    script_callbacks.on_ui_tabs(_on_ui_tabs)
