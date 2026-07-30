# Floor Similarity Ranking

This project looks at 5 photos of rooms (`query/`) and 20 photos of flooring products
(`sku/`), and for each room photo, it puts the 20 products in order from "most similar
looking floor" to "least similar."

## Files

| File | What it is |
|---|---|
| `rank_floors.py` | The script that does the matching. Run it to get the results. |
| `similarity_rankings.csv` | The results: every product ranked for every room photo. |
| `sku` | image which are to rank in this folder|
| `query` | query images with all objects |

## Approach and pipeline

The tricky part of this problem is that the two sets of photos don't look alike to begin
with. A product photo is a clean, straight-down close-up of just the flooring. A room photo
is a normal photo of a room — the floor is only part of it, there's furniture sitting on top
of it, and because of the camera angle, the wood lines run diagonally instead of straight up
and down. If you just compared the two whole photos directly, you'd mostly be comparing "does
this room look like this close-up," which it never will, even for a perfect match.

So before comparing anything, the code first cleans up the room photo so it's only looking at
floor:

1. **Find the floor.** It uses an  model to figure out, pixel by pixel, which part of the
   room photo is actually floor (as opposed to a wall, a sofa, a lamp, etc.), and throws away
   everything that isn't floor.
2. **Cut it into small clean pieces.** It chops the photo into a grid and only keeps the
   squares that are basically all floor, so a square that's half rug or half chair leg gets
   dropped.
3. **Describe what floor looks like**, in two separate ways: its color, and its texture
   (the grain pattern). Same for every product photo.
4. **Compare** the room's floor description to each product's description, and combine both
   into one score. Sort the products by that score, best match first.

## Models or techniques used

- **A pretrained segmentation model** (SegFormer, trained on a dataset called ADE20K) — this
  is what finds the floor in the room photo. It was trained to recognize dozens of things in
  indoor/outdoor scenes, including "floor" as one of its categories, so it can point out
  exactly which pixels are floor without us having to draw that by hand.
- **A pretrained image recognition model** (ResNet18, normally used to recognize objects in
  photos) — here it's not used to recognize objects at all. Instead, we borrow the numbers it
  produces internally while looking at an image, because those numbers turn out to describe
  the *texture* of an image well (grain, roughness, pattern)
- **A basic color comparison** — converting each photo's colors into a format that separates
  "how light or dark" from "what color" (called Lab color), then building a simple summary of
  what colors are present and how much of each.
- **A small trick for camera angle** — since product photos are straight-on and room photos
  are at an angle, each floor patch is checked in 4 different rotations and averaged, so the
  comparison isn't thrown off just because the wood lines point a different direction.

## How similarity is computed

Every photo (room floor, or product) ends up boiled down to two things:
- a **texture summary** (a list of numbers describing the grain pattern)
- a **color summary** (a list of numbers describing what colors/shades are present, and how much)

For each room photo compared to each product photo:

- Compare the two texture summaries → get a **texture similarity** number
- Compare the two color summaries → get a **color similarity** number
- Combine them: **final score = 45% texture similarity + 55% color similarity**

Color is weighted a bit more because, in practice, all the products came out looking fairly
similar to each other in texture (they're all wood grain, after all), so color ends up being
the bigger differentiator. All 20 products get this final score for a given room photo, and
they're simply sorted highest score first.

## Limitations and possible improvements

- **We don't have a "correct answer" to check against.** The 45/55 weighting, and a few other
  settings in the script, were chosen because they seemed reasonable and gave sensible-looking
  results — not because they were tested against a known right answer. 
- **If the floor-finding step gets confused**, the whole process for that photo gets weaker.
  This could happen with unusual lighting, a floor material the model isn't used to seeing, or
  heavy furniture coverage. There's a basic fallback (just grab the bottom-middle of the photo)
  but it's not a real fix.
- **The camera-angle fix is a rough patch, not a perfect solution.** It helps but doesn't fully
  erase the difference between a straight-on product shot and an angled room shot.
- **The texture-reading model wasn't built specifically for flooring or materials** — it was
  built to recognize everyday objects, and we're just repurposing part of it. A model actually
  trained on textures/materials would likely tell the 20 products apart more clearly than it
  does now.
- **Plank size/pattern scale isn't really considered.** Two floors could have the same color
  and grain style but very different plank widths (thin vs. wide boards), and the current
  method might still call them a close match.
- **The color comparison is fairly basic.** It can treat "slightly off" and "very different"
  colors too similarly in some cases. A more advanced color-distance method would likely be
  more accurate.


## Running it

```bash
pip install torch torchvision transformers numpy pillow
python rank_floors.py
```

This downloads two small pretrained models the first time you run it, then creates
`similarity_rankings.csv` with the results.

## What's in the results file (`similarity_rankings.csv`)

| column | meaning |
|---|---|
| `query` | Which room photo this row is about |
| `rank` | 1 = best match for that room |
| `sku` | Which product this row is about |
| `score` | Overall match score (higher = more similar) |
| `texture_sim` | Just the texture/grain match score |
| `color_sim` | Just the color match score |
