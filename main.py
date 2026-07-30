import os, glob, csv
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

BASE = r"C:\Users\Akshat.jain\Downloads\details (3)\Assess"
SKU_DIR = os.path.join(BASE, "sku")
QUERY_DIR = os.path.join(BASE, "query")
OUT_CSV = os.path.join(BASE, "similarity_rankings.csv")

device = torch.device("cpu")

seg_processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
seg_model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b0-finetuned-ade-512-512"
).to(device).eval()
FLOOR_ID = 3  


def get_floor_mask(pil_img):
    inputs = seg_processor(images=pil_img, return_tensors="pt")
    with torch.no_grad():
        logits = seg_model(**inputs).logits
    upsampled = F.interpolate(logits, size=pil_img.size[::-1], mode="bilinear", align_corners=False)
    pred = upsampled.argmax(dim=1)[0].numpy()
    return pred == FLOOR_ID


weights = ResNet18_Weights.IMAGENET1K_V1
resnet = resnet18(weights=weights).to(device).eval()

feat_layers = {}
def _hook(name):
    def fn(module, inp, out):
        feat_layers[name] = out.detach()
    return fn
resnet.layer1.register_forward_hook(_hook("layer1"))
resnet.layer2.register_forward_hook(_hook("layer2"))

norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def gram(feat):
    c, h, w = feat.shape
    f = feat.reshape(c, h * w)
    return (f @ f.t()) / (c * h * w)


def tile_texture_descriptor(tile_pil):
    base = tile_pil.resize((224, 224), Image.BILINEAR)
    vecs = []
    for angle in (0, 90, 180, 270):  # average over rotations -> orientation-robust
        im = base.rotate(angle, expand=False)
        t = transforms.functional.to_tensor(im)
        t = norm(t).unsqueeze(0).to(device)
        with torch.no_grad():
            resnet(t)
        g1 = gram(feat_layers["layer1"][0]).flatten()
        g2 = gram(feat_layers["layer2"][0]).flatten()
        v = torch.cat([g1, g2])
        v = v / (v.norm() + 1e-8)
        vecs.append(v.numpy())
    return np.mean(vecs, axis=0)


def rgb_to_lab(pixels_uint8):
    rgb = pixels_uint8.astype(np.float32) / 255.0
    low = rgb <= 0.04045
    lin = np.where(low, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    X = lin @ np.array([0.4124564, 0.3575761, 0.1804375], dtype=np.float32)
    Y = lin @ np.array([0.2126729, 0.7151522, 0.0721750], dtype=np.float32)
    Z = lin @ np.array([0.0193339, 0.1191920, 0.9503041], dtype=np.float32)
    Xn, Yn, Zn = 0.95047, 1.0, 1.08883
    d = 6 / 29
    def f(t):
        return np.where(t > d ** 3, np.cbrt(t), t / (3 * d * d) + 4 / 29)
    fx, fy, fz = f(X / Xn), f(Y / Yn), f(Z / Zn)
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    return np.stack([L, a, b], axis=1)


def color_descriptor(pil_img, mask=None):
    arr = np.array(pil_img.convert("RGB"))
    pixels = arr[mask] if mask is not None and mask.sum() > 1000 else arr.reshape(-1, 3)
    lab = rgb_to_lab(pixels)
    hist, _ = np.histogramdd(lab, bins=(8, 8, 8), range=[(0, 100), (-40, 60), (-40, 60)])
    hist = hist / (hist.sum() + 1e-8)
    return hist.flatten(), lab.mean(axis=0)


def hist_similarity(h1, h2):
    return float(np.sum(np.minimum(h1, h2)))  # histogram intersection, in [0,1]


def get_floor_tiles(pil_img, mask, n_grid=8, min_frac=0.85):
    W, H = pil_img.size
    arr = np.array(pil_img.convert("RGB"))
    tw, th = W // n_grid, H // n_grid
    def collect(thr):
        out = []
        for i in range(n_grid):
            for j in range(n_grid):
                y0, y1 = i * th, (i + 1) * th
                x0, x1 = j * tw, (j + 1) * tw
                m = mask[y0:y1, x0:x1]
                if m.size and m.mean() >= thr:
                    out.append(Image.fromarray(arr[y0:y1, x0:x1]))
        return out
    tiles = collect(min_frac)
    if len(tiles) < 4:
        tiles = collect(0.5)
    return tiles


def process_query(path):
    img = Image.open(path).convert("RGB")
    mask = get_floor_mask(img)
    floor_frac = float(mask.mean())
    tiles = get_floor_tiles(img, mask, n_grid=8, min_frac=0.85)
    if not tiles:
        W, H = img.size
        tiles = [img.crop((int(W * 0.1), int(H * 0.6), int(W * 0.9), H))]
    tex = np.mean([tile_texture_descriptor(t) for t in tiles], axis=0)
    tex = tex / (np.linalg.norm(tex) + 1e-8)
    color_hist, mean_lab = color_descriptor(img, mask=mask)
    return {"tex": tex, "color_hist": color_hist, "mean_lab": mean_lab,
            "floor_frac": floor_frac, "n_tiles": len(tiles)}


def process_sku(path):
    img = Image.open(path).convert("RGB")
    W, H = img.size
    n_grid = 4
    tw, th = W // n_grid, H // n_grid
    tiles = [img.crop((j * tw, i * th, (j + 1) * tw, (i + 1) * th))
             for i in range(n_grid) for j in range(n_grid)]
    tex = np.mean([tile_texture_descriptor(t) for t in tiles], axis=0)
    tex = tex / (np.linalg.norm(tex) + 1e-8)
    color_hist, mean_lab = color_descriptor(img, mask=None)
    return {"tex": tex, "color_hist": color_hist, "mean_lab": mean_lab}


def natkey(p):
    return int(os.path.splitext(os.path.basename(p))[0])


sku_files = sorted(glob.glob(os.path.join(SKU_DIR, "*.jpg")), key=natkey)
query_files = sorted(glob.glob(os.path.join(QUERY_DIR, "*.jpg")), key=natkey)

print(f"Found {len(sku_files)} SKU images, {len(query_files)} query images")

print("Processing SKU images...")
sku_data = {}
for p in sku_files:
    name = os.path.basename(p)
    sku_data[name] = process_sku(p)
    print(f"  {name} done")

print("Processing query images (floor segmentation)...")
query_data = {}
for p in query_files:
    name = os.path.basename(p)
    query_data[name] = process_query(p)
    d = query_data[name]
    print(f"  {name}: floor_frac={d['floor_frac']:.2f} tiles_used={d['n_tiles']} mean_Lab={np.round(d['mean_lab'],1)}")

print("\nSKU mean Lab values:")
for name, d in sku_data.items():
    print(f"  {name}: mean_Lab={np.round(d['mean_lab'],1)}")

TEX_W, COLOR_W = 0.45, 0.55

all_rows = []
print("\n================ RANKINGS ================")
for qname in [os.path.basename(p) for p in query_files]:
    qd = query_data[qname]
    scores = []
    for sname in [os.path.basename(p) for p in sku_files]:
        sd = sku_data[sname]
        tex_sim = float(np.dot(qd["tex"], sd["tex"]))
        color_sim = hist_similarity(qd["color_hist"], sd["color_hist"])
        final = TEX_W * tex_sim + COLOR_W * color_sim
        scores.append((sname, final, tex_sim, color_sim))
    scores.sort(key=lambda x: -x[1])
    print(f"\n=== Query {qname} ===")
    for rank, (sname, final, tex_sim, color_sim) in enumerate(scores, 1):
        print(f"{rank:2d}. {sname:8s} score={final:.4f}  (texture={tex_sim:.3f}, color={color_sim:.3f})")
        all_rows.append({"query": qname, "rank": rank, "sku": sname,
                          "score": round(final, 4), "texture_sim": round(tex_sim, 4),
                          "color_sim": round(color_sim, 4)})

with open(OUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["query", "rank", "sku", "score", "texture_sim", "color_sim"])
    writer.writeheader()
    writer.writerows(all_rows)

print(f"\nSaved full rankings to {OUT_CSV}")
