import sys, os, time, gc, json, numpy as np, torch, torch.nn as nn
sys.path.insert(0, os.getcwd())
from PIL import Image
from app.core.config import settings
from app.utils.model_manager import ModelManager
from diffusers import QwenImage21Pipeline

dev = "mps"; OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
PROMPTS = {
 "text":  'A storefront sign that reads "OPEN 24 HOURS" in glowing neon letters, rainy street at night, photo',
 "face":  "Close-up portrait of an elderly fisherman with deep wrinkles, natural window light, photorealistic",
 "glass": "A glass of water on a wooden table with refraction and caustics, macro photo",
 "ko":    "눈 내리는 서울 거리에서 한복을 입고 우산을 든 여성, 사진",
}
NF4 = torch.tensor([-1.0,-0.6961928009986877,-0.5250730514526367,-0.39491748809814453,-0.28444138169288635,-0.18477343022823334,
  -0.09105003625154495,0.0,0.07958029955625534,0.16093020141506195,0.24611230194568634,0.33791524171829224,
  0.44070982933044434,0.5626170039176941,0.7229568362236023,1.0])
MID = (NF4[1:] + NF4[:-1]) / 2

@torch.no_grad()
def nf4_roundtrip(w, block=64):
    shp, dt = w.shape, w.dtype
    x = w.float().reshape(-1, block)
    s = x.abs().amax(1, keepdim=True).clamp_min(1e-8)
    idx = torch.bucketize((x / s).cpu(), MID).to(x.device)
    y = NF4.to(x.device)[idx] * s
    return y.reshape(shp).to(dt)

def encode_all(pipe, tag):
    res = {}
    for k, p in PROMPTS.items():
        with torch.no_grad():
            e, m, _ = pipe.encode_prompt(prompt=p, device=dev)
        res[k] = (e.float().cpu(), m.cpu() if m is not None else None)
        print(f"[N] {tag} {k}: embeds {tuple(e.shape)} dtype={e.dtype} nan={bool(torch.isnan(e).any())} inf={bool(torch.isinf(e).any())}", flush=True)
    return res

t = time.perf_counter()
pipe = QwenImage21Pipeline.from_pretrained(settings.MODEL_PATH, transformer=None, vae=None, torch_dtype=torch.bfloat16).to(dev)
te = pipe.text_encoder
print(f"[N] encoder loaded {time.perf_counter()-t:.0f}s", flush=True)
embeds = {"A_bf16": encode_all(pipe, "A_bf16")}

lins = [(n, m) for n, m in te.named_modules() if isinstance(m, nn.Linear) and "visual" not in n and "lm_head" not in n]
print(f"[N] NF4 대상 Linear {len(lins)}개 (예: {lins[0][0]})", flush=True)
errs = []
with torch.no_grad():
    for n, m in lins:
        w = m.weight.data; q = nf4_roundtrip(w)
        errs.append(((q.float()-w.float()).norm() / w.float().norm()).item()); m.weight.data = q
print(f"[N] 가중치 상대오차 평균 {np.mean(errs):.4f} 최대 {np.max(errs):.4f}", flush=True)
embeds["B_nf4_bf16"] = encode_all(pipe, "B_nf4_bf16")

te.half()
embeds["C_nf4_fp16"] = encode_all(pipe, "C_nf4_fp16")
del pipe, te; gc.collect(); torch.mps.empty_cache()

# 임베딩 비교
ref = embeds["A_bf16"]; stats = {}
for v in ("B_nf4_bf16", "C_nf4_fp16"):
    for k in PROMPTS:
        a, ma = ref[k]; b, mb = embeds[v][k]
        if a.shape != b.shape: stats[f"{v}/{k}"] = "shape mismatch"; continue
        sel = (ma[0].bool() if ma is not None else torch.ones(a.shape[1], dtype=torch.bool))
        x, y = a[0][sel], b[0][sel]
        if not torch.isfinite(y).all(): stats[f"{v}/{k}"] = "non-finite"; continue
        cos = torch.nn.functional.cosine_similarity(x, y, dim=-1)
        stats[f"{v}/{k}"] = dict(cos_mean=round(cos.mean().item(),4), cos_min=round(cos.min().item(),4),
                                 rel_l2=round(((x-y).norm()/x.norm()).item(),4), tokens=int(sel.sum()))
print("[N] 임베딩 비교:", json.dumps(stats, ensure_ascii=False, indent=1), flush=True)

# 이미지 생성 (int8 DiT, 512, 20스텝, 시드 7)
mm = ModelManager(dev); gp = mm.load_generator("int8")
def gen(e, m):
    g = torch.Generator(device=dev).manual_seed(7)
    return gp(prompt_embeds=e.to(dev, dtype=torch.bfloat16), prompt_embeds_mask=m.to(dev) if m is not None else None,
              height=512, width=512, num_inference_steps=20, true_cfg_scale=1.0, generator=g).images[0]
for k in PROMPTS:
    for v in ("A_bf16", "B_nf4_bf16", "C_nf4_fp16"):
        e, m = embeds[v][k]
        if not torch.isfinite(e).all(): print(f"[N] skip {v}/{k} (non-finite)", flush=True); continue
        t = time.perf_counter(); gen(e, m).save(f"{OUT}/{k}_{v}.png"); print(f"[N] image {k} {v} {time.perf_counter()-t:.0f}s", flush=True)

def load(p): return np.asarray(Image.open(p).convert("RGB")).astype(np.float64)
rows = {}
for k in PROMPTS:
    a = load(f"{OUT}/{k}_A_bf16.png")
    for v in ("B_nf4_bf16", "C_nf4_fp16"):
        pth = f"{OUT}/{k}_{v}.png"
        if not os.path.exists(pth): continue
        b = load(pth); mse = ((a-b)**2).mean()
        rows[f"{k}/{v}"] = dict(psnr=round(10*np.log10(255**2/mse),1) if mse > 0 else "inf", mae=round(np.abs(a-b).mean(),2))
print("[N] 이미지 비교(A_bf16 기준):", json.dumps(rows, ensure_ascii=False, indent=1))
print("[N] DONE", flush=True)
