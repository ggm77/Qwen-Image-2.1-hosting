import sys, os, gc, json, time, numpy as np, torch
sys.path.insert(0, os.getcwd())
from PIL import Image
from app.core.config import settings
from app.utils.model_manager import ModelManager
from diffusers import QwenImage21Pipeline
dev="mps"; D=sys.argv[1]; OUT=D+"/g1060"; os.makedirs(OUT,exist_ok=True)
KS=["text","face","glass","ko"]
PR=['A storefront sign that reads "OPEN 24 HOURS" in glowing neon letters, rainy street at night, photo',
 "Close-up portrait of an elderly fisherman with deep wrinkles, natural window light, photorealistic",
 "A glass of water on a wooden table with refraction and caustics, macro photo",
 "눈 내리는 서울 거리에서 한복을 입고 우산을 든 여성, 사진"]
g1060=torch.load("/Users/user/embeds_nf4_gtx1060.pt",weights_only=True,map_location="cpu")
pipe=QwenImage21Pipeline.from_pretrained(settings.MODEL_PATH,transformer=None,vae=None,torch_dtype=torch.bfloat16).to(dev)
ref=[]
for p in PR:
    with torch.no_grad(): e,m,_=pipe.encode_prompt(prompt=p,device=dev)
    ref.append((e.float().cpu(),m))
del pipe; gc.collect(); torch.mps.empty_cache()
stats={}
for i,k in enumerate(KS):
    a=ref[i][0][0]; b=g1060[i][0][0]
    cos=torch.nn.functional.cosine_similarity(a,b,dim=-1)
    stats[k]=dict(cos_mean=round(cos.mean().item(),4),cos_min=round(cos.min().item(),4),
                  cos_min_idx=int(cos.argmin()),rel_l2=round(((a-b).norm()/a.norm()).item(),4))
print("[C] 임베딩(1060 NF4 fp16 vs Mac bf16):",json.dumps(stats,ensure_ascii=False),flush=True)
mm=ModelManager(dev); gp=mm.load_generator("int8")
for i,k in enumerate(KS):
    e,m=g1060[i]; g=torch.Generator(device=dev).manual_seed(7)
    t=time.perf_counter()
    img=gp(prompt_embeds=e.to(dev,dtype=torch.bfloat16),prompt_embeds_mask=m.to(dev),height=512,width=512,
           num_inference_steps=20,true_cfg_scale=1.0,generator=g).images[0]
    img.save(f"{OUT}/{k}_G.png"); print(f"[C] image {k} {time.perf_counter()-t:.0f}s",flush=True)
L=lambda p: np.asarray(Image.open(p).convert("RGB")).astype(np.float64)
def psnr(a,b):
    mse=((a-b)**2).mean(); return round(10*np.log10(255**2/mse),1)
rows={}
for k in KS:
    g=L(f"{OUT}/{k}_G.png"); a=L(f"{D}/{k}_A_bf16.png"); b=L(f"{D}/{k}_B_nf4_bf16.png")
    rows[k]=dict(G_vs_A=psnr(g,a),G_vs_B=psnr(g,b),B_vs_A=psnr(b,a))
print("[C] PSNR:",json.dumps(rows,ensure_ascii=False),flush=True)
W=Image.new("RGB",(300*3,300*4))
for r,k in enumerate(KS):
    for c,p in enumerate([f"{D}/{k}_A_bf16.png",f"{D}/{k}_B_nf4_bf16.png",f"{OUT}/{k}_G.png"]):
        W.paste(Image.open(p).resize((300,300)),(c*300,r*300))
W.save(D+"/../g1060_grid.png"); print("[C] DONE",flush=True)
