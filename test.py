import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = "/root/autodl-fs/.cache/huggingface"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from diffusers import QwenImageLayeredPipeline
import torch
from PIL import Image
import gc

torch.cuda.empty_cache()
gc.collect()

pipeline = QwenImageLayeredPipeline.from_pretrained(
    "Qwen/Qwen-Image-Layered",
    torch_dtype=torch.bfloat16,   # 直接加载为 bfloat16
    low_cpu_mem_usage=True,       # 开启低内存加载模式（关键）
    # device_map="balanced"       # 如果依然爆内存，可以尝试解开这行注释（需要安装 accelerate 库）
)
#pipeline = pipeline.to("cuda")
# ---【关键修改】显存优化策略 ---
# 不要用 pipeline.to("cuda")，那个是把所有东西一股脑塞进显卡
# 使用 enable_model_cpu_offload，它会自动在 CPU 和 GPU 之间搬运模型层
#pipeline.enable_model_cpu_offload()
pipeline.enable_sequential_cpu_offload()
# 开启 VAE 切片，防止生成最后大图时爆显存
#pipeline.enable_vae_slicing()
#pipeline.set_progress_bar_config(disable=None)

print("step1")

image = Image.open("test_images/1.png").convert("RGBA")
inputs = {
    "image": image,
    "generator": torch.Generator(device='cuda').manual_seed(777),
    "true_cfg_scale": 4.0,
    "negative_prompt": " ",
    "num_inference_steps": 50,
    "num_images_per_prompt": 1,
    "layers": 4,
    "resolution": 640,      # Using different bucket (640, 1024) to determine the resolution. For this version, 640 is recommended
    "cfg_normalize": True,  # Whether enable cfg normalization.
    "use_en_prompt": True,  # Automatic caption language if user does not provide caption
}
print("step2")

with torch.inference_mode():
    output = pipeline(**inputs)
    output_image = output.images[0]
    
print("step3")

for i, image in enumerate(output_image):
    image.save(f"{i}.png")
