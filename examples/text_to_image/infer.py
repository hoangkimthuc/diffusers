from diffusers import StableDiffusionPipeline
import torch
from time import time
model_path = "sd-pokemon-model"
pipe = StableDiffusionPipeline.from_pretrained(model_path)
# pipe.to("cuda")

start = time()
image = pipe(prompt="yoda").images[0]
image.save("yoda-pokemon-test.png")
print(f"Time elapsed: {time() - start} seconds")