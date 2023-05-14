from diffusers import StableDiffusionPipeline
import torch
from uuid import uuid4
from PIL import Image
import gradio as gr

model_path = "sd-pokemon-model"
pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
pipe.to("cuda")

def predict(prompt):
    image = pipe(prompt=prompt).images[0]
    tmp_filename = f"/tmp/{uuid4()}.png"
    image.save(tmp_filename)
    img = Image.open(tmp_filename)
    return img

title = "Stable Diffusion Pokemon Generator"
description = "Generate Pokemon from text prompts using Stable Diffusion v1.4"
article="<p style='text-align: center'><a href='https://github.com/hoangkimthuc/diffusers' target='_blank'>Click here to see the original repo of this app</a></p>"
examples = ["yoda", "pikachu", "charmander"]
interpretation='default'
enable_queue=True


text_to_image_app = gr.Interface(fn=predict, 
             inputs="text", 
             outputs="image",
             title=title,
             description=description,
             article=article,
             examples=examples,
             interpretation=interpretation,
             enable_queue=enable_queue
             )
text_to_image_app.launch(share=True)

    

