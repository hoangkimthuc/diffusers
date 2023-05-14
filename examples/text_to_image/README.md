# Task 1: Choosing model

# Stable Diffusion text-to-image fine-tuning

The `train_text_to_image.py` script shows how to fine-tune stable diffusion model on your own dataset.

### How to install the code requirements.

First, clone the repo and then create a conda env from the env.yaml file and activate the env
```bash
git clone https://github.com/hoangkimthuc/diffusers.git
cd diffusers/examples/text_to_image
conda env create -f env.yaml
conda activate stable_diffusion
```

Before running the scripts, make sure to install the library's training dependencies:

**Important**

To make sure you can successfully run the latest versions of the example scripts, we highly recommend **installing from source** and keeping the install up to date as we update the example scripts frequently and install some example-specific requirements. To do this, execute the following steps in a new virtual environment:
```bash
cd diffusers
pip install .
```

Then cd in the diffusers/examples/text_to_image folder and run
```bash
pip install -r requirements.txt
```

And initialize an [🤗Accelerate](https://github.com/huggingface/accelerate/) environment with:

```bash
accelerate config
```

### Steps to run the training.

You need to accept the model license before downloading or using the weights. In this example we'll use model version `v1-4`, so you'll need to visit [its card](https://huggingface.co/CompVis/stable-diffusion-v1-4), read the license and tick the checkbox if you agree. 

You have to be a registered user in 🤗 Hugging Face Hub, and you'll also need to use an access token for the code to work. For more information on access tokens, please refer to [this section of the documentation](https://huggingface.co/docs/hub/security-tokens).

Run the following command to authenticate your token

```bash
huggingface-cli login
```

If you have already cloned the repo, then you won't need to go through these steps.

<br>

#### Hardware
With `gradient_checkpointing` and `mixed_precision` it should be possible to fine tune the model on a single 24GB GPU. For higher `batch_size` and faster training it's better to use GPUs with >30GB memory.

**___Note: Change the `resolution` to 768 if you are using the [stable-diffusion-2](https://huggingface.co/stabilityai/stable-diffusion-2) 768x768 model.___**
<!-- accelerate_snippet_start -->
```bash
bash train.sh
```
<!-- accelerate_snippet_end -->

### Sample input/output after training

Once the training is finished the model will be saved in the `output_dir` specified in the command. In this example it's `sd-pokemon-model`. To load the fine-tuned model for inference just pass that path to `StableDiffusionPipeline`


```python
from diffusers import StableDiffusionPipeline

model_path = "sd-pokemon-model"
pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
pipe.to("cuda")

image = pipe(prompt="yoda").images[0]
image.save("yoda-pokemon.png")
```
The output with the prompt "yoda" is saved in the `yoda-pokemon.png` image file.

### Name and link to the training dataset.

Dataset name: pokemon-blip-captions

Dataset link: https://huggingface.co/datasets/lambdalabs/pokemon-blip-captions

### The number of model parameters to determine the model’s complexity.

Note: CLIPTextModel (text conditioning model) and AutoencoderKL (image generating model) are frozen, only the Unet (the diffusion model) is trained.
The number of trainable parameters in the script: 859_520_964
To get this number, you can put a breakpoint by calling `breakpoint()` at line 813 of the `train_text_to_image.py` file and then run `train.sh`. Once the pbd session stops at that line, you can check the model's parameters by `p unet.num_parameters()`.

### The model evaluation metric (CLIP score)
CLIP score is a measure of how well the generated images match the prompts.

Validation prompts to calculate the CLIP scores:
```python
prompts = [
    "a photo of an astronaut riding a horse on mars",
    "A high tech solarpunk utopia in the Amazon rainforest",
    "A pikachu fine dining with a view to the Eiffel Tower",
    "A mecha robot in a favela in expressionist style",
    "an insect robot preparing a delicious meal",
    "A small cabin on top of a snowy mountain in the style of Disney, artstation",
]
```
To calculate the CLIP score for the above prompts, run:
```bash
python metrics.py
```



