# Task 1: Choosing model

## Chosen model: Stable Diffusion text-to-image v.1-4

### Note:
The root directory of the project is `diffusers/examples/text_to_image`
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

```bash
bash train.sh
```

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
![alt text](https://github.com/hoangkimthuc/diffusers/blob/main/examples/text_to_image/yoda-pokemon.png?raw=true)

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

### Link to the trained model

https://drive.google.com/file/d/1xzVUO0nZn-0oaJgHOWjrYKHmGUlsoJ1g/view?usp=sharing

### Modifications made to the original code
- Add metrics and gradio_app scripts
- Remove redundunt code
- Add training bash script
- Improve readme
- Add conda env.yaml file and add more dependencies for the web app

# Task 2: Using the model in a web application

### Install the code requirements and run the web app locally

Dependencies for the app is already installed in the requirements.txt.

There are 2 ways to run the app with the trained model:

1. Follow Task 1 to train the model. The model cache is saved in the `sd-pokemon-model` directory

2. Download and decompress the zip file containing the trained model to a folder named `sd-pokemon-model` and put that folder in the root directory of the project.

To run the web app locally, run
```bash
python gradio_app.py
```
There will be also a link to access the web app publicly but it will only be available within 3 days.

### A screenshot of the output.

Below is the screenshot of the demo web app

![alt text](https://github.com/hoangkimthuc/diffusers/blob/main/examples/text_to_image/screenshot_webapp.png?raw=true)

# Task 3: Demonstration and discussion

### How my solution works to generate an image from text, telling it to a client with no technical ML experience.

The solution uses the state-of-the-art text to image AI model named Stable Diffusion v1.4. 

Stable Diffusion is a tool that can generate high-quality images from written descriptions. Think of it like a machine that can read your description of an object, like "a red car parked on a street," and create an image of a red car parked on a street.

To do this, Stable Diffusion has different parts that help it to create these images: 

1. The Text Encoder takes your description and changes it into a set of numbers that the machine can understand. The Latent Vector Generator takes those numbers and creates other numbers that represent different parts of the image, like the color, texture, and style.

2. Stable Diffusion then uses what's called a Diffusion Process to add noise (randomness) to these numbers to create an image that is similar to what you described. After that, the Decoder Network takes this noise-filled image and makes it clearer and more detailed, like a computer sharpening a blurry photo.

Below is an illustration of the diffusion process forward and backward pass:
![alt text](https://github.com/hoangkimthuc/diffusers/blob/main/examples/text_to_image/diffusion_illustration.png?raw=true)

3. Finally, the Discriminator Network looks at this final image and decides if it looks like what you described or if it's not quite right. This helps the machine learn how to make better images based on your descriptions in the future.

All of these parts work together to make Stable Diffusion a powerful tool for creating images from descriptions. By inputting your own descriptions, you can create custom images for various uses like marketing, advertising, and content creation.

There are a number of challenges that make training Stable Diffusion difficult, requiring deep theoretical understading of the model, resources, and good software engineering skills:


1. Large Amounts of Data: Stable diffusion models require large amounts of high-quality data, including text descriptions and corresponding images, to train effectively. Collecting and cleaning this data can be time-consuming and expensive.

2. Complex Architecture: Stable diffusion models have a complex architecture that includes many interconnected components. Optimizing the training of these components can be challenging and requires expertise in deep learning.

3. Computationally Intensive Training: Stable diffusion models require intensive training that can take several days or even weeks, depending on the size of the dataset and the complexity of the model. This requires access to high-end hardware and specialized computing resources.

4. Tuning Hyperparameters: Stable diffusion models have several hyperparameters that need to be tuned to optimize the performance of the model. Choosing the right hyperparameters can be difficult and requires expertise in machine learning.

5. Overfitting: Stable diffusion models can be prone to overfitting, which means that the model becomes too specialized to the training data and does not generalize well to new data. Regularization techniques, such as early stopping and dropout, can be used to combat overfitting.

### The client wants to move it to production and consume the model from multiple devices. Below is my proposed infra and the considerations while deploying the model.

### Some observations:

1. On CPUs, it takes around 260s to generate an image while it takes only around 4s on GPU. GPU RAM usage is 4-6GB. This is because Stable diffusion run the denoising process at inference time for 50-100 steps. So to get a timely response, a GPU-powered compute GPU is a must.

2. The models are complex and heavy (5GB), so it's not possible to deploy them on edge device like smartphones. Also, currently AI applications can't run directly on smartphone GPUs because of the lack of the AI software stack support.

### Proposed infra

Cloud provider: AWS cloud services

1. Amazon Elastic Container Service (ECS): Once the model is trained, we can create a docker container image of the stable diffusion model. We can use either Amazon ECS or Amazon EKS to launch and manage the docker container. ECS provides a scalable and cost-effective way to manage the container running the model and offers high availability and redundancy to ensure that the model is always up and running.
Specs of ECS: Configure the compute to be an the g3s.xlarge EC2 instance with 8GB GPU RAM seems to be reasonable. 


2. Application Programming Interface (API) Gateway: To access our Stable Diffusion model, we can use the AWS API Gateway service. It can be used to create and manage a RESTful API that can handle multiple requests from different devices. This service also has built-in caching and throttling features to ensure that the system can handle high volumes of requests without suffering from performance issues.

3. AWS CloudFront: AWS CloudFront is a globally distributed content delivery network that can be used to deliver the response from our API Gateway to our end-users around the world with low-latency connectivity. It caches the response so that it can be served quickly to subsequent requests.

4. Security Group and Virtual Private Cloud (VPC): To secure our Stable Diffusion model, we can make use of AWS VPC. It enables us to create a secure and isolated virtual network to deploy our model. We can also use a Security Group to allow traffic only from trusted sources.

### Tools and framework
Infra as code (cdk), Dev container, Pytest, and Git Actions for CI-CD. 
Development practice: Test Driven Development (TDD).
