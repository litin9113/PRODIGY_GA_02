from diffusers import StableDiffusionPipeline
import torch

# Load a more powerful pre-trained model
pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1")
pipe.to("cpu")

# Get user input for the prompt and filename
prompt = input("Enter your image prompt: ")
filename = input("Enter the filename to save the image (e.g., 'output.png'): ")

# Generate an image with higher inference steps and guidance scale
image = pipe(prompt, num_inference_steps=100, guidance_scale=7.5).images[0]

# Save the image with the specified filename
image.save(filename)

print(f"Image saved as {filename}")