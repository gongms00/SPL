"""
Generate prompts with OpenAI GPT-4V for image harmonization.
Generates captions based on composite and mask images.

Usage:
    python harmonization_gpt.py --images_root images/harmonization/composite --masks_root images/harmonization/mask

Requires OPENAI_API_KEY environment variable to be set.
"""
from openai import OpenAI
import glob
import os
import base64
from pydantic import BaseModel
import argparse

parser = argparse.ArgumentParser(description='Generate captions for composite images.')
parser.add_argument('--images_root', type=str, default='images/harmonization/composite', help='Path to the composite images.')
parser.add_argument('--masks_root', type=str, default='images/harmonization/mask', help='Path to the mask images.')
parser.add_argument('--new_caption_path', type=str, default='images/harmonization/caption_3.txt', help='Path to save the caption.')
parser.add_argument('--prompt_num', type=int, default=1, help='How many different prompts for one image.')

args = parser.parse_args()
images_root = args.images_root
masks_root = args.masks_root
new_caption_path = args.new_caption_path
prompt_num = args.prompt_num

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set. Please set it before running this script.")
client = OpenAI(api_key=api_key)

class PromptSet(BaseModel):
    foreground_object: str
    foreground_description: str
    background_description: str

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_caption(image_path, mask_path) -> PromptSet:
    try:
        # Convert images to base64
        img_base64 = encode_image(image_path)
        mask_base64 = encode_image(mask_path)

        # Construct prompt
        prompt = (
            "I want to choose some words to describe the composite image, which is made by superimposing a cut-out object onto the background image. "
            "Two images will be provided: the composite image and the mask image. "
            "The foreground region is the mask region, while the rest constitutes the background.\n"
            "Here, I provide a set of descriptive words categorized in a dictionary as follows:\n"
            "{'brightness':[dazzling, bright, dim, dull, shaded, shadowed],\n"
            "'weather':[cloudy, sunny, rainy, snowy, foggy, windy, stormy, clear, misty],\n"
            "'temperature':[hot, warm, cool, cold, icy],\n"
            "'season':[spring, summer, autumn, winter],\n"
            "'time':[dawn, sunrise, daylight, twilight, sunset, dusk, dark, night],\n"
            "'color tone':[greyscale, neon, golden, white, blue, green, yellow, orange, red, earthy],\n"
            "'environment':[city, rural, lake, ocean, mountain, forest, desert, grassland, sky, space, indoor, street]}\n"
            "Now, I need to first give the name of the foreground object and then select appropriate words from the above dictionary to describe both the foreground object and background. Here are the specific outputs I expect: \n "
            "1. Foreground object : Describe the name of the foreground object\n"
            "2. Foreground description : Choose one or two words from the entire dictionary that best describe the style of foreground. (e.g. brightness, color tone...)\n"
            "3. Background description : Choose one or two words from the entire dictionary that best describe the background. (e.g. brightness, weather, temperature, season ...)\n"
            "Note: Choose only one word from each list and ensure that a word from the 'brightness' list is included in the selection.\n"
            "Note: The word for description MUST be chosen from the dictionary provided above. Otherwise, it will be rejected. \n"
            "Note: Foreground object should be a single word. \n"
            "For example, (Foreground object) = dog, (Foreground description) = bright summer, (Background description) = winter dull greyscale. \n "
        )

        # Call OpenAI GPT-4V
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user",
                 "content": [
                     {
                         "type": "image_url",
                         "image_url": {
                             "url": f"data:image/jpeg;base64,{img_base64}",
                             "detail": "low"
                         }
                     },
                     {
                         "type": "image_url",
                         "image_url": {
                             "url": f"data:image/jpeg;base64,{mask_base64}",
                             "detail": "low"
                         }
                     }
                 ]},
            ],
            response_format=PromptSet
        )

        return completion.choices[0].message.parsed
    except Exception as e:
        print(e)
        print("get caption failed, try again...")
        return None

composite_images = []
mask_images = []
for i in glob.glob(os.path.join(images_root, "*")):
    composite_images.append(i)
for i in glob.glob(os.path.join(masks_root, "*")):
    mask_images.append(i)
composite_images = sorted(composite_images)
mask_images = sorted(mask_images)

for i in range(0, len(composite_images)):
    image_path = composite_images[i]
    mask_path = mask_images[i]
    while True:
        caption_set = get_caption(image_path, mask_path)
        if caption_set is not None:
            break
    new_caption = "{} {},{} {}".format(caption_set.foreground_description, caption_set.foreground_object,
                                       caption_set.background_description, caption_set.foreground_object)
    print(f"caption for {os.path.basename(image_path)} : {new_caption}")
    with open(new_caption_path, 'a') as f:
        f.write(new_caption)
        f.write('\n')
