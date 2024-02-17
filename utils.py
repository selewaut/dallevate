import requests
import openai
import os
import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

openai.api_key = os.getenv("OPENAI_API_KEY")

client = openai.OpenAI()
base_image_dir = os.path.join("images", "01_generations")
original_prompt = "two dogs playing chess, oil painting"
n_variations = 4
# regenarete multiple prompts from the original prompt.


def generate_query_variations_gpt(original_prompt, n_variations=4):

    template = f"""Generate {n_variations} prompts from this original prompt: {original_prompt}. This will be used to query a genai image generation model. 
            Generate funny variations, add different types of styling to the image, such as realistic, comical, animated, and coloring styles. 
            Remember to split each variation with a '\n' new line character to be easily parsable. Output in a json format that can be parsed easily, each prompt should be a key value pair."""
    # generate variations of the prompt using the OpenAI API GPT 4
    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        max_tokens=2000,
        temperature=0.9,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that can generate creative variations of prompts for image generation using stable difussion. Rembmer to not violate any content poliy restrictions. Dont generate harmful, bad content",
            },
            {"role": "assistant", "content": f"{template}"},
        ],
        response_format={"type": "json_object"},
    )
    # parse the response and convert to dict

    query_variations = response.choices[0].message.content

    query_variations = json.loads(query_variations)

    return query_variations


def query_multiple_variations_dalle(response_dict_gpt):

    # query the DALL-E model with the multiple variations of the prompt
    image_responses = []
    for prompt_nbr, variation in response_dict_gpt.items():
        print(variation)
        image_resp = client.images.generate(
            model="dall-e-3",
            prompt=variation,
            size="1024x1792",
            quality="standard",
            style="vivid",
            response_format="url",
            n=1,
        )
        image_responses.append((image_resp, variation))

    return image_responses


def process_dalle_images(image_list, filename, image_dir):
    # save the images

    # urls list
    images = [query[0] for query in image_list]
    urls = [image.data[0].url for image in images]
    images = [requests.get(url=url).content for url in urls]  # download images
    image_names = [
        f"{filename}_{i + 1}.png" for i in range(len(images))
    ]  # create names
    filepaths = [
        os.path.join(image_dir, name) for name in image_names
    ]  # create filepaths
    for image, filepath in zip(images, filepaths):  # loop through the variations
        with open(filepath, "wb") as image_file:  # open the file
            image_file.write(image)  # write the image to the file

    return filepaths
