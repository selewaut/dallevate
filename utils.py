import requests
import openai
import os
import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import datetime


openai.api_key = os.getenv("OPENAI_API_KEY")

client = openai.OpenAI()
base_image_dir = os.path.join("images", "01_generations")
original_prompt = "two dogs playing chess, oil painting"
# regenarete multiple prompts from the original prompt.


def generate_query_variations_gpt(original_prompt, n_variations=1):
    print("Enriching prompt")
    template = f"""Generate {n_variations} prompts from this original prompt: {original_prompt}. This will be used to query a genai image generation model. 
            Generate prompt variations to generate multiple images with the same concept simple prompt. Try to be as descriptive as possible. 
            Remember to split each variation with a '\n' new line character to be easily parsable. Output in a json format that can be parsed easily, each prompt should be a key value pair."""
    # generate variations of the prompt using the OpenAI API GPT 4
    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        max_tokens=4096,
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

    print("Querying DALL-E model")

    # query the DALL-E model with the multiple variations of the prompt
    # crop response dict if there are more variations than n_variations
    image_responses = []
    for prompt_nbr, variation in response_dict_gpt.items():
        print(variation)
        try:
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
        except openai.BadRequestError as e:
            continue

    return image_responses


def process_dalle_images(image_list, filename, image_dir, original_prompt):
    print("Processing images")
    # save the images
    # generate image_dir to be day, time plus image_dir
    time_day = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    image_dir = os.path.join(image_dir, time_day)
    # craete the directory
    os.makedirs(image_dir, exist_ok=True)

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
    for image, filepath in zip(images, filepaths):
        if image is not None:  # loop through the variations
            with open(filepath, "wb") as image_file:  # open the file
                image_file.write(image)  # write the image to the file
    # save prompt to a text file inside same folder.
    with open(os.path.join(image_dir, "prompts.txt"), "w") as prompt_file:
        prompt_file.write(original_prompt)
        # then split by newline, and write each variation to the file
        prompt_file.write("\n".join([query[1] for query in image_list]))
    return filepaths


def get_content(url):
    try:
        content = requests.get(url=url).content
    except requests.exceptions.RequestException as e:
        content = None
    return content
