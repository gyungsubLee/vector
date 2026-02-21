import matplotlib.pyplot as plt
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity
import torch
import numpy as np
from PIL import Image
from typing import Tuple, List
import base64
import requests
import os
import openai
import re
import json

os.environ['OPENAI_API_KEY']= "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"
openai.api_key = os.environ["OPENAI_API_KEY"]


def fetch_clip() -> Tuple[CLIPModel, CLIPProcessor]:
    """
    CLIP model과 processor

    Returns:
        Tuple[CLIPModel, CLIPProcessor]: A tuple containing the CLIP model and processor.
    """
    model_name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)

    return model, processor

def extract_img_features(PIL_image: Image.Image, processor: CLIPProcessor, model: CLIPModel) -> torch.Tensor:
    """
    CLIP 모델을 활용하여 image embedding vector 추출

    Args:
        PIL_image (Image.Image): The image to process.
        processor (CLIPProcessor): The CLIP processor.
        model (CLIPModel): The CLIP model.

    Returns:
        torch.Tensor: The extracted image features.
    """
    inputs = processor(images=PIL_image, return_tensors="pt") # pytorch format

    # CLIPModel에서는 get_image_features가 Tensor를 반환하지만,
    # 일부 환경/모델 객체에서는 ModelOutput이 반환될 수 있어 안전하게 처리한다.
    if hasattr(model, "get_image_features"):
        outputs = model.get_image_features(**inputs)
    else:
        outputs = model(**inputs)

    if isinstance(outputs, torch.Tensor):
        return outputs.detach()

    if hasattr(outputs, "image_embeds") and outputs.image_embeds is not None:
        return outputs.image_embeds.detach()

    if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
        return outputs.pooler_output.detach()

    if isinstance(outputs, (tuple, list)) and len(outputs) > 0 and isinstance(outputs[0], torch.Tensor):
        return outputs[0].detach()

    raise TypeError(f"Unsupported model output type: {type(outputs)}")

def search_image(query_feature: torch.Tensor, features: List[torch.Tensor], topk: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    주어진 vector들과 비교하여, query_feature와 유사한 vector의 index와 유사도를 제공함

    Args:
        query_feature (torch.Tensor): query image의 embedding vector
        features (List[torch.Tensor]): 이미지들의 embedding vector
        topk (int, optional): top-k

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: 유사한 이미지들의 index & cosine-distance
    """
    similarities = cosine_similarity(query_feature, torch.vstack(features)).flatten()
    # sort in descending order
    sorted_indices_desc = similarities.argsort()[::-1]
    topk_indices = sorted_indices_desc[0:topk]
    # cosine similarities for the top-k indices
    topk_similarities = similarities[topk_indices]

    return topk_indices, topk_similarities

def draw_images(images: List[Image.Image], texts: List[str]=None):
    """
    이미지 show

    Args:
        images (List[Image.Image]): 이미지 list
        texts (List[str], optional): 이미지 아래에 display 할 텍스트. Defaults to None.
    """
    k = len(images)
    if texts is None:
       texts = ['']*k
    # Set up the figure and axes for a 1x5 grid
    fig, axs = plt.subplots(1, k, figsize=(20, 4))

    for i in range(k):
        axs[i].imshow(images[i])
        axs[i].axis('off')
        axs[i].text(0.5, -0.1, texts[i], va='bottom', ha='center', fontsize=10, transform=axs[i].transAxes)

    plt.show()


def tensor2np(tensor) -> np.ndarray:
    """
    Converts a PyTorch tensor to a NumPy array.

    Args:
        tensor (torch.Tensor): The PyTorch tensor to convert.

    Returns:
        np.ndarray: The converted NumPy array.
    """
    if tensor.is_cuda:
      numpy_array = tensor.cpu().numpy()
    else:
      numpy_array = tensor.numpy()

    return numpy_array

def detect_objects(img_path: str, model) -> Tuple[object, dict]:
    """
    이미지를 읽어오고, object detection 결과를 제공함

    Args:
        img_path (str): The path to the image file.
        model: The object detection model to use (ultralytics YOLO).

    Returns:
        Tuple[object, dict]: The raw results from the model and a dictionary containing detected boxes, scores,
                             categories, and labels.
    """
    img = Image.open(img_path)
    results = model(img, imgsz=1280, augment=True)

    pred_dict = dict()
    boxes = results[0].boxes

    pred_dict['boxes'] = boxes.xyxy.cpu().numpy()       # x1, y1, x2, y2
    pred_dict['scores'] = boxes.conf.cpu().numpy()
    pred_dict['categories'] = boxes.cls.cpu().numpy()

    categories = results[0].names
    pred_dict['labels'] = [categories[int(i)] for i in pred_dict['categories']]

    return results, pred_dict

def crop_bbox(pil_image: Image.Image, bbox: Tuple[float, float, float, float]) -> Image.Image:
    """
    Crops a bounding box from an image.

    Args:
        pil_image (Image.Image): The PIL Image to crop.
        bbox (Tuple[float, float, float, float]): The bounding box coordinates (x_min, y_min, x_max, y_max).

    Returns:
        Image.Image: The cropped image.
    """
    x_min, y_min, x_max, y_max = bbox
    crop_box = (x_min, y_min, x_max, y_max)

    cropped_image = pil_image.crop(crop_box)

    return cropped_image

def normalize_image(pil_image: Image.Image, target_size: Tuple[int, int] = (224, 224)) -> Image.Image:
    """
    Normalizes an image by resizing it to a target size and scaling pixel values to [0, 1].

    Args:
        pil_image (Image.Image): The PIL Image to normalize.
        target_size (Tuple[int, int], optional): The target size (width, height). Defaults to (224, 224).

    Returns:
        Image.Image: The normalized image.
    """
    # resizing
    resized_image = pil_image.resize(target_size, Image.LANCZOS)

    # normalization
    np_image = np.array(resized_image).astype('float32')
    np_image /= 255.0  # pixel values to [0, 1]
    normalized_image = Image.fromarray((np_image * 255).astype('uint8'))
    return normalized_image


def filter_furniture(detections: Tuple[object, dict]) -> dict:
    """
    Detect 된 object 중 가구들만 선택하고 score>0.5 이상인 detection들만 선별함

    Args:
        detections (Tuple[object, dict]): YoloV5 output

    Returns:
        dict: A dictionary containing the filtered detections of furniture items.
    """

    furniture_class = [56, 57, 59, 60] # detections[0].names
    furniture_names = ['chair', 'couch', 'bed', 'dining table']
    furniture_detected = {}

    filter = [True if (i in furniture_names) and (s>0.5) else False for i, s in zip(detections[1]['labels'], detections[1]['scores'])]
    furniture_detected['boxes'] = detections[1]['boxes'][filter]
    furniture_detected['scores'] = detections[1]['scores'][filter]
    furniture_detected['categories'] = detections[1]['categories'][filter]
    furniture_detected['lables'] = [item for item, bool in zip(detections[1]['labels'], filter) if bool==True]
    
    return furniture_detected

def encode_image(image_path: str) -> str:
    """
    Encodes an image file to a base64 string.

    Args:
        image_path (str): The file path of the image to encode.

    Returns:
        str: The base64 encoded string of the image.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
  

def describe_image(image_paths: List[str]) -> str:
    """
    GPT-4V를 활용하여 거실의 이미지를 4가지 카테고리를 기준으로 설명함.

    Args:
        image_paths (List[str]): List of paths to the images to be analyzed.

    Returns:
        str: A JSON-like string containing descriptions for each image focusing on 'Color Scheme', 'Lighting',
             'Spatial Layout', and 'Architectural Features'.
    """

    text_prompt = """Please analyze the living room image provided.  
Include 'Color Scheme', 'Lighting', 'Spatial Layout', and 'Architectural Features' with descriptions based on the room's characteristics.
The output should be formatted in a JSON-like dictionary structure. Each image should be done separately.

Example output :

```json
{
"Color Scheme": <Description about color scheme>,
"Lighting": <Description about lighting>,
"Spatial Layout": <Description about spatial layouts >,
"Architectural Features": <Descrption about architectural features>
}
```
    """

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai.api_key}"
        }
    imgs = [encode_image(i) for i in image_paths]

    payload = {
            "model": "gpt-4-vision-preview",
            "messages": [{"role": "user",
                        "content": []
                        },
                        ],
            "max_tokens": 1000
            }

    img_contents = [{"type": "text", "text": text_prompt}]
    for img in imgs:
        input_template = {
            "type": "image_url",
            "image_url": {
            "url": f"data:image/jpeg;base64,{img}"
            }
        }
        img_contents.append(input_template)

    payload['messages'][0]['content'] = img_contents

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    output = response.json()['choices'][0]['message']['content']
    return output

def parse_response(text: str) -> List[dict]:
    """
    GPT-4V로부터 얻은 image description을 json 형태로 변환

    Args:
        text (str): The JSON-like string to parse.

    Returns:
        List[dict]: A list of dictionaries parsed from the input string.

    The function uses regular expressions to find JSON objects in the response and parses them into Python
    dictionaries.
    """

    matches = re.findall(r'(\{[\s\S]*?\})', text)
    matches = [json.loads(m) for m in matches]
    return matches

def reformat_img_description(input_content: str, model: str = 'gpt-4-turbo-preview') -> dict:
    """
    parse_response를 활용하여 json의 형태로 변환이 안되는 경우, gpt-4-turbo를 활용하여 정형화된 포멧으로 변환

    Args:
        input_content (str): The unstructured text containing image descriptions.
        model (str, optional): The model to use for reformatting. Defaults to 'gpt-4-turbo-preview'.

    Returns:
        dict: A dictionary containing the reformatted image descriptions.
    """
    output_formatting_prompt = """Using the provided text, find the smallest format of json there is and store them in a list as separate elements.
The ouput list should have two json objects found from the provided text.

Desired output :
{'list': [{'Image 1': {'Color Scheme': <Color Scheme>,
    'Lighting': <Lighting>,
    'Spatial Layout': <Spatial Layout>,
    'Architectural Features': <Architectural Features>}},
  {'Image 2': {'Color Scheme': <Color Scheme>,
    'Lighting': <Lighting>,
    'Spatial Layout': <Spatial Layout>,
    'Architectural Features': <Architectural Features>}}]}

Provided text : """

    client = openai.OpenAI()

    response = client.chat.completions.create(
        model=model,
        response_format={ "type": "json_object" },
        messages=[
            {"role": "system", "content": 'You are a smart and intelligent program that understands information and provides output in JSON format'},
            {"role": "user", "content":output_formatting_prompt + input_content}
        ]
        )
    return response
