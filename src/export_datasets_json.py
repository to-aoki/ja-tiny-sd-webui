import os
import json
from collections import OrderedDict
from tqdm import tqdm
from datasets import Dataset
from PIL import Image
import torch

from uform.gen_model import VLMForCausalLM, VLMProcessor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VLMForCausalLM.from_pretrained("unum-cloud/uform-gen").to(device)
processor = VLMProcessor.from_pretrained("unum-cloud/uform-gen")


def sd_prompt(image_path, captions=''):
    # 頭に元のcaptionをつけ後続を促す
    prompt = "[cap] Exhaustively details the visual content of the image." + captions
    image = Image.open(image_path)

    inputs = processor(texts=[prompt], images=[image], return_tensors="pt").to(device)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            do_sample=False,
            use_cache=True,
            max_new_tokens=64,
            eos_token_id=32001,
            pad_token_id=processor.tokenizer.pad_token_id
        )

    prompt_len = inputs["input_ids"].shape[1]
    captions = captions.rstrip()
    if not captions.endswith('.'):
        captions = captions + '.'
    result = captions + processor.batch_decode(output[:, prompt_len:], skip_special_tokens=True)[0]
    result = result.replace('.', ',')
    elements = [element.strip() for element in result.split(',')]
    concat_list = []
    for i, element in enumerate(elements):
        if i == 0:
            # オリジナルはそのまま入れる
            concat_list.append(element)
            continue
        # 「この画像は...」についての言及は消す
        element = element.lower().replace('a image', "").replace("the image", "")
        words = list(OrderedDict.fromkeys([word.strip() for word in element.split(' ')
                                           if word.lower() not in ['a', 'the']]))
        concat_list.append(' '.join(words))
    result = ",".join(concat_list)
    last_comma_index = result.rfind(',')
    if last_comma_index != -1:
        return result[:last_comma_index]
    else:
        return result


def export_json(stair_captions_json, coco_anotations_json, coco_images_dir, output_json):
    with open(stair_captions_json, 'r') as f:
        stair = json.load(f)
    with open(coco_anotations_json, 'r') as f:
        coco = json.load(f)

    stair_captions_by_id = {}
    coco_captions_by_id = {}

    dataset_dict = {
        "id": [],
        "file_name": [],
        "caption": [],
        "prompt": []
    }

    for annotation in stair['annotations']:
        stair_captions_by_id[annotation['image_id']] = annotation
    for annotation in coco['annotations']:
        coco_captions_by_id[annotation['image_id']] = annotation

    for image in tqdm(coco['images']):
        caption = stair_captions_by_id.get(image['id'], None)
        if caption is None:
            continue
        coco_caption = coco_captions_by_id.get(image['id'], None)
        if coco_caption is None:
            continue

        dataset_dict["id"].append(image['id'])
        dataset_dict["file_name"].append(image['file_name'])
        dataset_dict["caption"].append(caption['caption'])
        prompt = sd_prompt(os.path.join(coco_images_dir, image['file_name']), coco_caption['caption'])
        dataset_dict["prompt"].append(prompt)

    stair_captions_dataset = Dataset.from_dict(dataset_dict)
    stair_captions_dataset.to_json(output_json)


stair_captions_jsons = ["../data/stair_captions_v1.2/stair_captions_v1.2_train.json",
                        "../data/stair_captions_v1.2/stair_captions_v1.2_val.json"]
coco_annotations_jsons = ["../data/annotations/captions_train2014.json",
                          "../data/annotations/captions_val2014.json"]
coco_images_dirs = ["../data/train2014/",
                    "../data/val2014/"]
output_jsons = ["stair_captions_v1.2_train_with_prompt.json",
                "stair_captions_v1.2_val_with_prompt.json"]

for stair_path, coco_path, image_path, output_path in zip(
        stair_captions_jsons, coco_annotations_jsons, coco_images_dirs, output_jsons):
    export_json(stair_path, coco_path, image_path, output_path)
