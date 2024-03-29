# coding=utf-8
#
# Copyright 2024 Toshihiko Aoki
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os.path

import torch
from diffusers import AutoPipelineForText2Image, LCMScheduler
from llama_cpp import Llama
import gradio as gr

parser = argparse.ArgumentParser(description='Japanese translation and hallucinations for SD')
parser.add_argument('--gguf_path',
                    type=str,
                    default=None,
                    help='load gguf filepath')
parser.add_argument('--sd_model_name',
                    type=str,
                    default="segmind/SSD-1B",
                    help='sd model HF name')
parser.add_argument('--sd_adapter_name',
                    type=str,
                    default="latent-consistency/lcm-lora-ssd-1b",
                    help='sd lora adaptor HF name')
parser.add_argument('--cpu',
                    action='store_true',
                    help='force use cpu (intel).')
parser.add_argument('--share',
                    action='store_true',
                    help='force use cpu.')
parser.add_argument('--openvino_path',
                    type=str,
                    default=None,
                    help='load openvio model filepath')

args = parser.parse_args()

llm_model_path = args.gguf_path
sd_model_name = args.sd_model_name
sd_adapter_name = args.sd_adapter_name

use_cuda = torch.cuda.is_available() and not args.cpu

width = 512
height = 512
num_inference_steps = 4
guidance_scale = 1.0

if args.cpu:
    if args.openvino_path is None:
        if not os.path.exists('./sd-1.5-lcm-openvino'):
            from huggingface_hub import snapshot_download
            download_folder = snapshot_download(repo_id="Intel/sd-1.5-lcm-openvino")
            import shutil
            shutil.copytree(download_folder, "./sd-1.5-lcm-openvino'")
            args.openvino_path = './sd-1.5-lcm-openvino'
        else:
            args.openvino_path = './sd-1.5-lcm-openvino'
    from openvino_pipe import LatentConsistencyEngine
    pipe = LatentConsistencyEngine(
        args.openvino_path
    )
else:
    pipe = AutoPipelineForText2Image.from_pretrained(
        sd_model_name,
        torch_dtype=torch.float16 if use_cuda else torch.float32,
    )
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

    if use_cuda:
        pipe.enable_xformers_memory_efficient_attention()
        pipe.to("cuda")

    if sd_adapter_name is not None:
        pipe.load_lora_weights(sd_adapter_name)
        if use_cuda:
            pipe.fuse_lora()

if llm_model_path is None:
    from huggingface_hub import hf_hub_download
    llm_model_path = hf_hub_download(
        repo_id="taoki/llm-jp-1.3b-v1.0-staircaptions-FT",
        filename="llm-jp-1.3b-v1.0_staircaptions-FT_Q4_K_M.gguf",
    )

llm = Llama(
    model_path=llm_model_path,
    n_gpu_layers=25 if use_cuda else -1,
)

def ja2prompt(ja_prompt):
    response = llm(f"### Instruction:\n{ja_prompt}\n### Response:\n", max_tokens=128)
    return response['choices'][0]['text']


def prompt2img(sd_prompt):
    return pipe(
        sd_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=1.0,
    ).images[0]


with gr.Blocks(title="tiny sd web-ui") as demo:
    gr.Markdown(f"## Japanese translation and hallucinations for Stable Diffusion")
    with gr.Row():
        with gr.Column(scale=3):
            ja = gr.Text(label="日本語")
            translate = gr.Button("変換")
            prompt = gr.Text(label="プロンプト")
        with gr.Column(scale=2):
            result = gr.Image()
            t2i = gr.Button("生成")
    translate.click(ja2prompt, ja, prompt)
    t2i.click(prompt2img, prompt, result)

if args.share:
    demo.launch(share=True, server_name="0.0.0.0")
else:
    demo.launch()
