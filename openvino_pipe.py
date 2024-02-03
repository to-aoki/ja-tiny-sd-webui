# origin: https://github.com/intel/openvino-ai-plugins-gimp/blob/ae93e7291fab6d372c958da18e497acb9d927055/gimpopenvino/tools/openvino_common/models_ov/stable_diffusion_engine.py#L748

import os
from typing import Union, Optional, Any, List, Dict

import torch
from openvino.runtime import Core
from diffusers import DiffusionPipeline, LCMScheduler, ImagePipelineOutput
from diffusers.image_processor import VaeImageProcessor
from transformers import CLIPTokenizer


class LatentConsistencyEngine(DiffusionPipeline):
    def __init__(
        self,
            model="SimianLuo/LCM_Dreamshaper_v7",
            tokenizer="openai/clip-vit-large-patch14",
            device=["CPU", "CPU", "CPU"],
    ):
        super().__init__()
        try:
            self.tokenizer = CLIPTokenizer.from_pretrained(model, local_files_only=True)
        except:
            self.tokenizer = CLIPTokenizer.from_pretrained(tokenizer)
            self.tokenizer.save_pretrained(model)

        self.core = Core()
        self.core.set_property({'CACHE_DIR': os.path.join(model, 'cache')})  # adding caching to reduce init time
        # text features

        print("Text Device:", device[0])
        self.text_encoder = self.core.compile_model(os.path.join(model, "text_encoder.xml"), device[0])
        self._text_encoder_output = self.text_encoder.output(0)

        # diffusion
        print("unet Device:", device[1])
        self.unet = self.core.compile_model(os.path.join(model, "unet.xml"), device[1])
        self._unet_output = self.unet.output(0)
        self.infer_request = self.unet.create_infer_request()

        # decoder
        print("Vae Device:", device[2])

        self.vae_decoder = self.core.compile_model(os.path.join(model, "vae_decoder.xml"), device[2])
        self.infer_request_vae = self.vae_decoder.create_infer_request()
        self.safety_checker = None #pipe.safety_checker
        self.feature_extractor = None #pipe.feature_extractor
        self.vae_scale_factor = 2 ** 3
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.scheduler =  LCMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear"
        )

    def _encode_prompt(
        self,
        prompt,
        num_images_per_prompt,
        prompt_embeds: None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.
        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
        """

        if prompt_embeds is None:

            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(
                prompt, padding="longest", return_tensors="pt"
            ).input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[
                -1
            ] and not torch.equal(text_input_ids, untruncated_ids):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                )

            prompt_embeds = self.text_encoder(text_input_ids, share_inputs=True, share_outputs=True)
            prompt_embeds = torch.from_numpy(prompt_embeds[0])

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(
            bs_embed * num_images_per_prompt, seq_len, -1
        )

        # Don't need to get uncond prompt embedding because of LCM Guided Distillation
        return prompt_embeds

    def run_safety_checker(self, image, dtype):
        if self.safety_checker is None:
            has_nsfw_concept = None
        else:
            if torch.is_tensor(image):
                feature_extractor_input = self.image_processor.postprocess(
                    image, output_type="pil"
                )
            else:
                feature_extractor_input = self.image_processor.numpy_to_pil(image)
            safety_checker_input = self.feature_extractor(
                feature_extractor_input, return_tensors="pt"
            )
            image, has_nsfw_concept = self.safety_checker(
                images=image, clip_input=safety_checker_input.pixel_values.to(dtype)
            )
        return image, has_nsfw_concept

    def prepare_latents(
        self, batch_size, num_channels_latents, height, width, dtype, latents=None
    ):
        shape = (
            batch_size,
            num_channels_latents,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        if latents is None:
            latents = torch.randn(shape, dtype=dtype)
        # scale the initial noise by the standard deviation required by the scheduler
        return latents

    def get_w_embedding(self, w, embedding_dim=512, dtype=torch.float32):
        """
        see https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298
        Args:
        timesteps: torch.Tensor: generate embedding vectors at these timesteps
        embedding_dim: int: dimension of the embeddings to generate
        dtype: data type of the generated embeddings
        Returns:
        embedding vectors with shape `(len(timesteps), embedding_dim)`
        """
        assert len(w.shape) == 1
        w = w * 1000.0

        half_dim = embedding_dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
        emb = w.to(dtype)[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:  # zero pad
            emb = torch.nn.functional.pad(emb, (0, 1))
        assert emb.shape == (w.shape[0], embedding_dim)
        return emb

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        guidance_scale: float = 7.5,
        scheduler = None,
        num_images_per_prompt: Optional[int] = 1,
        latents: Optional[torch.FloatTensor] = None,
        num_inference_steps: int = 4,
        lcm_origin_steps: int = 50,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        model: Optional[Dict[str, any]] = None,
        seed: Optional[int] = 1234567,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback = None,
        callback_userdata = None
    ):

        # 1. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if seed is not None:
            torch.manual_seed(seed)

        #print("After Step 1: batch size is ", batch_size)
        # do_classifier_free_guidance = guidance_scale > 0.0
        # In LCM Implementation:  cfg_noise = noise_cond + cfg_scale * (noise_cond - noise_uncond) , (cfg_scale > 0.0 using CFG)

        # 2. Encode input prompt
        prompt_embeds = self._encode_prompt(
            prompt,
            num_images_per_prompt,
            prompt_embeds=prompt_embeds,
        )
        #print("After Step 2: prompt embeds is ", prompt_embeds)
        #print("After Step 2: scheduler is ", scheduler )
        # 3. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, original_inference_steps=lcm_origin_steps)
        timesteps = self.scheduler.timesteps

        #print("After Step 3: timesteps is ", timesteps)

        # 4. Prepare latent variable
        num_channels_latents = 4
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            latents,
        )
        latents = latents * self.scheduler.init_noise_sigma

        #print("After Step 4: ")
        bs = batch_size * num_images_per_prompt

        # 5. Get Guidance Scale Embedding
        w = torch.tensor(guidance_scale).repeat(bs)
        w_embedding = self.get_w_embedding(w, embedding_dim=256)
        #print("After Step 5: ")
        # 6. LCM MultiStep Sampling Loop:
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if callback:
                    callback(i+1, callback_userdata)

                ts = torch.full((bs,), t, dtype=torch.long)

                # model prediction (v-prediction, eps, x)
                model_pred = self.unet([latents, ts, prompt_embeds, w_embedding],share_inputs=True, share_outputs=True)[0]

                # compute the previous noisy sample x_t -> x_t-1
                latents, denoised = self.scheduler.step(
                    torch.from_numpy(model_pred), t, latents, return_dict=False
                )
                progress_bar.update()

        #print("After Step 6: ")

        #vae_start = time.time()

        if not output_type == "latent":
            image = torch.from_numpy(self.vae_decoder(denoised / 0.18215, share_inputs=True, share_outputs=True)[0])
        else:
            image = denoised

        #print("vae decoder done", time.time() - vae_start)
        #post_start = time.time()

        #if has_nsfw_concept is None:
        do_denormalize = [True] * image.shape[0]
        #else:
        #    do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        #print ("After do_denormalize: image is ", image)

        image = self.image_processor.postprocess(
            image, output_type=output_type, do_denormalize=do_denormalize
        )

        return ImagePipelineOutput([image[0]])