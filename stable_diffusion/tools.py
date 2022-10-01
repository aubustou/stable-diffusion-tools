import inspect
import logging
import math
import re
from io import BytesIO
from pathlib import Path
from pprint import pprint
from typing import Any, List, Optional, Union

import numpy as np
import requests
import torch
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DiffusionPipeline,
    PNDMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from IPython.display import display
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from torch import autocast
from tqdm.auto import tqdm
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

logger = logging.getLogger("stable_diffusion.tools")


class StableDiffusionImg2ImgPipeline(DiffusionPipeline):
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: Union[DDIMScheduler, PNDMScheduler],
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPFeatureExtractor,
    ):
        super().__init__()
        scheduler = scheduler.set_format("pt")
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        init_image: torch.FloatTensor,
        strength: float = 0.8,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 7.5,
        eta: Optional[float] = 0.0,
        generator: Optional[torch.Generator] = None,
        output_type: Optional[str] = "pil",
        latents: Any = None,
    ):

        if isinstance(prompt, str):
            batch_size = 1
        elif isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            raise ValueError(
                f"`prompt` has to be of type `str` or `list` but is {type(prompt)}"
            )

        if strength < 0 or strength > 1:
            raise ValueError(
                f"The value of strength should in [0.0, 1.0] but is {strength}"
            )

        # set timesteps
        accepts_offset = "offset" in set(
            inspect.signature(self.scheduler.set_timesteps).parameters.keys()
        )
        extra_set_kwargs = {}
        offset = 0
        if accepts_offset:
            offset = 1
            extra_set_kwargs["offset"] = 1

        self.scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)

        # encode the init image into latents and scale the latents
        init_latents = self.vae.encode(init_image.to(self.device)).sample()
        init_latents = 0.18215 * init_latents

        # prepare init_latents noise to latents
        init_latents = torch.cat([init_latents] * batch_size)

        # get the original timestep using init_timestep
        init_timestep = int(num_inference_steps * strength) + offset
        init_timestep = min(init_timestep, num_inference_steps)
        timesteps = self.scheduler.timesteps[-init_timestep]
        timesteps = torch.tensor(
            [timesteps] * batch_size, dtype=torch.long, device=self.device
        )

        # add noise to latents using the timesteps
        noise = torch.randn(init_latents.shape, generator=generator, device=self.device)
        init_latents = self.scheduler.add_noise(init_latents, noise, timesteps)

        # get prompt text embeddings
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0
        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            max_length = text_input.input_ids.shape[-1]
            uncond_input = self.tokenizer(
                [""] * batch_size,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )
            uncond_embeddings = self.text_encoder(
                uncond_input.input_ids.to(self.device)
            )[0]

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        latents = init_latents
        t_start = max(num_inference_steps - init_timestep + offset, 0)
        for i, t in tqdm(enumerate(self.scheduler.timesteps[t_start:])):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = (
                torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            )

            # predict the noise residual
            noise_pred = self.unet(
                latent_model_input, t, encoder_hidden_states=text_embeddings
            )["sample"]

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs)[
                "prev_sample"
            ]

        # scale and decode the image latents with vae
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents)

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()

        # run safety checker
        safety_cheker_input = self.feature_extractor(
            self.numpy_to_pil(image), return_tensors="pt"
        ).to(self.device)
        image, has_nsfw_concept = self.safety_checker(
            images=image, clip_input=safety_cheker_input.pixel_values
        )

        if output_type == "pil":
            image = self.numpy_to_pil(image)

        return {"sample": image, "nsfw_content_detected": has_nsfw_concept}


def preprocess(image):
    w, h = image.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)

    return 2.0 * image - 1.0


def get_pipe(
    model_path: str,
    device: str = "cuda",
    use_pndms_scheduler: bool = False,
    img2img: bool = False,
):
    if use_pndms_scheduler:
        # Using DDIMScheduler as an example,this also works with PNDMScheduler
        scheduler = PNDMScheduler.from_config(
            model_path, subfolder="scheduler", use_auth_token=True
        )
    else:
        scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
        )
    if img2img:
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_path,
            scheduler=scheduler,
            revision="fp16",
            torch_dtype=torch.float16,
            use_auth_token=True,
        )
    else:
        pipe = StableDiffusionPipeline.from_pretrained(
            model_path,
            scheduler=scheduler,
            revision="fp16",
            torch_dtype=torch.float16,
            use_auth_token=True,
        )

    pipe = pipe.to(device)
    pipe.safety_checker = fuck_safety_checker
    pipe.img2img = True

    return pipe


def fuck_safety_checker(images, **kwargs):
    return images, False


def image_grid(imgs, rows, cols):
    assert len(imgs) <= rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h), color=(255, 255, 255))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def get_latents(
    *,
    pipe,
    num_images: int = 1,
    user_seed: Optional[int] = None,
    device: str = "cuda",
    height: int = 512,
    width: int = 512,
) -> tuple[Optional[list[object]], list[int]]:
    generator = torch.Generator(device=device)

    latents = None
    seeds = []
    for _ in range(num_images):
        # Get a new random seed, store it and use it as the generator state
        seed = generator.seed()
        seeds.append(seed)
        generator = generator.manual_seed(seed)

        image_latents = torch.randn(
            (1, pipe.unet.in_channels, height // 8, width // 8),
            generator=generator,
            device=device,
        )
        latents = (
            image_latents if latents is None else torch.cat((latents, image_latents))
        )
    return latents, seeds


IMAGE_NAME_PATTERN = re.compile(r"(\d+)_(\d+)")


def truncate_name(name: str, length: int = 30) -> str:
    if len(name) > length:
        return name[:length]
    else:
        return name


def return_lists(n, *iterables):
    iter_zero = iterables[0]
    for index in range(0, len(iter_zero) + n - 1, n):
        if iter_zero[index : index + n]:
            yield tuple(iterable[index : index + n] for iterable in iterables)
        else:
            return


def get_base_image(
    img_url: Optional[str] = None,
    img_path: Optional[str] = None,
):
    if img_url:
        response = requests.get(img_url)
        content = response.content
    elif img_path:
        content = Path(img_path).read_bytes()
    else:
        raise ValueError
    init_img = Image.open(BytesIO(content)).convert("RGB")
    init_img = init_img.resize((512, 512))
    display(init_img)
    return preprocess(init_img)


def save_images(
    images: list[Image.Image],
    prompts: list[str],
    image_root_folder: Path,
    seeds: list[int],
    additional_metadata: Optional[dict[str, str]] = None,
):
    for index, image in enumerate(images):
        logger.info(f"Saving image {index + 1} of {len(images)}")

        prompt = prompts[index]
        str_seed = str(seeds[index])

        image_folder = image_root_folder / truncate_name(prompt)
        image_folder.mkdir(parents=True, exist_ok=True)

        image_path = image_folder / (str_seed + ".png")
        while image_path.exists():
            if match := IMAGE_NAME_PATTERN.match(image_path.stem):
                name, number = match.groups()
                number = int(number) + 1
            else:
                name = str_seed
                number = 1
            image_path = image_path.with_stem(f"{name}_{number}")

        metadata = PngInfo()
        metadata.add_text("compviz_seed", str_seed)
        metadata.add_text("compviz_prompt", prompt)
        for key, value in (additional_metadata or {}).items():
            metadata.add_text(key, value)

        image.save(str(image_path), pnginfo=metadata)


def get_images(
    prompts: Union[str, list[str]],
    pipe,
    generator,
    img_url: Optional[str] = None,
    img_path: Optional[str] = None,
    same_seed: bool = False,
    num_images: int = 1,
    seed: Optional[int] = None,
    num_inference_steps: int = 50,
    simultaneous_prompts: int = 6,
    image_root_folder: Path = Path("images"),
    device: str = "cuda",
    strength: float = 0.75,
    grid_rows: Optional[int] = None,
    grid_columns: Optional[int] = 3,
    do_not_show: bool = False,
    additional_metadata: Optional[dict[str, str]] = None,
):

    kwargs = {}
    if getattr(pipe, "img2img", False):
        kwargs = {"init_image": get_base_image(img_url, img_path)}

    if isinstance(prompts, str):
        prompts = [prompts]

    if len(prompts) == 1 and num_images > 1:
        prompts = prompts * num_images

    print(len(prompts))

    if grid_columns is None:
        if grid_rows is not None:
            grid_columns = math.ceil(len(prompts) / grid_rows)
        else:
            grid_columns = simultaneous_prompts
            grid_rows = math.ceil(len(prompts) / grid_columns)
    else:
        if grid_rows is not None:
            raise ValueError("Cannot set both grid_columns and grid_rows")
        else:
            grid_rows = math.ceil(len(prompts) / grid_columns)

    num_seeds = len(prompts)
    num_latents = len(prompts)

    if seed:
        latents = (None,)
        generator = generator.manual_seed(seed)
        seeds = [seed] * num_seeds
    else:
        latents, seeds = get_latents(pipe=pipe, num_images=num_latents, device=device)
        generator = None
    print(seeds)

    images = []
    with autocast(device):
        if same_seed:
            for prompt in prompts:
                generator = generator.manual_seed(seed)
                images.extend(
                    pipe(
                        prompt,
                        num_inference_steps=num_inference_steps,
                        generator=generator,
                        latents=latents,
                        strength=strength,
                        **kwargs,
                    )["sample"]
                )
        else:
            for sliced_prompts, sliced_latents in return_lists(
                simultaneous_prompts, prompts, latents
            ):
                pprint(sliced_prompts)
                images.extend(
                    pipe(
                        sliced_prompts,
                        num_inference_steps=num_inference_steps,
                        generator=generator,
                        latents=sliced_latents,
                        strength=strength,
                        **kwargs,
                    )["sample"]
                )

    save_images(images, prompts, image_root_folder, seeds, additional_metadata)

    if do_not_show:
        return
    else:
        return image_grid(images, rows=grid_rows, cols=grid_columns)
