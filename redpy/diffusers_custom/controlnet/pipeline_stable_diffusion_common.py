# Inspired by: https://github.com/haofanwang/ControlNet-for-Diffusers/

import inspect
from typing import Any, Callable, Dict, List, Optional, Union, Tuple

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
import re
import importlib
import os
import cv2
from PIL import Image
from torchvision.transforms import Resize
from IPython import embed

all = [
    'draw_kps',
    'StableDiffusionCommonPipeline',
]

LOADABLE_CLASSES = {
    "diffusers": {
        "ModelMixin": ["save_pretrained", "from_pretrained"],
        "SchedulerMixin": ["save_pretrained", "from_pretrained"],
        "DiffusionPipeline": ["save_pretrained", "from_pretrained"],
        "OnnxRuntimeModel": ["save_pretrained", "from_pretrained"],
    },
    "transformers": {
        "PreTrainedTokenizer": ["save_pretrained", "from_pretrained"],
        "PreTrainedTokenizerFast": ["save_pretrained", "from_pretrained"],
        "PreTrainedModel": ["save_pretrained", "from_pretrained"],
        "FeatureExtractionMixin": ["save_pretrained", "from_pretrained"],
        "ProcessorMixin": ["save_pretrained", "from_pretrained"],
        "ImageProcessingMixin": ["save_pretrained", "from_pretrained"],
    },
    "onnxruntime.training": {
        "ORTModule": ["save_pretrained", "from_pretrained"],
    },
}


from diffusers.models import (
    AutoencoderKL,
    ControlNetModel,
    UNet2DConditionModel,
)
from diffusers.pipelines import (
    DiffusionPipeline,
)
from diffusers.pipelines.stable_diffusion import (
    StableDiffusionPipelineOutput, 
    StableDiffusionSafetyChecker
)
from diffusers.schedulers import (
    KarrasDiffusionSchedulers
)
from diffusers.loaders import TextualInversionLoaderMixin, LoraLoaderMixin, FromCkptMixin
from diffusers.utils import (
    logging,
    PIL_INTERPOLATION,
    is_accelerate_available,
    is_accelerate_version,
    randn_tensor,
    replace_example_docstring,
    is_accelerate_available,
)

if is_accelerate_available():
    import accelerate





logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import numpy as np
        >>> import torch
        >>> from PIL import Image
        >>> from diffusers import ControlNetModel, UniPCMultistepScheduler
        >>> from diffusers.utils import load_image

        >>> input_image = load_image("https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png")

        >>> controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)

        >>> pipe_controlnet = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                controlnet=controlnet,
                safety_checker=None,
                torch_dtype=torch.float16
                )

        >>> pipe_controlnet.scheduler = UniPCMultistepScheduler.from_config(pipe_controlnet.scheduler.config)
        >>> pipe_controlnet.enable_xformers_memory_efficient_attention()
        >>> pipe_controlnet.enable_model_cpu_offload()

        # using image with edges for our canny controlnet
        >>> control_image = load_image(
            "https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/vermeer_canny_edged.png")


        >>> result_img = pipe_controlnet(controlnet_conditioning_image=control_image,
                        image=input_image,
                        prompt="an android robot, cyberpank, digitl art masterpiece",
                        num_inference_steps=20).images[0]

        >>> result_img.show()
        ```
"""


re_attention = re.compile(
    r"""
\\\(|
\\\)|
\\\[|
\\]|
\\\\|
\\|
\(|
\[|
:([+-]?[.\d]+)\)|
\)|
]|
[^\\()\[\]:]+|
:
""",
    re.X,
)


def prepare_image(image):
    if isinstance(image, torch.Tensor):
        # Batch single image
        if image.ndim == 3:
            image = image.unsqueeze(0)

        image = image.to(dtype=torch.float32)
    else:
        # preprocess image
        if isinstance(image, (PIL.Image.Image, np.ndarray)):
            image = [image]

        if isinstance(image, list) and isinstance(image[0], PIL.Image.Image):
            image = [np.array(i.convert("RGB"))[None, :] for i in image]
            image = np.concatenate(image, axis=0)
        elif isinstance(image, list) and isinstance(image[0], np.ndarray):
            image = np.concatenate([i[None, :] for i in image], axis=0)

        image = image.transpose(0, 3, 1, 2)
        image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

    return image


def prepare_controlnet_conditioning_image(
    controlnet_conditioning_image, width, height, batch_size, num_images_per_prompt, device, dtype
):
    if not isinstance(controlnet_conditioning_image, torch.Tensor):
        if isinstance(controlnet_conditioning_image, PIL.Image.Image):
            controlnet_conditioning_image = [controlnet_conditioning_image]

        if isinstance(controlnet_conditioning_image[0], PIL.Image.Image):
            controlnet_conditioning_image = [
                np.array(i.resize((width, height), resample=PIL_INTERPOLATION["lanczos"]))[None, :]
                for i in controlnet_conditioning_image
            ]
            controlnet_conditioning_image = np.concatenate(controlnet_conditioning_image, axis=0)
            controlnet_conditioning_image = np.array(controlnet_conditioning_image).astype(np.float32) / 255.0
            controlnet_conditioning_image = controlnet_conditioning_image.transpose(0, 3, 1, 2)
            controlnet_conditioning_image = torch.from_numpy(controlnet_conditioning_image)
        elif isinstance(controlnet_conditioning_image[0], torch.Tensor):
            controlnet_conditioning_image = torch.cat(controlnet_conditioning_image, dim=0)

    image_batch_size = controlnet_conditioning_image.shape[0]

    if image_batch_size == 1:
        repeat_by = batch_size
    else:
        # image batch size is the same as prompt batch size
        repeat_by = num_images_per_prompt

    controlnet_conditioning_image = controlnet_conditioning_image.repeat_interleave(repeat_by, dim=0)

    controlnet_conditioning_image = controlnet_conditioning_image.to(device=device, dtype=dtype)

    return controlnet_conditioning_image


def parse_prompt_attention(text):
    """
    Parses a string with attention tokens and returns a list of pairs: text and its associated weight.
    Accepted tokens are:
      (abc) - increases attention to abc by a multiplier of 1.1
      (abc:3.12) - increases attention to abc by a multiplier of 3.12
      [abc] - decreases attention to abc by a multiplier of 1.1
      \( - literal character '('
      \[ - literal character '['
      \) - literal character ')'
      \] - literal character ']'
      \\ - literal character '\'
      anything else - just text
    >>> parse_prompt_attention('normal text')
    [['normal text', 1.0]]
    >>> parse_prompt_attention('an (important) word')
    [['an ', 1.0], ['important', 1.1], [' word', 1.0]]
    >>> parse_prompt_attention('(unbalanced')
    [['unbalanced', 1.1]]
    >>> parse_prompt_attention('\(literal\]')
    [['(literal]', 1.0]]
    >>> parse_prompt_attention('(unnecessary)(parens)')
    [['unnecessaryparens', 1.1]]
    >>> parse_prompt_attention('a (((house:1.3)) [on] a (hill:0.5), sun, (((sky))).')
    [['a ', 1.0],
     ['house', 1.5730000000000004],
     [' ', 1.1],
     ['on', 1.0],
     [' a ', 1.1],
     ['hill', 0.55],
     [', sun, ', 1.1],
     ['sky', 1.4641000000000006],
     ['.', 1.1]]
    """

    res = []
    round_brackets = []
    square_brackets = []

    round_bracket_multiplier = 1.1
    square_bracket_multiplier = 1 / 1.1

    def multiply_range(start_position, multiplier):
        for p in range(start_position, len(res)):
            res[p][1] *= multiplier

    for m in re_attention.finditer(text):
        text = m.group(0)
        weight = m.group(1)

        if text.startswith("\\"):
            res.append([text[1:], 1.0])
        elif text == "(":
            round_brackets.append(len(res))
        elif text == "[":
            square_brackets.append(len(res))
        elif weight is not None and len(round_brackets) > 0:
            multiply_range(round_brackets.pop(), float(weight))
        elif text == ")" and len(round_brackets) > 0:
            multiply_range(round_brackets.pop(), round_bracket_multiplier)
        elif text == "]" and len(square_brackets) > 0:
            multiply_range(square_brackets.pop(), square_bracket_multiplier)
        else:
            res.append([text, 1.0])

    for pos in round_brackets:
        multiply_range(pos, round_bracket_multiplier)

    for pos in square_brackets:
        multiply_range(pos, square_bracket_multiplier)

    if len(res) == 0:
        res = [["", 1.0]]

    # merge runs of identical weights
    i = 0
    while i + 1 < len(res):
        if res[i][1] == res[i + 1][1]:
            res[i][0] += res[i + 1][0]
            res.pop(i + 1)
        else:
            i += 1

    return res


def get_prompts_with_weights(pipe, prompt: List[str], max_length: int):
    r"""
    Tokenize a list of prompts and return its tokens with weights of each token.

    No padding, starting or ending token is included.
    """
    tokens = []
    weights = []
    truncated = False
    for text in prompt:
        texts_and_weights = parse_prompt_attention(text)
        text_token = []
        text_weight = []
        for word, weight in texts_and_weights:
            # tokenize and discard the starting and the ending token
            token = pipe.tokenizer(word).input_ids[1:-1]
            text_token += token
            # copy the weight by length of token
            text_weight += [weight] * len(token)
            # stop if the text is too long (longer than truncation limit)
            if len(text_token) > max_length:
                truncated = True
                break
        # truncate
        if len(text_token) > max_length:
            truncated = True
            text_token = text_token[:max_length]
            text_weight = text_weight[:max_length]
        tokens.append(text_token)
        weights.append(text_weight)
    if truncated:
        logger.warning("Prompt was truncated. Try to shorten the prompt or increase max_embeddings_multiples")
    return tokens, weights


def pad_tokens_and_weights(tokens, weights, max_length, bos, eos, no_boseos_middle=True, chunk_length=77):
    r"""
    Pad the tokens (with starting and ending tokens) and weights (with 1.0) to max_length.
    """
    max_embeddings_multiples = (max_length - 2) // (chunk_length - 2)
    weights_length = max_length if no_boseos_middle else max_embeddings_multiples * chunk_length
    for i in range(len(tokens)):
        tokens[i] = [bos] + tokens[i] + [eos] * (max_length - 1 - len(tokens[i]))
        if no_boseos_middle:
            weights[i] = [1.0] + weights[i] + [1.0] * (max_length - 1 - len(weights[i]))
        else:
            w = []
            if len(weights[i]) == 0:
                w = [1.0] * weights_length
            else:
                for j in range(max_embeddings_multiples):
                    w.append(1.0)  # weight for starting token in this chunk
                    w += weights[i][j * (chunk_length - 2) : min(len(weights[i]), (j + 1) * (chunk_length - 2))]
                    w.append(1.0)  # weight for ending token in this chunk
                w += [1.0] * (weights_length - len(w))
            weights[i] = w[:]

    return tokens, weights


def get_unweighted_text_embeddings(
    pipe,
    text_input: torch.Tensor,
    chunk_length: int,
    no_boseos_middle: Optional[bool] = True,
):
    """
    When the length of tokens is a multiple of the capacity of the text encoder,
    it should be split into chunks and sent to the text encoder individually.
    """
    max_embeddings_multiples = (text_input.shape[1] - 2) // (chunk_length - 2)
    if max_embeddings_multiples > 1:
        text_embeddings = []
        for i in range(max_embeddings_multiples):
            # extract the i-th chunk
            text_input_chunk = text_input[:, i * (chunk_length - 2) : (i + 1) * (chunk_length - 2) + 2].clone()

            # cover the head and the tail by the starting and the ending tokens
            text_input_chunk[:, 0] = text_input[0, 0]
            text_input_chunk[:, -1] = text_input[0, -1]
            text_embedding = pipe.text_encoder(text_input_chunk)[0]

            if no_boseos_middle:
                if i == 0:
                    # discard the ending token
                    text_embedding = text_embedding[:, :-1]
                elif i == max_embeddings_multiples - 1:
                    # discard the starting token
                    text_embedding = text_embedding[:, 1:]
                else:
                    # discard both starting and ending tokens
                    text_embedding = text_embedding[:, 1:-1]

            text_embeddings.append(text_embedding)
        text_embeddings = torch.concat(text_embeddings, axis=1)
    else:
        text_embeddings = pipe.text_encoder(text_input)[0]
    return text_embeddings


def get_weighted_text_embeddings(
    pipe,
    prompt: Union[str, List[str]],
    uncond_prompt: Optional[Union[str, List[str]]] = None,
    max_embeddings_multiples: Optional[int] = 3,
    no_boseos_middle: Optional[bool] = False,
    skip_parsing: Optional[bool] = False,
    skip_weighting: Optional[bool] = False,
    device=None,
):
    r"""
    Prompts can be assigned with local weights using brackets. For example,
    prompt 'A (very beautiful) masterpiece' highlights the words 'very beautiful',
    and the embedding tokens corresponding to the words get multiplied by a constant, 1.1.

    Also, to regularize of the embedding, the weighted embedding would be scaled to preserve the original mean.

    Args:
        pipe (`StableDiffusionPipeline`):
            Pipe to provide access to the tokenizer and the text encoder.
        prompt (`str` or `List[str]`):
            The prompt or prompts to guide the image generation.
        uncond_prompt (`str` or `List[str]`):
            The unconditional prompt or prompts for guide the image generation. If unconditional prompt
            is provided, the embeddings of prompt and uncond_prompt are concatenated.
        max_embeddings_multiples (`int`, *optional*, defaults to `3`):
            The max multiple length of prompt embeddings compared to the max output length of text encoder.
        no_boseos_middle (`bool`, *optional*, defaults to `False`):
            If the length of text token is multiples of the capacity of text encoder, whether reserve the starting and
            ending token in each of the chunk in the middle.
        skip_parsing (`bool`, *optional*, defaults to `False`):
            Skip the parsing of brackets.
        skip_weighting (`bool`, *optional*, defaults to `False`):
            Skip the weighting. When the parsing is skipped, it is forced True.
    """
    max_length = (pipe.tokenizer.model_max_length - 2) * max_embeddings_multiples + 2
    if isinstance(prompt, str):
        prompt = [prompt]

    if not skip_parsing:
        prompt_tokens, prompt_weights = get_prompts_with_weights(pipe, prompt, max_length - 2)
        if uncond_prompt is not None:
            if isinstance(uncond_prompt, str):
                uncond_prompt = [uncond_prompt]
            uncond_tokens, uncond_weights = get_prompts_with_weights(pipe, uncond_prompt, max_length - 2)
    else:
        prompt_tokens = [
            token[1:-1] for token in pipe.tokenizer(prompt, max_length=max_length, truncation=True).input_ids
        ]
        prompt_weights = [[1.0] * len(token) for token in prompt_tokens]
        if uncond_prompt is not None:
            if isinstance(uncond_prompt, str):
                uncond_prompt = [uncond_prompt]
            uncond_tokens = [
                token[1:-1]
                for token in pipe.tokenizer(uncond_prompt, max_length=max_length, truncation=True).input_ids
            ]
            uncond_weights = [[1.0] * len(token) for token in uncond_tokens]

    # round up the longest length of tokens to a multiple of (model_max_length - 2)
    max_length = max([len(token) for token in prompt_tokens])
    if uncond_prompt is not None:
        max_length = max(max_length, max([len(token) for token in uncond_tokens]))

    max_embeddings_multiples = min(
        max_embeddings_multiples,
        (max_length - 1) // (pipe.tokenizer.model_max_length - 2) + 1,
    )
    max_embeddings_multiples = max(1, max_embeddings_multiples)
    max_length = (pipe.tokenizer.model_max_length - 2) * max_embeddings_multiples + 2

    # pad the length of tokens and weights
    bos = pipe.tokenizer.bos_token_id
    eos = pipe.tokenizer.eos_token_id
    prompt_tokens, prompt_weights = pad_tokens_and_weights(
        prompt_tokens,
        prompt_weights,
        max_length,
        bos,
        eos,
        no_boseos_middle=no_boseos_middle,
        chunk_length=pipe.tokenizer.model_max_length,
    )
    prompt_tokens = torch.tensor(prompt_tokens, dtype=torch.long, device=device)
    if uncond_prompt is not None:
        uncond_tokens, uncond_weights = pad_tokens_and_weights(
            uncond_tokens,
            uncond_weights,
            max_length,
            bos,
            eos,
            no_boseos_middle=no_boseos_middle,
            chunk_length=pipe.tokenizer.model_max_length,
        )
        uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=device)

    # get the embeddings
    text_embeddings = get_unweighted_text_embeddings(
        pipe,
        prompt_tokens,
        pipe.tokenizer.model_max_length,
        no_boseos_middle=no_boseos_middle,
    )
    prompt_weights = torch.tensor(prompt_weights, dtype=text_embeddings.dtype, device=device)
    if uncond_prompt is not None:
        uncond_embeddings = get_unweighted_text_embeddings(
            pipe,
            uncond_tokens,
            pipe.tokenizer.model_max_length,
            no_boseos_middle=no_boseos_middle,
        )
        uncond_weights = torch.tensor(uncond_weights, dtype=uncond_embeddings.dtype, device=device)

    # assign weights to the prompts and normalize in the sense of mean
    # TODO: should we normalize by chunk or in a whole (current implementation)?
    if (not skip_parsing) and (not skip_weighting):
        previous_mean = text_embeddings.float().mean(axis=[-2, -1]).to(text_embeddings.dtype)
        text_embeddings *= prompt_weights.unsqueeze(-1)
        current_mean = text_embeddings.float().mean(axis=[-2, -1]).to(text_embeddings.dtype)
        text_embeddings *= (previous_mean / current_mean).unsqueeze(-1).unsqueeze(-1)
        if uncond_prompt is not None:
            previous_mean = uncond_embeddings.float().mean(axis=[-2, -1]).to(uncond_embeddings.dtype)
            uncond_embeddings *= uncond_weights.unsqueeze(-1)
            current_mean = uncond_embeddings.float().mean(axis=[-2, -1]).to(uncond_embeddings.dtype)
            uncond_embeddings *= (previous_mean / current_mean).unsqueeze(-1).unsqueeze(-1)

    if uncond_prompt is not None:
        return text_embeddings, uncond_embeddings
    return text_embeddings, None


class VisualControlNetModel(ControlNetModel):
    def __init__(self, in_channels: int = 4, flip_sin_to_cos: bool = True, freq_shift: int = 0, down_block_types: Tuple[str] = ..., only_cross_attention: Union[bool, Tuple[bool]] = False, block_out_channels: Tuple[int] = ..., layers_per_block: int = 2, downsample_padding: int = 1, mid_block_scale_factor: float = 1, act_fn: str = "silu", norm_num_groups: Optional[int] = 32, norm_eps: float = 0.00001, cross_attention_dim: int = 1280, attention_head_dim: Union[int, Tuple[int]] = 8, use_linear_projection: bool = False, class_embed_type: Optional[str] = None, num_class_embeds: Optional[int] = None, upcast_attention: bool = False, resnet_time_scale_shift: str = "default", projection_class_embeddings_input_dim: Optional[int] = None, controlnet_conditioning_channel_order: str = "rgb", conditioning_embedding_out_channels: Optional[Tuple[int]] = ...):
        super().__init__(in_channels, flip_sin_to_cos, freq_shift, down_block_types, only_cross_attention, block_out_channels, layers_per_block, downsample_padding, mid_block_scale_factor, act_fn, norm_num_groups, norm_eps, cross_attention_dim, attention_head_dim, use_linear_projection, class_embed_type, num_class_embeds, upcast_attention, resnet_time_scale_shift, projection_class_embeddings_input_dim, controlnet_conditioning_channel_order, conditioning_embedding_out_channels)
        self.visual_projection = torch.nn.Linear(512, 768)
        self.visual_position_embedding = torch.nn.Embedding(77, 768)
        self.register_buffer("visual_nb_token_repeat", torch.tensor(64))
        self.register_buffer("visual_position_ids", torch.arange(512).reshape([1, -1]))

    def visual_embedding(self, visual_embs):
        # visual_embs [bs, 1, d0]
        visual_embs = self.visual_projection(visual_embs)  # bs, 1, d1
        visual_embs = torch.repeat_interleave(visual_embs, self.visual_nb_token_repeat, dim=1)  # bs, visual_nb_token_repeat, 768
        visual_position_embs = self.visual_position_embedding(self.visual_position_ids[:, :visual_embs.shape[1]])  # bs, visual_nb_token_repeat, 768
        visual_embs = visual_embs + visual_position_embs

        return visual_embs  # [bs, nb_face_token, 768]


def draw_kps(image_pil, kps, color_list=[(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255)], radius=10):
    w, h = image_pil.size
    out_img = np.zeros([h, w, 3])
    for idx_kp, kp in enumerate(kps):
        color = color_list[idx_kp]
        x, y = kp
        out_img = cv2.circle(out_img.copy(), (int(x), int(y)), radius, color, -1)
    out_img_pil = Image.fromarray(out_img.astype(np.uint8))
    return out_img_pil


class StableDiffusionCommonPipeline(DiffusionPipeline, TextualInversionLoaderMixin, LoraLoaderMixin, FromCkptMixin):

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: StableDiffusionSafetyChecker = None,
        feature_extractor: CLIPFeatureExtractor = None,
        # requires_safety_checker: bool = True,
        controlnet_list: torch.nn.ModuleList = [],
    ):

        super().__init__()

        controlnet_model_list = []
        for ctn in controlnet_list:
            try:
                ctn = ControlNetModel.from_pretrained(ctn)
            except Exception as e:
                ctn = VisualControlNetModel.from_pretrained(ctn, low_cpu_mem_usage=False, device_map=None, ignore_mismatched_sizes=True)
            controlnet_model_list.append(ctn)
        controlnet_model_list = torch.nn.ModuleList(controlnet_model_list)

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            controlnet_list=controlnet_model_list,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )
        self.controlnet_list.dtype = self.unet.dtype
        self.controlnet_list.device = self.unet.device

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        # self.register_to_config(requires_safety_checker=requires_safety_checker)


    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        safe_serialization: bool = False,
        variant: Optional[str] = None,
    ):
        """
        Save all variables of the pipeline that can be saved and loaded as well as the pipelines configuration file to
        a directory. A pipeline variable can be saved and loaded if its class implements both a save and loading
        method. The pipeline can easily be re-loaded using the `[`~DiffusionPipeline.from_pretrained`]` class method.

        Arguments:
            save_directory (`str` or `os.PathLike`):
                Directory to which to save. Will be created if it doesn't exist.
            safe_serialization (`bool`, *optional*, defaults to `False`):
                Whether to save the model using `safetensors` or the traditional PyTorch way (that uses `pickle`).
            variant (`str`, *optional*):
                If specified, weights are saved in the format pytorch_model.<variant>.bin.
        """
        self.save_config(save_directory)

        model_index_dict = dict(self.config)
        model_index_dict.pop("_class_name")
        model_index_dict.pop("_diffusers_version")
        model_index_dict.pop("_module", None)

        expected_modules, optional_kwargs = self._get_signature_keys(self)

        def is_saveable_module(name, value):
            if name not in expected_modules:
                return False
            if name in self._optional_components and value[0] is None:
                return False
            return True

        model_index_dict = {k: v for k, v in model_index_dict.items() if is_saveable_module(k, v)}

        for pipeline_component_name in model_index_dict.keys():
            sub_model = getattr(self, pipeline_component_name)
            model_cls = sub_model.__class__

            save_method_name = None
            # search for the model's base class in LOADABLE_CLASSES
            for library_name, library_classes in LOADABLE_CLASSES.items():
                library = importlib.import_module(library_name)
                for base_class, save_load_methods in library_classes.items():
                    class_candidate = getattr(library, base_class, None)
                    if class_candidate is not None and issubclass(model_cls, class_candidate):
                        # if we found a suitable base class in LOADABLE_CLASSES then grab its save method
                        save_method_name = save_load_methods[0]
                        break
                if save_method_name is not None:
                    break
            
            if sub_model is None:
                print(f"{pipeline_component_name} is None. Not save!")
                continue
            save_method = getattr(sub_model, save_method_name)

            # Call the save method with the argument safe_serialization only if it's supported
            save_method_signature = inspect.signature(save_method)
            save_method_accept_safe = "safe_serialization" in save_method_signature.parameters
            save_method_accept_variant = "variant" in save_method_signature.parameters

            save_kwargs = {}
            if save_method_accept_safe:
                save_kwargs["safe_serialization"] = safe_serialization
            if save_method_accept_variant:
                save_kwargs["variant"] = variant

            save_method(os.path.join(save_directory, pipeline_component_name), **save_kwargs)
            print(f"{pipeline_component_name} successfully saved in {os.path.join(save_directory, pipeline_component_name)}")

    def to(self, torch_device: Optional[Union[str, torch.device]] = None, silence_dtype_warnings: bool = False):
        if torch_device is None:
            return self

        # throw warning if pipeline is in "offloaded"-mode but user tries to manually set to GPU.
        def module_is_sequentially_offloaded(module):
            if not is_accelerate_available() or is_accelerate_version("<", "0.14.0"):
                return False

            return hasattr(module, "_hf_hook") and not isinstance(module._hf_hook, accelerate.hooks.CpuOffload)

        def module_is_offloaded(module):
            if not is_accelerate_available() or is_accelerate_version("<", "0.17.0.dev0"):
                return False

            return hasattr(module, "_hf_hook") and isinstance(module._hf_hook, accelerate.hooks.CpuOffload)

        # .to("cuda") would raise an error if the pipeline is sequentially offloaded, so we raise our own to make it clearer
        pipeline_is_sequentially_offloaded = any(
            module_is_sequentially_offloaded(module) for _, module in self.components.items()
        )
        if pipeline_is_sequentially_offloaded and torch.device(torch_device).type == "cuda":
            raise ValueError(
                "It seems like you have activated sequential model offloading by calling `enable_sequential_cpu_offload`, but are now attempting to move the pipeline to GPU. This is not compatible with offloading. Please, move your pipeline `.to('cpu')` or consider removing the move altogether if you use sequential offloading."
            )

        # Display a warning in this case (the operation succeeds but the benefits are lost)
        pipeline_is_offloaded = any(module_is_offloaded(module) for _, module in self.components.items())
        if pipeline_is_offloaded and torch.device(torch_device).type == "cuda":
            logger.warning(
                f"It seems like you have activated model offloading by calling `enable_model_cpu_offload`, but are now manually moving the pipeline to GPU. It is strongly recommended against doing so as memory gains from offloading are likely to be lost. Offloading automatically takes care of moving the individual components {', '.join(self.components.keys())} to GPU when needed. To make sure offloading works as expected, you should consider moving the pipeline back to CPU: `pipeline.to('cpu')` or removing the move altogether if you use offloading."
            )

        module_names, _, _ = self.extract_init_dict(dict(self.config))
        is_offloaded = pipeline_is_offloaded or pipeline_is_sequentially_offloaded
        for name in module_names.keys():
            module = getattr(self, name)
            if isinstance(module, torch.nn.Module):
                if (
                    module.dtype == torch.float16
                    and str(torch_device) in ["cpu"]
                    and not silence_dtype_warnings
                    and not is_offloaded
                ):
                    logger.warning(
                        "Pipelines loaded with `torch_dtype=torch.float16` cannot run with `cpu` device. It"
                        " is not recommended to move them to `cpu` as running them will fail. Please make"
                        " sure to use an accelerator to run the pipeline in inference, due to the lack of"
                        " support for`float16` operations on this device in PyTorch. Please, remove the"
                        " `torch_dtype=torch.float16` argument, or use another device for inference."
                    )
                module.to(torch_device)
        
        self.controlnet_list.device = self.unet.device
        self.controlnet_list.dtype = self.unet.dtype
        # print(self.controlnet_list[0].device, self.controlnet_list[0].dtype)
        
        return self

    def cuda_half(self, ):
        self = self.to(torch.float16)
        self.vae = self.vae.to(torch.float32)
        self = self.to('cuda')
        return self

    def enable_vae_slicing(self):
        r"""
        Enable sliced VAE decoding.

        When this option is enabled, the VAE will split the input tensor in slices to compute decoding in several
        steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_vae_slicing` was previously invoked, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_slicing()

    def enable_sequential_cpu_offload(self, gpu_id=0):
        r"""
        Offloads all models to CPU using accelerate, significantly reducing memory usage. When called, unet,
        text_encoder, vae, controlnet, and safety checker have their state dicts saved to CPU and then are moved to a
        `torch.device('meta') and loaded to GPU only when their specific submodule has its `forward` method called.
        Note that offloading happens on a submodule basis. Memory savings are higher than with
        `enable_model_cpu_offload`, but performance is lower.
        """
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError("Please install accelerate via `pip install accelerate`")

        device = torch.device(f"cuda:{gpu_id}")

        for cpu_offloaded_model in [self.unet, self.text_encoder, self.vae, self.controlnet]:
            cpu_offload(cpu_offloaded_model, device)

        if self.safety_checker is not None:
            cpu_offload(self.safety_checker, execution_device=device, offload_buffers=True)

    def enable_model_cpu_offload(self, gpu_id=0):
        r"""
        Offloads all models to CPU using accelerate, reducing memory usage with a low impact on performance. Compared
        to `enable_sequential_cpu_offload`, this method moves one whole model at a time to the GPU when its `forward`
        method is called, and the model remains in GPU until the next model runs. Memory savings are lower than with
        `enable_sequential_cpu_offload`, but performance is much better due to the iterative execution of the `unet`.
        """
        if is_accelerate_available() and is_accelerate_version(">=", "0.17.0.dev0"):
            from accelerate import cpu_offload_with_hook
        else:
            raise ImportError("`enable_model_offload` requires `accelerate v0.17.0` or higher.")

        device = torch.device(f"cuda:{gpu_id}")

        hook = None
        for cpu_offloaded_model in [self.text_encoder, self.unet, self.vae]:
            _, hook = cpu_offload_with_hook(cpu_offloaded_model, device, prev_module_hook=hook)

        if self.safety_checker is not None:
            # the safety checker can offload the vae again
            _, hook = cpu_offload_with_hook(self.safety_checker, device, prev_module_hook=hook)

        # control net hook has be manually offloaded as it alternates with unet
        cpu_offload_with_hook(self.controlnet, device)

        # We'll offload the last model manually.
        self.final_offload_hook = hook

    def set_use_memory_efficient_attention_xformers(
        self, valid: bool, attention_op: Optional[Callable] = None
    ) -> None:
        # Recursively walk through all the children.
        # Any children which exposes the set_use_memory_efficient_attention_xformers method
        # gets the message
        def fn_recursive_set_mem_eff(module: torch.nn.Module):
            if hasattr(module, "set_use_memory_efficient_attention_xformers"):
                module.set_use_memory_efficient_attention_xformers(valid, attention_op)

            for child in module.children():
                fn_recursive_set_mem_eff(child)

        # embed()
        # xxxxxx
        module_names, _ = self._get_signature_keys(self)
        module_names = list(module_names) + ['controlnet_list']
        modules = [getattr(self, n, None) for n in module_names]
        modules = [m for m in modules if isinstance(m, torch.nn.Module)]


        for module in modules:
            fn_recursive_set_mem_eff(module)

    @property
    def _execution_device(self):
        r"""
        Returns the device on which the pipeline's models will be executed. After calling
        `pipeline.enable_sequential_cpu_offload()` the execution device can only be inferred from Accelerate's module
        hooks.
        """
        if not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def run_safety_checker(self, image, device, dtype):
        if self.safety_checker is not None:
            safety_checker_input = self.feature_extractor(self.numpy_to_pil(image), return_tensors="pt").to(device)
            image, has_nsfw_concept = self.safety_checker(
                images=image, clip_input=safety_checker_input.pixel_values.to(dtype)
            )
        else:
            has_nsfw_concept = None
        return image, has_nsfw_concept

    def decode_latents(self, latents):
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(
        self,
        prompt,
        image,
        controlnet_conditioning_image,
        height,
        width,
        callback_steps,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        strength=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

        controlnet_cond_image_is_pil = isinstance(controlnet_conditioning_image, PIL.Image.Image)
        controlnet_cond_image_is_tensor = isinstance(controlnet_conditioning_image, torch.Tensor)
        controlnet_cond_image_is_pil_list = isinstance(controlnet_conditioning_image, list) and isinstance(
            controlnet_conditioning_image[0], PIL.Image.Image
        )
        controlnet_cond_image_is_tensor_list = isinstance(controlnet_conditioning_image, list) and isinstance(
            controlnet_conditioning_image[0], torch.Tensor
        )

        if (
            not controlnet_cond_image_is_pil
            and not controlnet_cond_image_is_tensor
            and not controlnet_cond_image_is_pil_list
            and not controlnet_cond_image_is_tensor_list
        ):
            raise TypeError(
                "image must be passed and be one of PIL image, torch tensor, list of PIL images, or list of torch tensors"
            )

        if controlnet_cond_image_is_pil:
            controlnet_cond_image_batch_size = 1
        elif controlnet_cond_image_is_tensor:
            controlnet_cond_image_batch_size = controlnet_conditioning_image.shape[0]
        elif controlnet_cond_image_is_pil_list:
            controlnet_cond_image_batch_size = len(controlnet_conditioning_image)
        elif controlnet_cond_image_is_tensor_list:
            controlnet_cond_image_batch_size = len(controlnet_conditioning_image)

        if prompt is not None and isinstance(prompt, str):
            prompt_batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            prompt_batch_size = len(prompt)
        elif prompt_embeds is not None:
            prompt_batch_size = prompt_embeds.shape[0]

        if controlnet_cond_image_batch_size != 1 and controlnet_cond_image_batch_size != prompt_batch_size:
            raise ValueError(
                f"If image batch size is not 1, image batch size must be same as prompt batch size. image batch size: {controlnet_cond_image_batch_size}, prompt batch size: {prompt_batch_size}"
            )

        if isinstance(image, torch.Tensor):
            if image.ndim != 3 and image.ndim != 4:
                raise ValueError("`image` must have 3 or 4 dimensions")

            # if mask_image.ndim != 2 and mask_image.ndim != 3 and mask_image.ndim != 4:
            #     raise ValueError("`mask_image` must have 2, 3, or 4 dimensions")

            if image.ndim == 3:
                image_batch_size = 1
                image_channels, image_height, image_width = image.shape
            elif image.ndim == 4:
                image_batch_size, image_channels, image_height, image_width = image.shape

            if image_channels != 3:
                raise ValueError("`image` must have 3 channels")

            if image.min() < -1 or image.max() > 1:
                raise ValueError("`image` should be in range [-1, 1]")

        if self.vae.config.latent_channels != self.unet.config.in_channels:
            raise ValueError(
                f"The config of `pipeline.unet` expects {self.unet.config.in_channels} but received"
                f" latent channels: {self.vae.config.latent_channels},"
                f" Please verify the config of `pipeline.unet` and the `pipeline.vae`"
            )

        if strength < 0 or strength > 1:
            raise ValueError(f"The value of strength should in [0.0, 1.0] but is {strength}")

    def get_timesteps(self, num_inference_steps, strength, device):
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start:]

        return timesteps, num_inference_steps - t_start

    def prepare_latents(self, image, timestep, batch_size, num_images_per_prompt, dtype, device, generator=None, use_last_noise=False):
        if not isinstance(image, (torch.Tensor, PIL.Image.Image, list)):
            raise ValueError(
                f"`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}"
            )

        image = image.to(device=device, dtype=dtype)

        batch_size = batch_size * num_images_per_prompt
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if isinstance(generator, list):
            init_latents = [
                self.vae.encode(image[i : i + 1]).latent_dist.sample(generator[i]) for i in range(batch_size)
            ]
            init_latents = torch.cat(init_latents, dim=0)
        else:
            init_latents = self.vae.encode(image).latent_dist.sample(generator)

        init_latents = self.vae.config.scaling_factor * init_latents

        if batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] == 0:
            raise ValueError(
                f"Cannot duplicate `image` of batch size {init_latents.shape[0]} to {batch_size} text prompts."
            )
        else:
            init_latents = torch.cat([init_latents], dim=0)

        shape = init_latents.shape
        if use_last_noise and hasattr(self, 'last_noise'):
            noise = self.last_noise
        else:
            noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        if not hasattr(self, 'last_noise'):
            self.last_noise = noise

        # get latents
        init_latents = self.scheduler.add_noise(init_latents, noise, timestep)
        latents = init_latents

        return latents

    def _default_height_width(self, height, width, image):
        if isinstance(image, list):
            image = image[0]

        if height is None:
            if isinstance(image, PIL.Image.Image):
                height = image.height
            elif isinstance(image, torch.Tensor):
                height = image.shape[3]

            height = (height // 8) * 8  # round down to nearest multiple of 8

        if width is None:
            if isinstance(image, PIL.Image.Image):
                width = image.width
            elif isinstance(image, torch.Tensor):
                width = image.shape[2]

            width = (width // 8) * 8  # round down to nearest multiple of 8

        return height, width

    def _encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt,
        max_embeddings_multiples,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `list(int)`):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
            max_embeddings_multiples (`int`, *optional*, defaults to `3`):
                The max multiple length of prompt embeddings compared to the max output length of text encoder.
        """
        batch_size = len(prompt) if isinstance(prompt, list) else 1

        if negative_prompt is None:
            negative_prompt = [""] * batch_size
        elif isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt] * batch_size
        if batch_size != len(negative_prompt):
            raise ValueError(
                f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                " the batch size of `prompt`."
            )

        text_embeddings, uncond_embeddings = get_weighted_text_embeddings(
            pipe=self,
            prompt=prompt,
            uncond_prompt=negative_prompt if do_classifier_free_guidance else None,
            max_embeddings_multiples=max_embeddings_multiples,
            device=device
        )
        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_images_per_prompt, 1)
        text_embeddings = text_embeddings.view(bs_embed * num_images_per_prompt, seq_len, -1)

        if do_classifier_free_guidance:
            bs_embed, seq_len, _ = uncond_embeddings.shape
            uncond_embeddings = uncond_embeddings.repeat(1, num_images_per_prompt, 1)
            uncond_embeddings = uncond_embeddings.view(bs_embed * num_images_per_prompt, seq_len, -1)
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        return text_embeddings

    def _prepare_latent_couple_mask(self, mask_image_raw, height, width, device, dtype):
        mask_image = mask_image_raw.resize((width//8, height//8))
        mask_np = np.array(mask_image) / 255.
        mask_np = mask_np[..., 0]
        mask_tensor = torch.from_numpy(mask_np).to(device).to(dtype)
        return mask_tensor

    def _prepare_latent_couple_attention_mask(self, mask_tensor, sequence_length):
        attention_mask = mask_tensor.reshape([-1, 1])
        attention_mask = attention_mask.repeat_interleave(sequence_length, dim=1)
        attention_mask = attention_mask.unsqueeze(0)
        # [1, hw, sequence_length]
        return attention_mask

    def _prepare_latent_couple_attention_probs_weight(self, attention_probs_weight, sequence_length):
        attention_probs_weight = attention_probs_weight.reshape([-1, 1])
        attention_probs_weight = attention_probs_weight.repeat_interleave(sequence_length, dim=1)
        attention_probs_weight = attention_probs_weight.unsqueeze(0)
        # [1, hw, sequence_length]
        return attention_probs_weight


    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def img2img(
        self,
        prompt: Union[str, List[str]] = None,
        image: Union[torch.Tensor, PIL.Image.Image] = None,
        controlnet_conditioning=[],
        latent_couple_conditioning=[],
        strength: float = 0.8,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        max_embeddings_multiples=3,
        use_last_noise=False,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            image (`torch.Tensor` or `PIL.Image.Image`):
                `Image`, or tensor representing an image batch which will be inpainted, *i.e.* parts of the image will
                be masked out with `mask_image` and repainted according to `prompt`.
            controlnet_conditioning_image (`torch.FloatTensor`, `PIL.Image.Image`, `List[torch.FloatTensor]` or `List[PIL.Image.Image]`):
                The ControlNet input condition. ControlNet uses this input condition to generate guidance to Unet. If
                the type is specified as `Torch.FloatTensor`, it is passed to ControlNet as is. PIL.Image.Image` can
                also be accepted as an image. The control image is automatically resized to fit the output image.
            strength (`float`, *optional*):
                Conceptually, indicates how much to transform the reference `image`. Must be between 0 and 1. `image`
                will be used as a starting point, adding more noise to it the larger the `strength`. The number of
                denoising steps depends on the amount of noise initially added. When `strength` is 1, added noise will
                be maximum and the denoising process will run for the full number of iterations specified in
                `num_inference_steps`. A value of 1, therefore, essentially ignores `image`.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds`. instead. If not defined, one has to pass `negative_prompt_embeds`. instead.
                Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttnProcessor` as defined under
                `self.processor` in
                [diffusers.cross_attention](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).
            controlnet_conditioning_scale (`float`, *optional*, defaults to 1.0):
                The outputs of the controlnet are multiplied by `controlnet_conditioning_scale` before they are added
                to the residual in the original unet.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """
        # 0. Default height and width to unet
        height, width = self._default_height_width(height, width, image)

        # 1. Check inputs. Raise error if not correct
        # self.check_inputs(
        #     prompt,
        #     image,
        #     # mask_image,
        #     controlnet_conditioning_image_list,
        #     height,
        #     width,
        #     callback_steps,
        #     negative_prompt,
        #     prompt_embeds,
        #     negative_prompt_embeds,
        #     strength,
        # )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        dtype = self.unet.dtype
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            max_embeddings_multiples,
        )
        encoder_hidden_states = prompt_embeds

        if len(latent_couple_conditioning) > 0:
            lc_prompt = [lc['prompt'] for lc in latent_couple_conditioning]
            lc_prompt_embeds = self._encode_prompt(
                prompt=lc_prompt,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                do_classifier_free_guidance=False,
                negative_prompt=None,
                max_embeddings_multiples=max_embeddings_multiples,
            )
            lc_encoder_hidden_states = lc_prompt_embeds
        for idx_lc in range(len(latent_couple_conditioning)):
            mask_image_raw = latent_couple_conditioning[idx_lc]['mask']
            latent_couple_conditioning[idx_lc]['mask'] = self._prepare_latent_couple_mask(mask_image_raw, height, width, device, dtype)
            if 'controlnet_conditioning' in latent_couple_conditioning[idx_lc]:
                for idx_lc_cc in range(len(latent_couple_conditioning[idx_lc]['controlnet_conditioning'])):
                    latent_couple_conditioning[idx_lc]['controlnet_conditioning'][idx_lc_cc]['image'] = prepare_controlnet_conditioning_image(
                        latent_couple_conditioning[idx_lc]['controlnet_conditioning'][idx_lc_cc]['image'],
                        width,
                        height,
                        batch_size * num_images_per_prompt,
                        num_images_per_prompt,
                        device,
                        self.controlnet_list[0].dtype,
                    )
                    # lc_controlnet_index_list.append(latent_couple_conditioning[idx_lc]['control']['index'])
                    if 'visual_emb' in latent_couple_conditioning[idx_lc]['controlnet_conditioning'][idx_lc_cc]:
                        face_emb = torch.tensor(latent_couple_conditioning[idx_lc]['controlnet_conditioning'][idx_lc_cc]['visual_emb']).to(device=device, dtype=dtype)
                        face_emb = face_emb.reshape([-1, 1, 512])
                        latent_couple_conditioning[idx_lc]['controlnet_conditioning'][idx_lc_cc]['visual_emb'] = face_emb

        # 4. Prepare mask
        image = prepare_image(image)

        # Prepare control_image
        for idx in range(len(controlnet_conditioning)):
            cc_info = controlnet_conditioning[idx]
            cc_info['control_image'] = prepare_controlnet_conditioning_image(
                cc_info['control_image'],
                width,
                height,
                batch_size * num_images_per_prompt,
                num_images_per_prompt,
                device,
                self.controlnet_list[0].dtype,
            )
            if 'control_visual_emb' in cc_info:
                face_emb = torch.tensor(cc_info['control_visual_emb']).to(device=device, dtype=dtype)
                face_emb = face_emb.reshape([-1, 1, 512])
                cc_info['control_visual_emb'] = face_emb

        # controlnet_conditioning_image = prepare_controlnet_conditioning_image(
        #     controlnet_conditioning_image,
        #     width,
        #     height,
        #     batch_size * num_images_per_prompt,
        #     num_images_per_prompt,
        #     device,
        #     self.controlnet.dtype,
        # )

        # masked_image = image * (mask_image < 0.5)

        # 5. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength, device)
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)

        # 6. Prepare latent variables
        latents = self.prepare_latents(
            image,
            latent_timestep,
            batch_size,
            num_images_per_prompt,
            self.vae.dtype,
            device,
            generator,
            use_last_noise=use_last_noise,
        )
        latents = latents.to(self.unet.dtype)

        if do_classifier_free_guidance:
            for idx in range(len(controlnet_conditioning)):
                cc_info = controlnet_conditioning[idx]

                # control_image
                control_image_list = [cc_info['control_image']]
                # for idx_lc in range(len(latent_couple_conditioning)):
                #     control_image_list.append(cc_info['control_image'])
                control_image_list.append(cc_info['control_image'])
                cc_info['control_image'] = torch.cat(control_image_list)

                # control_visual_emb
                if 'control_visual_emb' in cc_info:
                    if 'control_visual_emb_neg' in cc_info:
                        control_visual_emb_neg = cc_info['control_visual_emb_neg']
                        control_visual_emb_neg = torch.tensor(control_visual_emb_neg).to(device=device, dtype=dtype)
                        control_visual_emb_neg = control_visual_emb_neg.reshape([-1, 1, 512])
                    else:
                        control_visual_emb_neg = torch.zeros_like(cc_info['control_visual_emb'])
                    control_visual_emb_list = [control_visual_emb_neg]
                    # for idx_lc in range(len(latent_couple_conditioning)):
                    #     control_visual_emb_list.append(cc_info['control_visual_emb'])
                    control_visual_emb_list.append(cc_info['control_visual_emb'])
                    cc_info['control_visual_emb'] = torch.cat(control_visual_emb_list)


        # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)


        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                if len(controlnet_conditioning) > 0:
                    down_block_res_samples_list, mid_block_res_sample_list = [], []
                    for cc_info in controlnet_conditioning:
                        controlnet_conditioning_image = cc_info['control_image']
                        controlnet = self.controlnet_list[int(cc_info['control_index'])]
                        controlnet_conditioning_scale = cc_info['control_weight']
                        if 'control_visual_emb' in cc_info:
                            controlnet_encoder_hidden_states = controlnet.visual_embedding(cc_info['control_visual_emb'])
                        else:
                            controlnet_encoder_hidden_states = encoder_hidden_states
                        down_block_res_samples, mid_block_res_sample = controlnet(
                            latent_model_input,
                            t,
                            encoder_hidden_states=controlnet_encoder_hidden_states,
                            controlnet_cond=controlnet_conditioning_image,
                            return_dict=False,
                        )
                        down_block_res_samples = [
                            down_block_res_sample * controlnet_conditioning_scale
                            for down_block_res_sample in down_block_res_samples
                        ]
                        mid_block_res_sample *= controlnet_conditioning_scale
                        down_block_res_samples_list.append(down_block_res_samples)
                        mid_block_res_sample_list.append(mid_block_res_sample)

                    down_block_res_samples, mid_block_res_sample = [0]*len(down_block_res_samples_list[0]), 0
                    for idx_controlnet in range(len(down_block_res_samples_list)):
                        need_permute = False
                        if isinstance(mid_block_res_sample, torch.Tensor):
                            _,_,h1,w1 = mid_block_res_sample_list[idx_controlnet].shape
                            _,_,h2,w2 = mid_block_res_sample.shape
                            if h1 != h2 and w1 != w2:
                                need_permute = True
                        if need_permute:
                            mid_block_res_sample = torch.permute(mid_block_res_sample, [0,2,3,1]) + torch.permute(mid_block_res_sample_list[idx_controlnet], [0,2,3,1])
                            mid_block_res_sample = torch.permute(mid_block_res_sample, [0,3,1,2])
                        else:
                            mid_block_res_sample += mid_block_res_sample_list[idx_controlnet]
                        for idx_dbrs in range(len(down_block_res_samples_list[idx_controlnet])):
                            need_permute = False
                            if isinstance(down_block_res_samples[idx_dbrs], torch.Tensor):
                                _,_,h1,w1 = down_block_res_samples_list[idx_controlnet][idx_dbrs].shape
                                _,_,h2,w2 = down_block_res_samples[idx_dbrs].shape
                                if h1 != h2 and w1 != w2:
                                    need_permute = True
                            if need_permute:
                                down_block_res_samples[idx_dbrs] = torch.permute(down_block_res_samples[idx_dbrs], [0,2,3,1]) + torch.permute(down_block_res_samples_list[idx_controlnet][idx_dbrs], [0,2,3,1])
                                down_block_res_samples[idx_dbrs] = torch.permute(down_block_res_samples[idx_dbrs], [0,3,1,2])
                            else:
                                down_block_res_samples[idx_dbrs] += down_block_res_samples_list[idx_controlnet][idx_dbrs]
                else:
                    down_block_res_samples, mid_block_res_sample = [0] * 12, 0

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                ).sample

                # latent couple every sample
                if len(latent_couple_conditioning) > 0:
                    lc_noise_pred_list = []
                    for idx_lc in range(len(latent_couple_conditioning)):
                        lcc_info = latent_couple_conditioning[idx_lc]
                        lc_controlnet_conditioning = lcc_info.get('controlnet_conditioning', [])

                        # controlnet
                        if len(lc_controlnet_conditioning) > 0:
                            lc_down_block_res_samples_list, lc_mid_block_res_sample_list = [], []
                            for cc_info in lc_controlnet_conditioning:
                                controlnet_conditioning_image = cc_info['image']
                                controlnet = self.controlnet_list[int(cc_info['index'])]
                                controlnet_conditioning_scale = cc_info['weight']
                                if 'visual_emb' in cc_info:
                                    controlnet_encoder_hidden_states = controlnet.visual_embedding(cc_info['visual_emb'])
                                else:
                                    controlnet_encoder_hidden_states = lc_encoder_hidden_states[idx_lc:idx_lc+1]
                                lc_down_block_res_samples, lc_mid_block_res_sample = controlnet(
                                    latent_model_input[0:1],
                                    t,
                                    encoder_hidden_states=controlnet_encoder_hidden_states,
                                    controlnet_cond=controlnet_conditioning_image,
                                    return_dict=False,
                                )
                                lc_down_block_res_samples = [
                                    down_block_res_sample * controlnet_conditioning_scale
                                    for down_block_res_sample in lc_down_block_res_samples
                                ]
                                lc_mid_block_res_sample *= controlnet_conditioning_scale
                                lc_down_block_res_samples_list.append(lc_down_block_res_samples)
                                lc_mid_block_res_sample_list.append(lc_mid_block_res_sample)

                            lc_down_block_res_samples, lc_mid_block_res_sample = [0] * len(lc_down_block_res_samples_list[0]), 0
                            for idx_controlnet in range(len(lc_down_block_res_samples_list)):
                                lc_mid_block_res_sample += lc_mid_block_res_sample_list[idx_controlnet]
                                for idx_dbrs in range(len(lc_down_block_res_samples_list[idx_controlnet])):
                                    lc_down_block_res_samples[idx_dbrs] += lc_down_block_res_samples_list[idx_controlnet][idx_dbrs]
                        else:
                            lc_down_block_res_samples, lc_mid_block_res_sample = [0] * 12, 0

                        lc_noise_pred = self.unet(
                            latent_model_input[0:1],
                            t,
                            encoder_hidden_states=lc_encoder_hidden_states[idx_lc:idx_lc+1],
                            cross_attention_kwargs=cross_attention_kwargs,
                            down_block_additional_residuals=lc_down_block_res_samples,
                            mid_block_additional_residual=lc_mid_block_res_sample,
                        ).sample
                        lc_noise_pred_list.append(lc_noise_pred)


                # perform guidance
                if do_classifier_free_guidance:
                    if len(latent_couple_conditioning) > 0:
                        noise_pred_uncond = noise_pred[0:1]
                        noise_preds = [noise_pred[1:2]] + lc_noise_pred_list  # prompt, lc1, lc2 ....
                        noise_preds = torch.cat(noise_preds)
                        noise_preds = noise_pred_uncond + guidance_scale * (noise_preds - noise_pred_uncond)

                        background_weight = (1 - np.mean([lcc['weight'] for lcc in latent_couple_conditioning]))
                        result = noise_preds[0:1] * background_weight
                        mask_empty = torch.ones_like(latent_couple_conditioning[0]['mask'])  # 没被lc覆盖的背景区域
                        for idx_lc in range(1, len(noise_preds)):
                            weight_idx = latent_couple_conditioning[idx_lc-1]['weight']
                            mask_idx = latent_couple_conditioning[idx_lc-1]['mask']

                            # 当前lc和历史lc的交集区域
                            mask_intersection = mask_idx * (1 - mask_empty)

                            result += noise_preds[idx_lc:idx_lc+1] * weight_idx * mask_idx * (1 - mask_intersection)
                            result = (0.5 * result + 0.5 * (weight_idx * noise_preds[idx_lc:idx_lc+1] + (1 - weight_idx) * noise_preds[0:1])) * mask_intersection + result * (1 - mask_intersection)

                            # 没被lc覆盖的背景区域
                            mask_empty = mask_empty * (1 - mask_idx)

                            # result += noise_preds[idx_lc:idx_lc+1] * weight_idx * mask_idx
                        result += noise_preds[0:1] * (1 - background_weight) * mask_empty
                        noise_pred = result
                    else:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        # If we do sequential model offloading, let's offload unet and controlnet
        # manually for max memory savings
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.unet.to("cpu")
            self.controlnet.to("cpu")
            torch.cuda.empty_cache()

        latents = latents.to(self.vae.dtype)
        if output_type == "latent":
            image = latents
            has_nsfw_concept = None
        elif output_type == "pil":
            # 8. Post-processing
            image = self.decode_latents(latents)

            # 9. Run safety checker
            image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)

            # 10. Convert to PIL
            image = self.numpy_to_pil(image)
        else:
            # 8. Post-processing
            image = self.decode_latents(latents)

            # 9. Run safety checker
            image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)

    def text2img_default_height_width(self, height, width, image):
        if image is None:
            return height, width
        
        height = None 
        width = None

        if isinstance(image, list):
            image = image[0]

        if height is None:
            if isinstance(image, PIL.Image.Image):
                height = image.height
            elif isinstance(image, torch.Tensor):
                height = image.shape[3]

            height = (height // 8) * 8  # round down to nearest multiple of 8

        if width is None:
            if isinstance(image, PIL.Image.Image):
                width = image.width
            elif isinstance(image, torch.Tensor):
                width = image.shape[2]

            width = (width // 8) * 8  # round down to nearest multiple of 8

        return height, width

    def text2img_prepare_latents(self, image, timestep, batch_size, num_images_per_prompt, height, width,dtype, device, generator=None, use_last_noise=False, noise_type='random'):

        if image is None:
            shape = (
                batch_size * num_images_per_prompt,
                self.unet.in_channels,
                height // self.vae_scale_factor,
                width // self.vae_scale_factor,
            )
            # latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

            if use_last_noise and hasattr(self, 'last_noise'):
                noise = self.last_noise
            else:
                noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
                if noise_type == 'symmetry':
                    bs, c, h, w = noise.shape
                    noise[:, :, :, w//2:] = torch.flip(noise[:, :, :, :w//2], dims=[3])
            if not hasattr(self, 'last_noise'):
                self.last_noise = noise
            latents = noise

            return latents
        else:
            if not isinstance(image, (torch.Tensor, PIL.Image.Image, list)):
                raise ValueError(
                    f"`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}"
                )

            image = image.to(device=device, dtype=dtype)

            batch_size = batch_size * num_images_per_prompt
            if isinstance(generator, list) and len(generator) != batch_size:
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )

            if isinstance(generator, list):
                init_latents = [
                    self.vae.encode(image[i : i + 1]).latent_dist.sample(generator[i]) for i in range(batch_size)
                ]
                init_latents = torch.cat(init_latents, dim=0)
            else:
                init_latents = self.vae.encode(image).latent_dist.sample(generator)

            init_latents = self.vae.config.scaling_factor * init_latents

            if batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] == 0:
                raise ValueError(
                    f"Cannot duplicate `image` of batch size {init_latents.shape[0]} to {batch_size} text prompts."
                )
            else:
                init_latents = torch.cat([init_latents], dim=0)

            shape = init_latents.shape
            noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

            # get latents
            init_latents = self.scheduler.add_noise(init_latents, noise, timestep)
            latents = init_latents

            return latents

    @torch.no_grad()
    def text2img(
        self,
        prompt: Union[str, List[str]] = None,
        controlnet_conditioning=[],
        latent_couple_conditioning=[],
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        max_embeddings_multiples=3,
        use_last_noise=False,
        noise_type='random',
    ):
        
        # 1. Default height and width to unet
        height, width = self.text2img_default_height_width(
            height, 
            width, 
            controlnet_conditioning[0]['control_image'] if len(controlnet_conditioning) > 0 else None
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        dtype = self.unet.dtype
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            max_embeddings_multiples,
        )
        encoder_hidden_states = prompt_embeds

        # latent couple prompt
        if len(latent_couple_conditioning) > 0:
            lc_prompt = [lc['prompt'] for lc in latent_couple_conditioning]
            lc_prompt_embeds = self._encode_prompt(
                prompt=lc_prompt,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                do_classifier_free_guidance=False,
                negative_prompt=None,
                max_embeddings_multiples=max_embeddings_multiples,
            )
            # nb_repeat = int(np.ceil(max(encoder_hidden_states.shape[1], lc_prompt_embeds.shape[1]) / min(encoder_hidden_states.shape[1], lc_prompt_embeds.shape[1])))
            # if encoder_hidden_states.shape[1] > lc_prompt_embeds.shape[1]:
            #     lc_prompt_embeds = torch.cat([lc_prompt_embeds] * nb_repeat, dim=1)
            # else:
            #     encoder_hidden_states = torch.cat([encoder_hidden_states] * nb_repeat, dim=1)
            # length_prompt = min(encoder_hidden_states.shape[1], lc_prompt_embeds.shape[1])
            # lc_prompt_embeds = lc_prompt_embeds[:, :length_prompt]
            # encoder_hidden_states = encoder_hidden_states[:, :length_prompt]
            # encoder_hidden_states = torch.cat([encoder_hidden_states[0:1], lc_prompt_embeds, encoder_hidden_states[1:2]])
        
            lc_encoder_hidden_states = lc_prompt_embeds

        # latent couple mask and control
        # lc_controlnet_index_list = []
        for idx_lc in range(len(latent_couple_conditioning)):
            mask_image_raw = latent_couple_conditioning[idx_lc]['mask']
            latent_couple_conditioning[idx_lc]['mask'] = self._prepare_latent_couple_mask(mask_image_raw, height, width, device, dtype)
            # latent_couple_conditioning[idx_lc]['attention_mask'] = self._prepare_latent_couple_attention_mask(latent_couple_conditioning[idx_lc]['mask'], lc_encoder_hidden_states.shape[1])
            # latent_couple_conditioning[idx_lc]['attention_probs_weight'] = self._prepare_latent_couple_attention_probs_weight(latent_couple_conditioning[idx_lc]['mask'], lc_encoder_hidden_states.shape[1])
            if 'controlnet_conditioning' in latent_couple_conditioning[idx_lc]:
                for idx_lc_cc in range(len(latent_couple_conditioning[idx_lc]['controlnet_conditioning'])):
                    latent_couple_conditioning[idx_lc]['controlnet_conditioning'][idx_lc_cc]['image'] = prepare_controlnet_conditioning_image(
                        latent_couple_conditioning[idx_lc]['controlnet_conditioning'][idx_lc_cc]['image'],
                        width,
                        height,
                        batch_size * num_images_per_prompt,
                        num_images_per_prompt,
                        device,
                        self.controlnet_list[0].dtype,
                    )
                    # lc_controlnet_index_list.append(latent_couple_conditioning[idx_lc]['control']['index'])
                    if 'visual_emb' in latent_couple_conditioning[idx_lc]['controlnet_conditioning'][idx_lc_cc]:
                        face_emb = torch.tensor(latent_couple_conditioning[idx_lc]['controlnet_conditioning'][idx_lc_cc]['visual_emb']).to(device=device, dtype=dtype)
                        face_emb = face_emb.reshape([-1, 1, 512])
                        latent_couple_conditioning[idx_lc]['controlnet_conditioning'][idx_lc_cc]['visual_emb'] = face_emb
        # lc_controlnet_index_list = np.unique(lc_controlnet_index_list).tolist()




        # 4. Prepare mask, image, and controlnet_conditioning_image
        # if image is not None:
        #     image = prepare_image(image)

        # Prepare control_image
        for idx in range(len(controlnet_conditioning)):
            cc_info = controlnet_conditioning[idx]
            cc_info['control_image'] = prepare_controlnet_conditioning_image(
                cc_info['control_image'],
                width,
                height,
                batch_size * num_images_per_prompt,
                num_images_per_prompt,
                device,
                self.controlnet_list[0].dtype,
            )
            if 'control_visual_emb' in cc_info:
                face_emb = torch.tensor(cc_info['control_visual_emb']).to(device=device, dtype=dtype)
                face_emb = face_emb.reshape([-1, 1, 512])
                cc_info['control_visual_emb'] = face_emb

        # 5. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, 1., device)
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)

        # 6. Prepare latent variables
        latents = self.text2img_prepare_latents(
            None,
            latent_timestep,
            batch_size,
            num_images_per_prompt,
            height,
            width,
            self.vae.dtype,
            device,
            generator,
            use_last_noise,
            noise_type,
        )
        latents = latents.to(self.unet.dtype)

        if do_classifier_free_guidance:
            for idx in range(len(controlnet_conditioning)):
                cc_info = controlnet_conditioning[idx]

                # control_image
                control_image_list = [cc_info['control_image']]
                # for idx_lc in range(len(latent_couple_conditioning)):
                #     control_image_list.append(cc_info['control_image'])
                control_image_list.append(cc_info['control_image'])
                cc_info['control_image'] = torch.cat(control_image_list)

                # control_visual_emb
                if 'control_visual_emb' in cc_info:
                    if 'control_visual_emb_neg' in cc_info:
                        control_visual_emb_neg = cc_info['control_visual_emb_neg']
                        control_visual_emb_neg = torch.tensor(control_visual_emb_neg).to(device=device, dtype=dtype)
                        control_visual_emb_neg = control_visual_emb_neg.reshape([-1, 1, 512])
                    else:
                        control_visual_emb_neg = torch.zeros_like(cc_info['control_visual_emb'])
                    control_visual_emb_list = [control_visual_emb_neg]
                    # for idx_lc in range(len(latent_couple_conditioning)):
                    #     control_visual_emb_list.append(cc_info['control_visual_emb'])
                    control_visual_emb_list.append(cc_info['control_visual_emb'])
                    cc_info['control_visual_emb'] = torch.cat(control_visual_emb_list)

        # latent couple control
        # lc_controlnet_conditioning = []
        # for lc_controlnet_index in lc_controlnet_index_list:
        #     # control_image
        #     lc_controlnet_conditioning_image = torch.zeros([batch_size, 3, height, width], dtype=dtype, device=device)
        #     nb_repeat = 1 + len(latent_couple_conditioning) + 1 if do_classifier_free_guidance else 1 + len(latent_couple_conditioning)
        #     lc_controlnet_conditioning_image = torch.repeat_interleave(lc_controlnet_conditioning_image, nb_repeat, dim=0)
        #     for idx_lc in range(len(latent_couple_conditioning)):
        #         if not ('control' in latent_couple_conditioning[idx_lc] and latent_couple_conditioning[idx_lc]['control']['index'] == lc_controlnet_index):
        #             continue
        #         lc_controlnet_conditioning_image[1+idx_lc:1+idx_lc+1] = latent_couple_conditioning[idx_lc]['control']['image']
            
        #     # control_index
        #     lc_controlnet_index = lc_controlnet_index

        #     # control_weight
        #     lc_control_weight = latent_couple_conditioning[idx_lc]['control']['weight']

        #     cc_info = dict(
        #         control_image=lc_controlnet_conditioning_image,
        #         control_index=lc_controlnet_index,
        #         control_weight=lc_control_weight,
        #     )
        #     lc_controlnet_conditioning.append(cc_info)


        # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # controlnet
                if len(controlnet_conditioning) > 0:
                    down_block_res_samples_list, mid_block_res_sample_list = [], []
                    for cc_info in controlnet_conditioning:
                        controlnet_conditioning_image = cc_info['control_image']
                        controlnet = self.controlnet_list[int(cc_info['control_index'])]
                        controlnet_conditioning_scale = cc_info['control_weight']
                        if 'control_visual_emb' in cc_info:
                            controlnet_encoder_hidden_states = controlnet.visual_embedding(cc_info['control_visual_emb'])
                        else:
                            controlnet_encoder_hidden_states = encoder_hidden_states
                        down_block_res_samples, mid_block_res_sample = controlnet(
                            latent_model_input,
                            t,
                            encoder_hidden_states=controlnet_encoder_hidden_states,
                            controlnet_cond=controlnet_conditioning_image,
                            return_dict=False,
                        )
                        down_block_res_samples = [
                            down_block_res_sample * controlnet_conditioning_scale
                            for down_block_res_sample in down_block_res_samples
                        ]
                        mid_block_res_sample *= controlnet_conditioning_scale
                        down_block_res_samples_list.append(down_block_res_samples)
                        mid_block_res_sample_list.append(mid_block_res_sample)

                    down_block_res_samples, mid_block_res_sample = [0]*len(down_block_res_samples_list[0]), 0
                    for idx_controlnet in range(len(down_block_res_samples_list)):
                        need_permute = False
                        if isinstance(mid_block_res_sample, torch.Tensor):
                            _,_,h1,w1 = mid_block_res_sample_list[idx_controlnet].shape
                            _,_,h2,w2 = mid_block_res_sample.shape
                            if h1 != h2 and w1 != w2:
                                need_permute = True
                        if need_permute:
                            mid_block_res_sample = torch.permute(mid_block_res_sample, [0,2,3,1]) + torch.permute(mid_block_res_sample_list[idx_controlnet], [0,2,3,1])
                            mid_block_res_sample = torch.permute(mid_block_res_sample, [0,3,1,2])
                        else:
                            mid_block_res_sample += mid_block_res_sample_list[idx_controlnet]
                        for idx_dbrs in range(len(down_block_res_samples_list[idx_controlnet])):
                            need_permute = False
                            if isinstance(down_block_res_samples[idx_dbrs], torch.Tensor):
                                _,_,h1,w1 = down_block_res_samples_list[idx_controlnet][idx_dbrs].shape
                                _,_,h2,w2 = down_block_res_samples[idx_dbrs].shape
                                if h1 != h2 and w1 != w2:
                                    need_permute = True
                            if need_permute:
                                down_block_res_samples[idx_dbrs] = torch.permute(down_block_res_samples[idx_dbrs], [0,2,3,1]) + torch.permute(down_block_res_samples_list[idx_controlnet][idx_dbrs], [0,2,3,1])
                                down_block_res_samples[idx_dbrs] = torch.permute(down_block_res_samples[idx_dbrs], [0,3,1,2])
                            else:
                                down_block_res_samples[idx_dbrs] += down_block_res_samples_list[idx_controlnet][idx_dbrs]
                else:
                    down_block_res_samples, mid_block_res_sample = [0] * 12, 0

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                ).sample

                # latent couple every sample
                if len(latent_couple_conditioning) > 0:
                    lc_noise_pred_list = []
                    for idx_lc in range(len(latent_couple_conditioning)):
                        lcc_info = latent_couple_conditioning[idx_lc]
                        lc_controlnet_conditioning = lcc_info.get('controlnet_conditioning', [])
                        # lc_attention_probs_weight = lcc_info['attention_probs_weight']

                        # controlnet
                        if len(lc_controlnet_conditioning) > 0:
                            lc_down_block_res_samples_list, lc_mid_block_res_sample_list = [], []
                            for cc_info in lc_controlnet_conditioning:
                                controlnet_conditioning_image = cc_info['image']
                                controlnet = self.controlnet_list[int(cc_info['index'])]
                                controlnet_conditioning_scale = cc_info['weight']
                                if 'visual_emb' in cc_info:
                                    controlnet_encoder_hidden_states = controlnet.visual_embedding(cc_info['visual_emb'])
                                else:
                                    controlnet_encoder_hidden_states = lc_encoder_hidden_states[idx_lc:idx_lc+1]
                                lc_down_block_res_samples, lc_mid_block_res_sample = controlnet(
                                    latent_model_input[0:1],
                                    t,
                                    encoder_hidden_states=controlnet_encoder_hidden_states,
                                    controlnet_cond=controlnet_conditioning_image,
                                    return_dict=False,
                                )
                                lc_down_block_res_samples = [
                                    down_block_res_sample * controlnet_conditioning_scale
                                    for down_block_res_sample in lc_down_block_res_samples
                                ]
                                lc_mid_block_res_sample *= controlnet_conditioning_scale
                                lc_down_block_res_samples_list.append(lc_down_block_res_samples)
                                lc_mid_block_res_sample_list.append(lc_mid_block_res_sample)

                            lc_down_block_res_samples, lc_mid_block_res_sample = [0] * len(lc_down_block_res_samples_list[0]), 0
                            for idx_controlnet in range(len(lc_down_block_res_samples_list)):
                                lc_mid_block_res_sample += lc_mid_block_res_sample_list[idx_controlnet]
                                for idx_dbrs in range(len(lc_down_block_res_samples_list[idx_controlnet])):
                                    lc_down_block_res_samples[idx_dbrs] += lc_down_block_res_samples_list[idx_controlnet][idx_dbrs]
                        else:
                            lc_down_block_res_samples, lc_mid_block_res_sample = [0] * 12, 0

                        lc_noise_pred = self.unet(
                            latent_model_input[0:1],
                            t,
                            encoder_hidden_states=lc_encoder_hidden_states[idx_lc:idx_lc+1],
                            cross_attention_kwargs=cross_attention_kwargs,
                            down_block_additional_residuals=lc_down_block_res_samples,
                            mid_block_additional_residual=lc_mid_block_res_sample,
                            # attention_probs_weight=lc_attention_probs_weight,
                        ).sample
                        lc_noise_pred_list.append(lc_noise_pred)


                # perform guidance
                if do_classifier_free_guidance:
                    if len(latent_couple_conditioning) > 0:
                        noise_pred_uncond = noise_pred[0:1]
                        noise_preds = [noise_pred[1:2]] + lc_noise_pred_list  # prompt, lc1, lc2 ....
                        noise_preds = torch.cat(noise_preds)
                        noise_preds = noise_pred_uncond + guidance_scale * (noise_preds - noise_pred_uncond)

                        background_weight = 1 - np.mean([lcc['weight'] for lcc in latent_couple_conditioning])
                        result = noise_preds[0:1] * background_weight
                        mask_empty = torch.ones_like(latent_couple_conditioning[0]['mask'])
                        for idx_lc in range(1, len(noise_preds)):
                            weight_idx = latent_couple_conditioning[idx_lc-1]['weight']
                            mask_idx = latent_couple_conditioning[idx_lc-1]['mask']

                            # 当前lc和历史lc的交集区域
                            mask_intersection = mask_idx * (1 - mask_empty)

                            result += noise_preds[idx_lc:idx_lc+1] * weight_idx * mask_idx * (1 - mask_intersection)
                            result = (0.5 * result + 0.5 * (weight_idx * noise_preds[idx_lc:idx_lc+1] + (1 - weight_idx) * noise_preds[0:1])) * mask_intersection + result * (1 - mask_intersection)

                            # 没被lc覆盖的背景区域
                            mask_empty = mask_empty * (1 - mask_idx)

                            # result += noise_preds[idx_lc:idx_lc+1] * weight_idx * mask_idx
                        result += noise_preds[0:1] * (1 - background_weight) * mask_empty
                        noise_pred = result
                    else:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)


        # If we do sequential model offloading, let's offload unet and controlnet
        # manually for max memory savings
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.unet.to("cpu")
            self.controlnet.to("cpu")
            torch.cuda.empty_cache()

        latents = latents.to(self.vae.dtype)

        if output_type == "latent":
            image = latents
            has_nsfw_concept = None
        elif output_type == "pil":
            # 8. Post-processing
            image = self.decode_latents(latents)

            # 9. Run safety checker
            image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)

            # 10. Convert to PIL
            image = self.numpy_to_pil(image)
        else:
            # 8. Post-processing
            image = self.decode_latents(latents)

            # 9. Run safety checker
            image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)

    def inpatinting_prepare_latents(self, image, timestep, batch_size, num_images_per_prompt, dtype, device, generator=None, init_mask=None):
        if not isinstance(image, (torch.Tensor, PIL.Image.Image, list)):
            raise ValueError(
                f"`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}"
            )

        image = image.to(device=device, dtype=dtype)

        batch_size = batch_size * num_images_per_prompt
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if isinstance(generator, list):
            init_latents = [
                self.vae.encode(image[i : i + 1]).latent_dist.sample(generator[i]) for i in range(batch_size)
            ]
            init_latents = torch.cat(init_latents, dim=0)
        else:
            init_latents = self.vae.encode(image).latent_dist.sample(generator)

        init_latents = self.vae.config.scaling_factor * init_latents

        if batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] == 0:
            raise ValueError(
                f"Cannot duplicate `image` of batch size {init_latents.shape[0]} to {batch_size} text prompts."
            )
        else:
            init_latents = torch.cat([init_latents], dim=0)
        init_latents_orig = init_latents
        
        shape = init_latents.shape
        noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

        # get latents
        init_latents = self.scheduler.add_noise(init_latents, noise, timestep)
        latents = init_latents
        
        return latents, init_latents_orig, noise

    @torch.no_grad()
    def inpainting(
        self,
        prompt: Union[str, List[str]] = None,
        image: Union[torch.Tensor, PIL.Image.Image] = None,
        mask_image: Union[torch.Tensor, PIL.Image.Image] = None,
        controlnet_conditioning_image: Union[
            torch.FloatTensor, PIL.Image.Image, List[torch.FloatTensor], List[PIL.Image.Image]
        ] = None,
        strength: float = 0.8,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
        controlnet_conditioning=[],
        max_embeddings_multiples=3,
        add_predicted_noise_type='neg',
    ):

        # 0. Default height and width to unet
        height, width = self._default_height_width(height, width, image)

        # 1. Check inputs. Raise error if not correct
        # self.check_inputs(
        #     prompt,
        #     image,
        #     mask_image,
        #     controlnet_conditioning_image,
        #     height,
        #     width,
        #     callback_steps,
        #     negative_prompt,
        #     prompt_embeds,
        #     negative_prompt_embeds,
        #     strength,
        # )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        dtype = self.unet.dtype
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0


        # 3. Encode input prompt
        prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            max_embeddings_multiples,
        )
        encoder_hidden_states = prompt_embeds

        # 4. Prepare image, and controlnet_conditioning_image
        image = prepare_image(image) # torch.Size([1, 3, H, W])

        # Prepare control_image
        for idx in range(len(controlnet_conditioning)):
            cc_info = controlnet_conditioning[idx]
            cc_info['control_image'] = prepare_controlnet_conditioning_image(
                cc_info['control_image'],
                width,
                height,
                batch_size * num_images_per_prompt,
                num_images_per_prompt,
                device,
                self.controlnet_list[0].dtype,
            )
            if 'control_visual_emb' in cc_info:
                face_emb = torch.tensor(cc_info['control_visual_emb']).to(device=device, dtype=dtype)
                face_emb = face_emb.reshape([-1, 1, 512])
                cc_info['control_visual_emb'] = face_emb

        # 5. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength, device)
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)

        # 6. Prepare latent variables
        latents, init_latents_orig, noise = self.inpatinting_prepare_latents(
            image,
            latent_timestep,
            batch_size,
            num_images_per_prompt,
            self.vae.dtype,
            device,
            generator,
            mask_image
        ) # torch.Size([1, 4, H//60, W//60])    
        latents = latents.to(dtype)

        if do_classifier_free_guidance:
            for idx in range(len(controlnet_conditioning)):
                cc_info = controlnet_conditioning[idx]
                cc_info['control_image'] = torch.cat([cc_info['control_image']] * 2)
                if 'control_visual_emb' in cc_info:
                    cc_info['control_visual_emb'] = torch.cat([torch.zeros_like(cc_info['control_visual_emb']), cc_info['control_visual_emb']])

        # 6.5 Prepare image
        init_latent = latents.clone()
        init_mask = mask_image
        latmask = init_mask.convert('RGB').resize((latents.shape[3], latents.shape[2]))
        latmask = np.moveaxis(np.array(latmask, dtype=np.float32), 2, 0) / 255
        latmask = latmask[0]
        latmask = np.around(latmask)
        latmask = np.tile(latmask[None], (4, 1, 1))

        mask = torch.asarray(1.0 - latmask).to(latents.device).type(self.vae.dtype) # torch.Size([4, H//60, W//60])        
        nmask = torch.asarray(latmask).to(latents.device).type(self.vae.dtype) # torch.Size([4, H//60, W//60])   

        # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                non_inpainting_latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                non_inpainting_latent_model_input = self.scheduler.scale_model_input(non_inpainting_latent_model_input, t)

                if len(controlnet_conditioning) > 0:
                    down_block_res_samples_list, mid_block_res_sample_list = [], []
                    for cc_info in controlnet_conditioning:
                        controlnet_conditioning_image = cc_info['control_image']
                        controlnet = self.controlnet_list[int(cc_info['control_index'])]
                        controlnet_conditioning_scale = cc_info['control_weight']
                        if 'control_visual_emb' in cc_info:
                            controlnet_encoder_hidden_states = controlnet.visual_embedding(cc_info['control_visual_emb'])
                        else:
                            controlnet_encoder_hidden_states = prompt_embeds
                        down_block_res_samples, mid_block_res_sample = controlnet(
                            non_inpainting_latent_model_input,
                            t,
                            encoder_hidden_states=controlnet_encoder_hidden_states,
                            controlnet_cond=controlnet_conditioning_image,
                            return_dict=False,
                        )
                        down_block_res_samples = [
                            down_block_res_sample * controlnet_conditioning_scale
                            for down_block_res_sample in down_block_res_samples
                        ]
                        mid_block_res_sample *= controlnet_conditioning_scale
                        down_block_res_samples_list.append(down_block_res_samples)
                        mid_block_res_sample_list.append(mid_block_res_sample)

                    down_block_res_samples, mid_block_res_sample = [0]*len(down_block_res_samples_list[0]), 0
                    for idx_controlnet in range(len(down_block_res_samples_list)):
                        need_permute = False
                        if isinstance(mid_block_res_sample, torch.Tensor):
                            _,_,h1,w1 = mid_block_res_sample_list[idx_controlnet].shape
                            _,_,h2,w2 = mid_block_res_sample.shape
                            if h1 != h2 and w1 != w2:
                                need_permute = True
                        if need_permute:
                            mid_block_res_sample = torch.permute(mid_block_res_sample, [0,2,3,1]) + torch.permute(mid_block_res_sample_list[idx_controlnet], [0,2,3,1])
                            mid_block_res_sample = torch.permute(mid_block_res_sample, [0,3,1,2])
                        else:
                            mid_block_res_sample += mid_block_res_sample_list[idx_controlnet]
                        for idx_dbrs in range(len(down_block_res_samples_list[idx_controlnet])):
                            need_permute = False
                            if isinstance(down_block_res_samples[idx_dbrs], torch.Tensor):
                                _,_,h1,w1 = down_block_res_samples_list[idx_controlnet][idx_dbrs].shape
                                _,_,h2,w2 = down_block_res_samples[idx_dbrs].shape
                                if h1 != h2 and w1 != w2:
                                    need_permute = True
                            if need_permute:
                                down_block_res_samples[idx_dbrs] = torch.permute(down_block_res_samples[idx_dbrs], [0,2,3,1]) + torch.permute(down_block_res_samples_list[idx_controlnet][idx_dbrs], [0,2,3,1])
                                down_block_res_samples[idx_dbrs] = torch.permute(down_block_res_samples[idx_dbrs], [0,3,1,2])
                            else:
                                down_block_res_samples[idx_dbrs] += down_block_res_samples_list[idx_controlnet][idx_dbrs]
                else:
                    down_block_res_samples, mid_block_res_sample = [0] * 12, 0

                # predict the noise residual
                noise_pred = self.unet(
                    non_inpainting_latent_model_input,
                    t,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                ).sample

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                
                #noise_pred = init_latent * mask + nmask * noise_pred
                
                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # masking
                if add_predicted_noise_type == 'pos':
                    init_latents_proper = self.scheduler.add_noise(init_latents_orig, noise_pred_text, torch.tensor([t]))
                elif add_predicted_noise_type == 'neg':
                    init_latents_proper = self.scheduler.add_noise(init_latents_orig, noise_pred_uncond, torch.tensor([t]))
                else:
                    init_latents_proper = self.scheduler.add_noise(init_latents_orig, noise, torch.tensor([t]))

                latents = init_latents_proper.to(latents.dtype) * mask.to(latents.dtype) + latents * nmask.to(latents.dtype)
                latents = latents.to(dtype)
                
                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)
                
                # init_latent: torch.Size([1, 4, 60, 135])
                # mask: torch.Size([4, 60, 135])
                # nmask: torch.Size([4, 60, 135])
                # latents: torch.Size([1, 4, 60, 135])
                #latents = init_latent * mask + nmask * latents
                #print(mask.max(), mask.min())

        # use original latents corresponding to unmasked portions of the image
        latents = latents.to(self.vae.dtype)
        latents = init_latents_orig * mask + latents * nmask
        # latents = init_latents_orig
        
        # If we do sequential model offloading, let's offload unet and controlnet
        # manually for max memory savings
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.unet.to("cpu")
            self.controlnet.to("cpu")
            torch.cuda.empty_cache()

        if output_type == "latent":
            image = latents
            has_nsfw_concept = None
        elif output_type == "pil":
            # 8. Post-processing
            image = self.decode_latents(latents)

            # 9. Run safety checker
            image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)

            # 10. Convert to PIL
            image = self.numpy_to_pil(image)
        else:
            # 8. Post-processing
            image = self.decode_latents(latents)

            # 9. Run safety checker
            image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)

