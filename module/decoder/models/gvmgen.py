# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Main model for using GVMGen. This will combine all the required components
and provide easy access to the generation API.
"""

import typing as tp
import warnings

import torch

from .encodec import CompressionModel
from .genmodel import BaseGenModel

from .lm import LMModel
from .builders import get_debug_compression_model, get_debug_lm_model
from .loaders import load_compression_model, load_lm_model
from ..data.audio_utils import convert_audio
from ..modules.conditioners import ConditioningAttributes, WavCondition


MelodyList = tp.List[tp.Optional[torch.Tensor]]
MelodyType = tp.Union[torch.Tensor, MelodyList]

import bitsandbytes as bnb

def convert_to_linear8bit(model):
    for name, module in model.named_children():
        # Nếu gặp Linear → thay bằng Linear8bitLt
        if isinstance(module, torch.nn.Linear):
            in_f = module.in_features
            out_f = module.out_features
            bias = module.bias is not None

            new_linear = bnb.nn.Linear8bitLt(
                in_f,
                out_f,
                bias=bias,
            )

            # copy weight/bias từ Linear cũ
            new_linear.weight.data = module.weight.data.clone()
            if bias:
                new_linear.bias.data = module.bias.data.clone()

            setattr(model, name, new_linear)

        else:
            # Đệ quy để đi sâu vào model
            convert_to_linear8bit(module)

    return model

def convert_to_linear4bit(model, device):
    for name, module in model.named_children():
        # Nếu gặp Linear → thay bằng Linear8bitLt
        if isinstance(module, torch.nn.Linear):
            in_f = module.in_features
            out_f = module.out_features
            bias = module.bias is not None

            new_linear = bnb.nn.Linear4bit(
                in_f,
                out_f,
                bias=bias,
            ).to(device)

            # copy weight/bias từ Linear cũ
            new_linear.weight.data = module.weight.data.clone()
            if bias:
                new_linear.bias.data = module.bias.data.clone()

            setattr(model, name, new_linear)

        else:
            # Đệ quy để đi sâu vào model
            convert_to_linear4bit(module, device)

    return model

def _is_linear_like(module):
    # Các kiểu linear hay gặp: nn.Linear, NonDynamicallyQuantizableLinear (torch internal),
    # hoặc Conv1D của transformers (OpenAI-style)
    name = module.__class__.__name__
    return isinstance(module, torch.nn.Linear) \
        or name == "NonDynamicallyQuantizableLinear" \
        or name == "Conv1D" \
        or isinstance(module, torch.nn.modules.linear.NonDynamicallyQuantizableLinear) if hasattr(torch.nn.modules.linear, "NonDynamicallyQuantizableLinear") else False

def convert_module_to_4bit(module, device="cuda", verbose=False):
    """
    Convert all linear-like submodules of `module` to bitsandbytes Linear4bit, in-place.
    Returns module.
    """
    for child_name, child in list(module.named_children()):
        # if already bnb Linear4bit -> skip
        if isinstance(child, bnb.nn.Linear4bit):
            if verbose:
                print(f"[skip] {child_name}: already Linear4bit")
            continue

        # detect linear-like
        if _is_linear_like(child):
            in_f = getattr(child, "in_features", None)
            out_f = getattr(child, "out_features", None)

            # Some Conv1D/OpenAI GPT layers use attributes like .weight with shape (out, in)
            # try to infer from weight if in/out None
            wt = getattr(child, "weight", None)
            if wt is None:
                if verbose:
                    print(f"[warn] {child_name} has no weight; skipping")
                continue

            # weight.shape convention: (out_features, in_features) for Linear
            w_shape = tuple(wt.data.shape)
            if in_f is None or out_f is None:
                # try infer
                if len(w_shape) == 2:
                    out_f, in_f = w_shape
                else:
                    # unexpected shape: skip
                    if verbose:
                        print(f"[warn] {child_name} weight shape {w_shape} unexpected; skipping")
                    continue

            bias = getattr(child, "bias", None) is not None

            if verbose:
                print(f"[convert] {child_name}: {child.__class__.__name__} -> Linear4bit (in={in_f}, out={out_f}, bias={bias})")

            # create new Linear4bit
            try:
                new_linear = bnb.nn.Linear4bit(
                    in_f,
                    out_f,
                    bias=bias,
                    quant_type='nf4',              # recommended
                    compress_statistics=True,
                )
            except Exception as e:
                # fallback: try without compress_statistics
                try:
                    new_linear = bnb.nn.Linear4bit(in_f, out_f, bias=bias, quant_type='nf4')
                except Exception as e2:
                    raise RuntimeError(f"Không thể tạo bnb.nn.Linear4bit: {e} / {e2}")

            # move new module to device
            try:
                new_linear.to(device)
            except Exception:
                # move later when whole model to device
                pass

            # quantize weight from float -> Params4bit if API exists
            # Prefer: bnb.nn.Params4bit.from_float(...) or bnb.nn.Params4bit(...)
            assigned = False
            with torch.no_grad():
                # Ensure source weight is contiguous CPU float
                w_float = wt.data.clone().contiguous().to("cpu")

                # Try Params4bit.from_float
                if hasattr(bnb.nn, "Params4bit") and hasattr(bnb.nn.Params4bit, "from_float"):
                    try:
                        params4 = bnb.nn.Params4bit.from_float(w_float)
                        # assign params object to new_linear.weight (some versions accept weight attr)
                        try:
                            new_linear.weight = params4
                            assigned = True
                        except Exception:
                            # try setting internal .weight.data if present
                            if hasattr(new_linear, "weight") and isinstance(new_linear.weight, torch.Tensor):
                                new_linear.weight.data = params4.to(new_linear.weight.data.device)  # unlikely but try
                                assigned = True
                    except Exception as e:
                        if verbose:
                            print(f"[info] Params4bit.from_float failed: {e}")

                # Some versions expose helper to convert: bnb.nn.Linear4bit.from_float
                if not assigned and hasattr(bnb.nn.Linear4bit, "from_float"):
                    try:
                        # returns a Linear4bit instance already quantized (some APIs)
                        new_linear_from = bnb.nn.Linear4bit.from_float(child)  # if available
                        new_linear_from.to(device)
                        new_linear = new_linear_from
                        assigned = True
                    except Exception as e:
                        if verbose:
                            print(f"[info] Linear4bit.from_float failed: {e}")

                # Final fallback: try to set raw weights (may not be valid for bnb4bit)
                if not assigned:
                    try:
                        # some bnb Linear4bit store .weight.data as placeholder; try to copy raw then call .pack or similar
                        if isinstance(new_linear.weight, torch.Tensor):
                            # copy & hope bitsandbytes will handle (less safe)
                            new_linear.weight.data = w_float.to(new_linear.weight.data.device, dtype=new_linear.weight.data.dtype)
                            assigned = True
                            if verbose:
                                print(f"[warn] Fallback: copied raw FP32 into Linear4bit.weight (may be incorrect for some bnb versions).")
                    except Exception as e:
                        if verbose:
                            print(f"[error] fallback assign weight failed: {e}")

                # bias copy (convert dtype/device)
                if bias and hasattr(new_linear, "bias") and new_linear.bias is not None:
                    try:
                        new_linear.bias.data = child.bias.data.clone().to(new_linear.bias.data.device, dtype=new_linear.bias.data.dtype)
                    except Exception:
                        try:
                            new_linear.bias = torch.nn.Parameter(child.bias.data.clone().to(new_linear.bias.device))
                        except Exception:
                            if verbose:
                                print(f"[warn] couldn't copy bias for {child_name}")

            # replace module
            setattr(module, child_name, new_linear)

        else:
            # recursion
            convert_module_to_4bit(child, device=device, verbose=verbose)

    return module


def convert_model_to_4bit(model, device="cuda", verbose=False, exclude_substrings=None, include_only=None):
    """
    High-level wrapper.
    - model: torch.nn.Module
    - device: 'cuda' or 'cpu' or 'cuda:0'
    - exclude_substrings: list of name substrings; if a module path contains any of these, it will be skipped (useful to skip CLIP.visual)
    - include_only: list of name substrings; if provided, only module paths containing one of these will be converted
    """
    # optional filtering wrapper: we iterate through top-level children and decide skip/include
    if exclude_substrings is None:
        exclude_substrings = []
    if include_only is not None:
        include_only = list(include_only)

    def _should_skip(module_path):
        for s in exclude_substrings:
            if s and s in module_path:
                return True
        if include_only is None:
            return False
        # if include_only specified, skip unless any include substring matches
        for s in include_only:
            if s and s in module_path:
                return False
        return True

    # We'll walk model.named_modules to find top-level submodules to convert
    # but keep conversion local by calling convert_module_to_4bit on each top-level child
    for name, child in list(model.named_children()):
        full_path = name
        if _should_skip(full_path):
            if verbose:
                print(f"[skip top] {full_path} (excluded)")
            continue
        if include_only is not None:
            # if include_only specified but this top-level child not matched -> skip
            matched = any(s in full_path for s in include_only)
            if not matched:
                if verbose:
                    print(f"[skip top] {full_path} (not in include_only)")
                continue

        try:
            convert_module_to_4bit(child, device=device, verbose=verbose)
            # set back (child modified in place)
            setattr(model, name, child)
        except Exception as e:
            print(f"[error] converting top-level module {name}: {e}")

    # move whole model to device (recommended)
    try:
        model.to(device)
    except Exception:
        if verbose:
            print(f"[warn] couldn't move model to {device} automatically. Move manually.")

    return model

class GVMGen(BaseGenModel):
    """GVMGen main model with convenient generation API.

    Args:
        name (str): name of the model.
        compression_model (CompressionModel): Compression model
            used to map audio to invertible discrete representations.
        lm (LMModel): Language model over discrete representations.
        max_duration (float, optional): maximum duration the model can produce,
            otherwise, inferred from the training params.
    """
    def __init__(self, name: str, compression_model: CompressionModel, lm: LMModel,
                 max_duration: tp.Optional[float] = None, cond_type: str = 'text'):
        super().__init__(name, compression_model, lm, max_duration, cond_type)
        self.set_generation_params(duration=15)  # default duration

    @staticmethod
    def get_pretrained(name: str = 'gvmgen', device=None):
        
        if device is None:
            if torch.cuda.device_count():
                device = 'cuda'
            else:
                device = 'cpu'

        lm = convert_model_to_4bit(load_lm_model(name, device='cuda:0'), 'cuda:0')
        print(lm)
        compression_model = load_compression_model(name, device='cuda:1')
        if 'self_wav' in lm.condition_provider.conditioners:
            lm.condition_provider.conditioners['self_wav'].match_len_on_eval = True
            lm.condition_provider.conditioners['self_wav']._use_masking = False

        return GVMGen(name, compression_model, lm, cond_type=lm.condition_provider.cond_type)

    def set_generation_params(self, use_sampling: bool = True, top_k: int = 250,
                              top_p: float = 0.0, temperature: float = 1.0,
                              duration: float = 30.0, cfg_coef: float = 3.0,
                              two_step_cfg: bool = False, extend_stride: float = 18):
        """Set the generation parameters for GVMGen.

        Args:
            use_sampling (bool, optional): Use sampling if True, else do argmax decoding. Defaults to True.
            top_k (int, optional): top_k used for sampling. Defaults to 250.
            top_p (float, optional): top_p used for sampling, when set to 0 top_k is used. Defaults to 0.0.
            temperature (float, optional): Softmax temperature parameter. Defaults to 1.0.
            duration (float, optional): Duration of the generated waveform. Defaults to 30.0.
            cfg_coef (float, optional): Coefficient used for classifier free guidance. Defaults to 3.0.
            two_step_cfg (bool, optional): If True, performs 2 forward for Classifier Free Guidance,
                instead of batching together the two. This has some impact on how things
                are padded but seems to have little impact in practice.
            extend_stride: when doing extended generation (i.e. more than 30 seconds), by how much
                should we extend the audio each time. Larger values will mean less context is
                preserved, and shorter value will require extra computations.
        """
        assert extend_stride < self.max_duration, "Cannot stride by more than max generation duration."
        self.extend_stride = extend_stride
        self.duration = duration
        self.generation_params = {
            'use_sampling': use_sampling,
            'temp': temperature,
            'top_k': top_k,
            'top_p': top_p,
            'cfg_coef': cfg_coef,
            'two_step_cfg': two_step_cfg,
        }

    def generate_with_chroma(self, descriptions: tp.List[str], melody_wavs: MelodyType,
                             melody_sample_rate: int, progress: bool = False,
                             return_tokens: bool = False) -> tp.Union[torch.Tensor,
                                                                      tp.Tuple[torch.Tensor, torch.Tensor]]:
        """Generate samples conditioned on text/video and melody.

        Args:
            descriptions (list of str): A list of strings used as text conditioning.
            melody_wavs: (torch.Tensor or list of Tensor): A batch of waveforms used as
                melody conditioning. Should have shape [B, C, T] with B matching the description length,
                C=1 or 2. It can be [C, T] if there is a single description. It can also be
                a list of [C, T] tensors.
            melody_sample_rate: (int): Sample rate of the melody waveforms.
            progress (bool, optional): Flag to display progress of the generation process. Defaults to False.
        """
        if isinstance(melody_wavs, torch.Tensor):
            if melody_wavs.dim() == 2:
                melody_wavs = melody_wavs[None]
            if melody_wavs.dim() != 3:
                raise ValueError("Melody wavs should have a shape [B, C, T].")
            melody_wavs = list(melody_wavs)
        else:
            for melody in melody_wavs:
                if melody is not None:
                    assert melody.dim() == 2, "One melody in the list has the wrong number of dims."

        melody_wavs = [
            convert_audio(wav, melody_sample_rate, self.sample_rate, self.audio_channels)
            if wav is not None else None
            for wav in melody_wavs]
        attributes, prompt_tokens = self._prepare_tokens_and_attributes(descriptions=descriptions, prompt=None,
                                                                        melody_wavs=melody_wavs)
        assert prompt_tokens is None
        tokens = self._generate_tokens(attributes, prompt_tokens, progress)
        if return_tokens:
            return self.generate_audio(tokens), tokens
        return self.generate_audio(tokens)

    @torch.no_grad()
    def _prepare_tokens_and_attributes(
            self,
            descriptions: tp.Sequence[tp.Optional[str]],
            prompt: tp.Optional[torch.Tensor],
            melody_wavs: tp.Optional[MelodyList] = None,
    ) -> tp.Tuple[tp.List[ConditioningAttributes], tp.Optional[torch.Tensor]]:
        """Prepare model inputs.

        Args:
            descriptions (list of str): A list of strings used as text/video conditioning.
            prompt (torch.Tensor): A batch of waveforms used for continuation.
            melody_wavs (torch.Tensor, optional): A batch of waveforms
                used as melody conditioning. Defaults to None.
        """
        if self.cond_type == 'text':
            attributes = [
                ConditioningAttributes(text={'description': description})
                for description in descriptions]
        elif self.cond_type == 'video':
            attributes = [
                ConditioningAttributes(video={'visual_content': description})
                for description in descriptions]
        else:
            raise ValueError(f"Unknown conditioning type: {type}")

        if melody_wavs is None:
            for attr in attributes:
                attr.wav['self_wav'] = WavCondition(
                    torch.zeros((1, 1, 1), device=self.device),
                    torch.tensor([0], device=self.device),
                    sample_rate=[self.sample_rate],
                    path=[None])
        else:
            if 'self_wav' not in self.lm.condition_provider.conditioners:
                raise RuntimeError("This model doesn't support melody conditioning. "
                                   "Use the `melody` model.")
            assert len(melody_wavs) == len(descriptions), \
                f"number of melody wavs must match number of descriptions! " \
                f"got melody len={len(melody_wavs)}, and descriptions len={len(descriptions)}"
            for attr, melody in zip(attributes, melody_wavs):
                if melody is None:
                    attr.wav['self_wav'] = WavCondition(
                        torch.zeros((1, 1, 1), device=self.device),
                        torch.tensor([0], device=self.device),
                        sample_rate=[self.sample_rate],
                        path=[None])
                else:
                    attr.wav['self_wav'] = WavCondition(
                        melody[None].to(device=self.device),
                        torch.tensor([melody.shape[-1]], device=self.device),
                        sample_rate=[self.sample_rate],
                        path=[None],
                    )

        if prompt is not None:
            if descriptions is not None:
                assert len(descriptions) == len(prompt), "Prompt and nb. descriptions doesn't match"
            prompt = prompt.to('cuda:1')
            prompt_tokens, scale = self.compression_model.encode(prompt)
            assert scale is None
        else:
            prompt_tokens = None
        return attributes, prompt_tokens

    def _generate_tokens(self, attributes: tp.List[ConditioningAttributes],
                         prompt_tokens: tp.Optional[torch.Tensor], progress: bool = False) -> torch.Tensor:
        """Generate discrete audio tokens given audio prompt and/or conditions.

        Args:
            attributes (list of ConditioningAttributes): Conditions used for generation (text/melody).
            prompt_tokens (torch.Tensor, optional): Audio prompt used for continuation.
            progress (bool, optional): Flag to display progress of the generation process. Defaults to False.
        Returns:
            torch.Tensor: Generated audio, of shape [B, C, T], T is defined by the generation params.
        """
        total_gen_len = int(self.duration * self.frame_rate)
        max_prompt_len = int(min(self.duration, self.max_duration) * self.frame_rate)
        current_gen_offset: int = 0

        def _progress_callback(generated_tokens: int, tokens_to_generate: int):
            generated_tokens += current_gen_offset
            if self._progress_callback is not None:
                # Note that total_gen_len might be quite wrong depending on the
                # codebook pattern used, but with delay it is almost accurate.
                self._progress_callback(generated_tokens, tokens_to_generate)
            else:
                print(f'{generated_tokens: 6d} / {tokens_to_generate: 6d}', end='\r')

        if prompt_tokens is not None:
            assert max_prompt_len >= prompt_tokens.shape[-1], \
                "Prompt is longer than audio to generate"

        callback = None
        if progress:
            callback = _progress_callback

        if self.duration <= self.max_duration:
            # generate by sampling from LM, simple case.
            with self.autocast:
                prompt_tokens = prompt_tokens.to('cuda:0')
                gen_tokens = self.lm.generate(
                    prompt_tokens, attributes,
                    callback=callback, max_gen_len=total_gen_len, **self.generation_params)

        else:
            # now this gets a bit messier, we need to handle prompts,
            # melody conditioning etc.
            ref_wavs = [attr.wav['self_wav'] for attr in attributes]
            all_tokens = []
            if prompt_tokens is None:
                prompt_length = 0
            else:
                all_tokens.append(prompt_tokens)
                prompt_length = prompt_tokens.shape[-1]

            assert self.extend_stride is not None, "Stride should be defined to generate beyond max_duration"
            assert self.extend_stride < self.max_duration, "Cannot stride by more than max generation duration."
            stride_tokens = int(self.frame_rate * self.extend_stride)

            while current_gen_offset + prompt_length < total_gen_len:
                time_offset = current_gen_offset / self.frame_rate
                chunk_duration = min(self.duration - time_offset, self.max_duration)
                max_gen_len = int(chunk_duration * self.frame_rate)
                for attr, ref_wav in zip(attributes, ref_wavs):
                    wav_length = ref_wav.length.item()
                    if wav_length == 0:
                        continue
                    # We will extend the wav periodically if it not long enough.
                    # we have to do it here rather than in conditioners.py as otherwise
                    # we wouldn't have the full wav.
                    initial_position = int(time_offset * self.sample_rate)
                    wav_target_length = int(self.max_duration * self.sample_rate)
                    positions = torch.arange(initial_position,
                                             initial_position + wav_target_length, device=self.device)
                    attr.wav['self_wav'] = WavCondition(
                        ref_wav[0][..., positions % wav_length],
                        torch.full_like(ref_wav[1], wav_target_length),
                        [self.sample_rate] * ref_wav[0].size(0),
                        [None], [0.])
                with self.autocast:
                    if isinstance(prompt_tokens, torch.Tensor):
                        prompt_tokens = prompt_tokens.to('cuda:0')
                    gen_tokens = self.lm.generate(
                        prompt_tokens, attributes,
                        callback=callback, max_gen_len=max_gen_len, **self.generation_params)
                if prompt_tokens is None:
                    all_tokens.append(gen_tokens)
                else:
                    all_tokens.append(gen_tokens[:, :, prompt_tokens.shape[-1]:])
                prompt_tokens = gen_tokens[:, :, stride_tokens:]
                prompt_length = prompt_tokens.shape[-1]
                current_gen_offset += stride_tokens

            gen_tokens = torch.cat(all_tokens, dim=-1)
        return gen_tokens
