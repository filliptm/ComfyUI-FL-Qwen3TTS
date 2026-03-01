"""
Custom voice generation node for Qwen3-TTS.
"""

import torch
import logging

from comfy.utils import ProgressBar

from ..modules.model_info import LANGUAGES
from ..modules.audio_utils import numpy_to_comfyui_audio, empty_audio

logger = logging.getLogger("FL_Qwen3TTS")


def set_seed(seed):
    """Set random seed for reproducibility."""
    if seed < 0:
        seed = torch.randint(0, 999999999, (1,)).item()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


class FL_Qwen3TTS_CustomVoice:
    """Generate speech using predefined speakers with optional style instructions."""

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "FL/Qwen3TTS"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("QWEN3TTS_MODEL",),
                "text": ("STRING", {"multiline": True, "default": "Hello, this is a test of the Qwen3 text to speech system."}),
                "speaker": ("STRING", {"default": "Ryan"}),
                "language": (LANGUAGES, {"default": "English"}),
                "top_k": ("INT", {"default": 50, "min": 1, "max": 200}),
                "top_p": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 1.0, "step": 0.05}),
                "temperature": ("FLOAT", {"default": 0.9, "min": 0.1, "max": 2.0, "step": 0.05}),
                "repetition_penalty": ("FLOAT", {"default": 1.05, "min": 1.0, "max": 2.0, "step": 0.01}),
                "max_new_tokens": ("INT", {"default": 2048, "min": 128, "max": 8192}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xFFFFFFFFFFFFFFFF}),
            },
            "optional": {
                "instruct": ("STRING", {"multiline": False, "default": ""}),
            }
        }

    def generate(self, model, text, speaker, language, top_k, top_p, temperature,
                 repetition_penalty, max_new_tokens, seed, instruct=None):
        if not model or "model" not in model:
            raise ValueError("No model provided. Please connect a Model Loader node.")

        tts_model = model["model"]

        # Check model type compatibility
        model_type = getattr(tts_model.model, 'tts_model_type', None)
        if model_type != "custom_voice":
            raise ValueError(
                f"Wrong model type for Custom Voice node!\n\n"
                f"You are using: {model.get('model_name', 'Unknown')} (type: {model_type})\n\n"
                f"This node requires: Qwen3-TTS-12Hz-1.7B-CustomVoice\n\n"
                f"Please change your Model Loader to use 'Qwen3-TTS-12Hz-1.7B-CustomVoice' "
                f"which has 9 predefined speakers (Ryan, Vivian, Serena, etc.).\n\n"
                f"Alternatively, use the Voice Clone node with your current Base model "
                f"to clone a voice from reference audio."
            )

        actual_seed = set_seed(seed)
        logger.info(f"Generating speech with speaker '{speaker}', seed: {actual_seed}")

        generate_config = {
            "do_sample": True,
            "top_k": top_k,
            "top_p": top_p,
            "temperature": temperature,
            "repetition_penalty": repetition_penalty,
            "max_new_tokens": max_new_tokens,
        }

        # Setup progress bar with granular generation progress
        # Total: 1 (tokenize) + max_new_tokens (generate) + 1 (decode)
        total_steps = 2 + max_new_tokens
        pbar = ProgressBar(total_steps)

        def progress_callback(stage, current, total):
            if stage == "tokenizing" and current == total:
                pbar.update_absolute(1, total_steps)
            elif stage == "generating":
                # Show per-token progress during generation
                pbar.update_absolute(1 + current, total_steps)
            elif stage == "decoding" and current == total:
                pbar.update_absolute(total_steps, total_steps)

        try:
            wavs, sr = tts_model.generate_custom_voice(
                text=text,
                speaker=speaker,
                language=language,
                instruct=instruct if instruct and instruct.strip() else None,
                progress_callback=progress_callback,
                **generate_config,
            )

            output_audio = numpy_to_comfyui_audio(wavs[0], sr)
            logger.info(f"Generated audio: {wavs[0].shape} samples at {sr}Hz")

            return (output_audio,)

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            import traceback
            traceback.print_exc()
            return (empty_audio(24000),)
