"""
ComfyUI-FL-Qwen3TTS - Qwen3 Text-to-Speech nodes for ComfyUI

Provides high-quality TTS with voice cloning, voice design, and custom voices.
"""

import os
import sys
import logging
from pathlib import Path

# Setup logging
logger = logging.getLogger("FL_Qwen3TTS")
logger.setLevel(logging.INFO)

# Add src directory to path for bundled qwen_tts
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Register JavaScript directory for frontend extensions
try:
    import nodes
    js_dir = os.path.join(current_dir, "js")
    if os.path.exists(js_dir):
        nodes.EXTENSION_WEB_DIRS["FL_Qwen3TTS"] = js_dir
        logger.info(f"Registered JS extension directory: {js_dir}")
except Exception as e:
    logger.warning(f"Could not register JS extension: {e}")

# Import folder_paths for ComfyUI integration
try:
    import folder_paths

    # Register TTS folder for model storage
    tts_base_path = os.path.join(folder_paths.models_dir, "tts")
    qwen3tts_path = os.path.join(tts_base_path, "Qwen3TTS")

    os.makedirs(tts_base_path, exist_ok=True)
    os.makedirs(qwen3tts_path, exist_ok=True)

    # Register TTS folder if not already registered
    supported_extensions = {".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".json"}

    if "tts" not in folder_paths.folder_names_and_paths:
        folder_paths.folder_names_and_paths["tts"] = ([tts_base_path], supported_extensions)
    else:
        # Add to existing paths if not already there
        existing_paths, existing_exts = folder_paths.folder_names_and_paths["tts"]
        if tts_base_path not in existing_paths:
            existing_paths.append(tts_base_path)

    logger.info(f"Qwen3TTS models directory: {qwen3tts_path}")

except ImportError:
    logger.warning("folder_paths not available - running outside ComfyUI?")

# Populate model registries
from .modules.model_info import (
    MODEL_CONFIGS,
    TOKENIZER_CONFIGS,
    AVAILABLE_QWEN3TTS_MODELS,
    AVAILABLE_TOKENIZERS,
)

# Register official models
for model_name, config in MODEL_CONFIGS.items():
    AVAILABLE_QWEN3TTS_MODELS[model_name] = {
        "type": "official",
        **config,
    }

# Register official tokenizers
for tokenizer_name, config in TOKENIZER_CONFIGS.items():
    AVAILABLE_TOKENIZERS[tokenizer_name] = {
        "type": "official",
        **config,
    }

# Scan for local models
try:
    qwen3tts_models_dir = Path(qwen3tts_path)
    if qwen3tts_models_dir.exists():
        for item in qwen3tts_models_dir.iterdir():
            if item.is_dir():
                config_file = item / "config.json"
                if config_file.exists():
                    # Check if it's a model or tokenizer
                    if item.name not in AVAILABLE_QWEN3TTS_MODELS and item.name not in AVAILABLE_TOKENIZERS:
                        # Try to determine type from config
                        import json
                        with open(config_file, "r") as f:
                            cfg = json.load(f)

                        if "tokenizer" in item.name.lower() or cfg.get("model_type") == "tokenizer":
                            AVAILABLE_TOKENIZERS[item.name] = {
                                "type": "local",
                                "path": str(item),
                            }
                            logger.info(f"Found local tokenizer: {item.name}")
                        else:
                            AVAILABLE_QWEN3TTS_MODELS[item.name] = {
                                "type": "local",
                                "path": str(item),
                            }
                            logger.info(f"Found local model: {item.name}")
except Exception as e:
    logger.warning(f"Error scanning for local models: {e}")

logger.info(f"Available Qwen3TTS models: {list(AVAILABLE_QWEN3TTS_MODELS.keys())}")
logger.info(f"Available tokenizers: {list(AVAILABLE_TOKENIZERS.keys())}")

# Import nodes
try:
    from .nodes import (
        FL_Qwen3TTS_ModelLoader,
        FL_Qwen3TTS_TokenizerLoader,
        FL_Qwen3TTS_CustomVoice,
        FL_Qwen3TTS_VoiceDesign,
        FL_Qwen3TTS_VoiceClone,
        FL_Qwen3TTS_VoiceClonePrompt,
        FL_Qwen3TTS_AudioEncode,
        FL_Qwen3TTS_AudioDecode,
        FL_Qwen3TTS_Transcribe,
        FL_Qwen3TTS_TrainingUI,
    )

    NODE_CLASS_MAPPINGS = {
        "FL_Qwen3TTS_ModelLoader": FL_Qwen3TTS_ModelLoader,
        "FL_Qwen3TTS_TokenizerLoader": FL_Qwen3TTS_TokenizerLoader,
        "FL_Qwen3TTS_CustomVoice": FL_Qwen3TTS_CustomVoice,
        "FL_Qwen3TTS_VoiceDesign": FL_Qwen3TTS_VoiceDesign,
        "FL_Qwen3TTS_VoiceClone": FL_Qwen3TTS_VoiceClone,
        "FL_Qwen3TTS_VoiceClonePrompt": FL_Qwen3TTS_VoiceClonePrompt,
        "FL_Qwen3TTS_AudioEncode": FL_Qwen3TTS_AudioEncode,
        "FL_Qwen3TTS_AudioDecode": FL_Qwen3TTS_AudioDecode,
        "FL_Qwen3TTS_Transcribe": FL_Qwen3TTS_Transcribe,
        "FL_Qwen3TTS_TrainingUI": FL_Qwen3TTS_TrainingUI,
    }

    NODE_DISPLAY_NAME_MAPPINGS = {
        "FL_Qwen3TTS_ModelLoader": "FL Qwen3 TTS Model Loader",
        "FL_Qwen3TTS_TokenizerLoader": "FL Qwen3 TTS Tokenizer Loader",
        "FL_Qwen3TTS_CustomVoice": "FL Qwen3 TTS Custom Voice",
        "FL_Qwen3TTS_VoiceDesign": "FL Qwen3 TTS Voice Design",
        "FL_Qwen3TTS_VoiceClone": "FL Qwen3 TTS Voice Clone",
        "FL_Qwen3TTS_VoiceClonePrompt": "FL Qwen3 TTS Voice Clone Prompt",
        "FL_Qwen3TTS_AudioEncode": "FL Qwen3 TTS Audio Encode",
        "FL_Qwen3TTS_AudioDecode": "FL Qwen3 TTS Audio Decode",
        "FL_Qwen3TTS_Transcribe": "FL Qwen3 TTS Transcribe",
        "FL_Qwen3TTS_TrainingUI": "FL Qwen3 TTS Training UI",
    }

    logger.info("FL Qwen3TTS nodes loaded successfully!")

except Exception as e:
    logger.error(f"Failed to load FL Qwen3TTS nodes: {e}")
    import traceback
    traceback.print_exc()

    # Provide empty mappings on failure
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

# Print banner (separate from node registration so encoding errors don't break loading)
if NODE_CLASS_MAPPINGS:
    try:
        ascii_art = """
 ⣏⡉ ⡇    ⡎⢱ ⡇⢸ ⣏⡉ ⡷⣸   ⢹⠁ ⢹⠁ ⢎⡑
 ⠇  ⠧⠤   ⠣⠪ ⠟⠻ ⠧⠤ ⠇⠹   ⠸  ⠸  ⠢⠜
"""
        print(f"\033[35m{ascii_art}\033[0m")
    except (UnicodeEncodeError, OSError):
        pass
    print("FL Qwen3 TTS Custom Nodes Loaded - Version 1.0.5")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
