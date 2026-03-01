"""
Model configuration and registry for Qwen3-TTS models.
"""

# Model configurations with HuggingFace repo IDs
MODEL_CONFIGS = {
    "Qwen3-TTS-12Hz-1.7B-Base": {
        "repo_id": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        "model_type": "base",
        "tokenizer_version": "12hz",
        "description": "Base model for voice cloning and fine-tuning",
    },
    "Qwen3-TTS-12Hz-1.7B-CustomVoice": {
        "repo_id": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "model_type": "custom_voice",
        "tokenizer_version": "12hz",
        "description": "9 predefined speakers with optional style instructions",
    },
    "Qwen3-TTS-12Hz-1.7B-VoiceDesign": {
        "repo_id": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        "model_type": "voice_design",
        "tokenizer_version": "12hz",
        "description": "Create voices from natural language descriptions",
    },
    "Qwen3-TTS-12Hz-0.6B-Base": {
        "repo_id": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        "model_type": "base",
        "tokenizer_version": "12hz",
        "description": "Lightweight base model (0.6B) for voice cloning — lower VRAM",
    },
    "Qwen3-TTS-12Hz-0.6B-CustomVoice": {
        "repo_id": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        "model_type": "custom_voice",
        "tokenizer_version": "12hz",
        "description": "Lightweight custom voice model (0.6B) with predefined speakers — lower VRAM",
    },
}

# Tokenizer configurations
TOKENIZER_CONFIGS = {
    "Qwen3-TTS-Tokenizer-12Hz": {
        "repo_id": "Qwen/Qwen3-TTS-Tokenizer-12Hz",
        "version": "12hz",
        "sample_rate": 24000,
        "description": "12Hz tokenizer for all Qwen3-TTS models",
    },
}

# Predefined speakers for CustomVoice model
SPEAKERS = [
    "Vivian",      # Bright, edgy female (Chinese)
    "Serena",      # Warm, gentle female (Chinese)
    "Uncle_Fu",    # Seasoned male (Chinese)
    "Dylan",       # Beijing male (Beijing dialect)
    "Eric",        # Sichuan male (Sichuan dialect)
    "Ryan",        # Dynamic male (English)
    "Aiden",       # American male (English)
    "Ono_Anna",    # Japanese female
    "Sohee",       # Korean female
]

# Supported languages
LANGUAGES = [
    "Chinese",
    "English",
    "Japanese",
    "Korean",
    "German",
    "French",
    "Russian",
    "Portuguese",
    "Spanish",
    "Italian",
]

# Runtime registry - will be populated at startup
AVAILABLE_QWEN3TTS_MODELS = {}
AVAILABLE_TOKENIZERS = {}


def get_available_devices():
    """Get list of available compute devices."""
    import torch

    devices = []
    if torch.cuda.is_available():
        devices.append("cuda")
        # Add individual CUDA devices
        for i in range(torch.cuda.device_count()):
            devices.append(f"cuda:{i}")

    # Check for other backends
    try:
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            devices.append("mps")
    except:
        pass

    devices.append("cpu")
    return devices


def get_dtype_options():
    """Get available dtype options."""
    return ["bfloat16", "float16", "float32"]


def get_attention_options():
    """Get available attention implementations."""
    return ["sdpa", "flash_attention_2", "eager"]
