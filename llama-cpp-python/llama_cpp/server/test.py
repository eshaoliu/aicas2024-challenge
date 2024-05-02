from __future__ import annotations

import os
import sys
import argparse
import llama_cpp
import uvicorn

from typing import Optional, List, Literal, Union
from pydantic import Field

from llama_cpp.server.app import create_app
from llama_cpp.server.settings import (
    Settings,
    BaseSettings,
    #ModelSettings,
    ConfigFileSettings,
)

class ModelSettings(BaseSettings):
    """Model settings used to load a Llama model."""

    model: str = Field(
        default='/root/data/Qwen-1_8B/ggml-model-q5_k_m.gguf',
        description="The path to the model to use for generating completions."
    )
    model_alias: Optional[str] = Field(
        default=None,
        description="The alias of the model to use for generating completions.",
    )
    # Model Params
    n_gpu_layers: int = Field(
        default=0,
        ge=-1,
        description="The number of layers to put on the GPU. The rest will be on the CPU. Set -1 to move all to GPU.",
    )
    split_mode: int = Field(
        default=llama_cpp.LLAMA_SPLIT_MODE_LAYER,
        description="The split mode to use.",
    )
    main_gpu: int = Field(
        default=0,
        ge=0,
        description="Main GPU to use.",
    )
    tensor_split: Optional[List[float]] = Field(
        default=None,
        description="Split layers across multiple GPUs in proportion.",
    )
    vocab_only: bool = Field(
        default=False, description="Whether to only return the vocabulary."
    )
    use_mmap: bool = Field(
        default=llama_cpp.llama_supports_mmap(),
        description="Use mmap.",
    )
    use_mlock: bool = Field(
        default=llama_cpp.llama_supports_mlock(),
        description="Use mlock.",
    )
    kv_overrides: Optional[List[str]] = Field(
        default=None,
        description="List of model kv overrides in the format key=type:value where type is one of (bool, int, float). Valid true values are (true, TRUE, 1), otherwise false.",
    )
    # Context Params
    seed: int = Field(
        default=llama_cpp.LLAMA_DEFAULT_SEED, description="Random seed. -1 for random."
    )
    n_ctx: int = Field(default=1024, ge=0, description="The context size.")
    n_batch: int = Field(
        default=512, ge=1, description="The batch size to use per eval."
    )
    n_threads: int = Field(
        default=8,
        ge=1,
        description="The number of threads to use.",
    )
    n_threads_batch: int = Field(
        default=8,
        ge=0,
        description="The number of threads to use when batch processing.",
    )
    rope_scaling_type: int = Field(
        default=llama_cpp.LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED
    )
    rope_freq_base: float = Field(default=0.0, description="RoPE base frequency")
    rope_freq_scale: float = Field(
        default=0.0, description="RoPE frequency scaling factor"
    )
    yarn_ext_factor: float = Field(default=-1.0)
    yarn_attn_factor: float = Field(default=1.0)
    yarn_beta_fast: float = Field(default=32.0)
    yarn_beta_slow: float = Field(default=1.0)
    yarn_orig_ctx: int = Field(default=0)
    mul_mat_q: bool = Field(
        default=True, description="if true, use experimental mul_mat_q kernels"
    )
    logits_all: bool = Field(default=True, description="Whether to return logits.")
    embedding: bool = Field(default=True, description="Whether to use embeddings.")
    offload_kqv: bool = Field(
        default=True, description="Whether to offload kqv to the GPU."
    )
    # Sampling Params
    last_n_tokens_size: int = Field(
        default=64,
        ge=0,
        description="Last n tokens to keep for repeat penalty calculation.",
    )
    # LoRA Params
    lora_base: Optional[str] = Field(
        default=None,
        description="Optional path to base model, useful if using a quantized base model and you want to apply LoRA to an f16 model.",
    )
    lora_path: Optional[str] = Field(
        default=None,
        description="Path to a LoRA file to apply to the model.",
    )
    # Backend Params
    numa: Union[bool, int] = Field(
        default=False,
        description="Enable NUMA support.",
    )
    # Chat Format Params
    chat_format: Optional[str] = Field(
        default=None,
        description="Chat format to use.",
    )
    clip_model_path: Optional[str] = Field(
        default=None,
        description="Path to a CLIP model to use for multi-modal chat completion.",
    )
    # Cache Params
    cache: bool = Field(
        default=False,
        description="Use a cache to reduce processing times for evaluated prompts.",
    )
    cache_type: Literal["ram", "disk"] = Field(
        default="ram",
        description="The type of cache to use. Only used if cache is True.",
    )
    cache_size: int = Field(
        default=2 << 30,
        description="The size of the cache in bytes. Only used if cache is True.",
    )
    # Tokenizer Options
    hf_tokenizer_config_path: Optional[str] = Field(
        default=None,
        description="The path to a HuggingFace tokenizer_config.json file.",
    )
    hf_pretrained_model_name_or_path: Optional[str] = Field(
        default=None,
        description="The model name or path to a pretrained HuggingFace tokenizer model. Same as you would pass to AutoTokenizer.from_pretrained().",
    )
    # Loading from HuggingFace Model Hub
    hf_model_repo_id: Optional[str] = Field(
        default=None,
        description="The model repo id to use for the HuggingFace tokenizer model.",
    )
    # Speculative Decoding
    draft_model: Optional[str] = Field(
        default=None,
        description="Method to use for speculative decoding. One of (prompt-lookup-decoding).",
    )
    draft_model_num_pred_tokens: int = Field(
        default=10,
        description="Number of tokens to predict using the draft model.",
    )
    # KV Cache Quantization
    type_k: Optional[int] = Field(
        default=None,
        description="Type of the key cache quantization.",
    )
    type_v: Optional[int] = Field(
        default=None,
        description="Type of the value cache quantization.",
    )
    # Misc
    verbose: bool = Field(
        default=True, description="Whether to print debug information."
    )


class ServerSettings(BaseSettings):
    """Server settings used to configure the FastAPI and Uvicorn server."""

    # Uvicorn Settings
    host: str = Field(default="localhost", description="Listen address")
    port: int = Field(default=8000, description="Listen port")
    ssl_keyfile: Optional[str] = Field(
        default=None, description="SSL key file for HTTPS"
    )
    ssl_certfile: Optional[str] = Field(
        default=None, description="SSL certificate file for HTTPS"
    )
    # FastAPI Settings
    api_key: Optional[str] = Field(
        default=None,
        description="API key for authentication. If set all requests need to be authenticated.",
    )
    interrupt_requests: bool = Field(
        default=True,
        description="Whether to interrupt requests when a new request is received.",
    )

def main():
    description = "🦙 Llama.cpp python server. Host your own LLMs!🚀"
    parser = argparse.ArgumentParser(description=description)

    # add_args_from_model(parser, Settings)
    # parser.add_argument(
    #     "--config_file",
    #     type=str,
    #     help="Path to a config file to load.",
    # )
    server_settings: ServerSettings | None = None
    model_settings: list[ModelSettings] = []
    args = parser.parse_args()
    try:
        # Load server settings from config_file if provided
        server_settings = ServerSettings
        model_settings = [ModelSettings]
    except Exception as e:
        print(e, file=sys.stderr)
        parser.print_help()
        sys.exit(1)
    assert server_settings is not None
    assert model_settings is not None
    app = create_app(
        server_settings=server_settings,
        model_settings=model_settings,
    )
    uvicorn.run(
        app,
        host=os.getenv("HOST", server_settings.host),
        port=int(os.getenv("PORT", server_settings.port)),
        ssl_keyfile=server_settings.ssl_keyfile,
        ssl_certfile=server_settings.ssl_certfile,
    )


if __name__ == "__main__":
    main()