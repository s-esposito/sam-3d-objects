# Copyright (c) Meta Platforms, Inc. and affiliates.
from typing import *
from loguru import logger

BACKEND = "sdpa"
DEBUG = False


def __from_env():
    import os

    global BACKEND
    global DEBUG

    # env_attn_backend = "sdpa" # os.environ.get("ATTN_BACKEND")
    env_attn_backend = "xformers"
    env_sttn_debug = os.environ.get("ATTN_DEBUG")

    if env_attn_backend is not None and env_attn_backend in [
        "xformers",
        "flash_attn",
        "torch_flash_attn",
        "sdpa",
        "naive",
    ]:
        BACKEND = env_attn_backend
    # BACKEND = "sdpa"
    if env_sttn_debug is not None:
        DEBUG = env_sttn_debug == "1"

    logger.info(f"[ATTENTION] Using backend: {BACKEND}")


__from_env()


def set_backend(backend: Literal["xformers", "flash_attn", "torch_flash_attn"]):
    global BACKEND
    BACKEND = backend
    print(f"[ATTENTION] Using backend: {BACKEND}")


def set_debug(debug: bool):
    global DEBUG
    DEBUG = debug
    print(f"[ATTENTION] Debug mode: {DEBUG}")


from .full_attn import *
from .modules import *
