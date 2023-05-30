"""_summary_

This module contains the main FastAPI application.
"""
import logging
import os
from types import FunctionType
from typing import (
    Any,
    Annotated,
)
from fastapi import FastAPI, Depends
from pydantic import BaseModel
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedTokenizer,
    PreTrainedModel
)
import torch

CHECK_POINT = os.environ.get("CHECK_POINT", "replit/replit-code-v1-3b")
DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

LOGGING_LEVEL = os.environ.get("LOGGING_LEVEL", "INFO")
MODELS_FOLDER = os.environ.get("MODELS_FOLDER", "models")
CACHE_FOLDER = os.environ.get("MODELS_FOLDER", "cache")

log = logging.getLogger("uvicorn")


class CodeCompletionRequestBody(BaseModel):
    """Request body for /chat/completions."""

    inputs: str
    parameters: dict[str, Any]


async def get_generatel():
    """_summary_

    Args:
        body (CodeompletionRequestBody): _description_

    Returns:
        _type_: _description_
    """
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        CHECK_POINT,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        cache_dir=CACHE_FOLDER,
        local_dir=MODELS_FOLDER,
    ).to(DEVICE)
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
        CHECK_POINT,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        cache_dir=CACHE_FOLDER,
        local_dir=MODELS_FOLDER,
    )
    default_parameters = dict(
        do_sample=True,
        top_p=0.95,
        top_k=4,
        pad_token_id=tokenizer.eos_token_id,
        temperature=0.2,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
    )
    default_parameters.pop("stop")

    def generate(inputs: str):
        input_ids = tokenizer.encode(inputs, return_tensors="pt")
        output_ids: torch.Tensor = model.generate(input_ids, default_parameters)
        output_text: str = tokenizer.decode(
            output_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text

    return generate


app = FastAPI()


@app.on_event("startup")
async def startup_event():
    """_summary_
    Starts up the server, setting log level, downloading the default model if necessary.
    """
    log.info("Starting up...")
    log.setLevel(LOGGING_LEVEL)
    log.info("Log level set to %s", LOGGING_LEVEL)


@app.get("/ping")
async def ping():
    """_summary_

    Returns:
        _type_: pong!
    """
    return {"ialacol-code": "pong"}


@app.post("/v1/code/completions", response_model=Any)
async def chat_completions(
    body: CodeCompletionRequestBody,
    generate: Annotated[FunctionType, Depends(get_generatel)],
):
    """_summary_
        Compatible with https://platform.openai.com/docs/api-reference/chat
    Args:
        body (ChatCompletionRequestBody): parsed request body

    Returns:
        StreamingResponse: response
    """
    log.debug("Body:%s", str(body))
    generation_result = generate(body.inputs)
    log.debug("http_response:%s ", generation_result)
    return generation_result
