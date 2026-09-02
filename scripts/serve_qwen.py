#!/usr/bin/env python3
"""Start the local Qwen vLLM OpenAI-compatible server."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from typing import List, Optional


VLLM_COMPAT_ENV = {
    "VLLM_USE_FLASHINFER_SAMPLER": "0",
    "VLLM_USE_V1": "0",
    "VLLM_ATTENTION_BACKEND": "XFORMERS",
}


def build_command(
    *,
    executable: str,
    model: str,
    served_model_name: str,
    host: str,
    port: int,
    api_key: str,
    max_model_len: int,
    max_num_seqs: int,
    gpu_memory_utilization: float,
    enable_auto_tool_choice: bool = True,
    tool_call_parser: str = "hermes",
    enable_lora: bool = False,
    lora_modules: Optional[List[str]] = None,
) -> List[str]:
    """Build the vLLM CLI command without starting a subprocess."""
    command = [
        executable,
        "serve",
        model,
        "--host",
        host,
        "--port",
        str(port),
        "--served-model-name",
        served_model_name,
        "--api-key",
        api_key,
        "--max-model-len",
        str(max_model_len),
        "--max-num-seqs",
        str(max_num_seqs),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
    ]
    if enable_auto_tool_choice:
        command.append("--enable-auto-tool-choice")
    if tool_call_parser:
        command.extend(["--tool-call-parser", tool_call_parser])
    if enable_lora:
        command.append("--enable-lora")
        if lora_modules:
            command.extend(["--lora-modules", *lora_modules])
    return command


def build_module_command(
    *,
    model: str,
    served_model_name: str,
    host: str,
    port: int,
    api_key: str,
    max_model_len: int,
    max_num_seqs: int,
    gpu_memory_utilization: float,
    enable_auto_tool_choice: bool = True,
    tool_call_parser: str = "hermes",
    enable_lora: bool = False,
    lora_modules: Optional[List[str]] = None,
) -> List[str]:
    """Build the Python-module fallback for environments without a vllm shim."""
    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model,
        "--host",
        host,
        "--port",
        str(port),
        "--served-model-name",
        served_model_name,
        "--api-key",
        api_key,
        "--max-model-len",
        str(max_model_len),
        "--max-num-seqs",
        str(max_num_seqs),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
    ]
    if enable_auto_tool_choice:
        command.append("--enable-auto-tool-choice")
    if tool_call_parser:
        command.extend(["--tool-call-parser", tool_call_parser])
    if enable_lora:
        command.append("--enable-lora")
        if lora_modules:
            command.extend(["--lora-modules", *lora_modules])
    return command


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.environ.get("QWEN_MODEL", "Qwen/Qwen3.5-2B"))
    parser.add_argument("--served-model-name", default="qwen-base")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY", "EMPTY"))
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--enable-auto-tool-choice", action="store_true", default=True)
    parser.add_argument("--tool-call-parser", default="hermes")
    parser.add_argument("--enable-lora", action="store_true")
    parser.add_argument("--lora-modules", nargs="+", default=None)
    args = parser.parse_args()

    os.environ.update(VLLM_COMPAT_ENV)
    executable = shutil.which("vllm")
    if executable:
        command = build_command(
            executable=executable,
            model=args.model,
            served_model_name=args.served_model_name,
            host=args.host,
            port=args.port,
            api_key=args.api_key,
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enable_auto_tool_choice=args.enable_auto_tool_choice,
            tool_call_parser=args.tool_call_parser,
            enable_lora=args.enable_lora,
            lora_modules=args.lora_modules,
        )
    else:
        command = build_module_command(
            model=args.model,
            served_model_name=args.served_model_name,
            host=args.host,
            port=args.port,
            api_key=args.api_key,
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enable_auto_tool_choice=args.enable_auto_tool_choice,
            tool_call_parser=args.tool_call_parser,
            enable_lora=args.enable_lora,
            lora_modules=args.lora_modules,
        )

    print("Starting vLLM with:")
    print("  model:", args.model)
    print("  endpoint:", f"http://{args.host}:{args.port}/v1")
    print("  VLLM_USE_FLASHINFER_SAMPLER=0")
    print("  VLLM_USE_V1=0")
    print("  VLLM_ATTENTION_BACKEND=XFORMERS")
    return subprocess.run(command, env=os.environ.copy(), check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
