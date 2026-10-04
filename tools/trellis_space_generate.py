#!/usr/bin/env python3
"""Generate a GLB through the public Microsoft TRELLIS.2 Hugging Face Space.

The GitHub runner only acts as a lightweight client. Heavy GPU inference happens
inside the remote Space.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

from gradio_client import Client, handle_file


def endpoint_name(api_info: dict[str, Any], needle: str, required: bool = True) -> str | None:
    named = api_info.get("named_endpoints", {})
    exact = f"/{needle}"
    if exact in named:
        return exact
    for name in named:
        if needle.lower() in name.lower():
            return name
    if required:
        raise RuntimeError(
            f"Endpoint {needle!r} not found. Available named endpoints: {list(named)}"
        )
    return None


def find_downloaded_glb(value: Any) -> Path | None:
    if isinstance(value, (str, os.PathLike)):
        path = Path(value)
        if path.suffix.lower() == ".glb" and path.exists():
            return path
        return None
    if isinstance(value, dict):
        for key in ("path", "name", "file", "value"):
            if key in value:
                found = find_downloaded_glb(value[key])
                if found:
                    return found
        for item in value.values():
            found = find_downloaded_glb(item)
            if found:
                return found
        return None
    if isinstance(value, (list, tuple)):
        for item in value:
            found = find_downloaded_glb(item)
            if found:
                return found
    return None


def as_file_input(value: Any) -> Any:
    if isinstance(value, (str, os.PathLike)) and Path(value).exists():
        return handle_file(str(value))
    if isinstance(value, dict):
        for key in ("path", "name"):
            candidate = value.get(key)
            if candidate and Path(str(candidate)).exists():
                return handle_file(str(candidate))
    if isinstance(value, (list, tuple)) and value:
        return as_file_input(value[0])
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--space", default="microsoft/TRELLIS.2")
    parser.add_argument("--resolution", choices=["512", "1024", "1536"], default="512")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--decimation-target", type=int, default=250000)
    parser.add_argument("--texture-size", type=int, choices=[1024, 2048, 4096], default=2048)
    args = parser.parse_args()

    image_path = Path(args.image)
    output_path = Path(args.output)
    if not image_path.is_file():
        raise FileNotFoundError(image_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    downloads = output_path.parent / "_gradio_downloads"
    downloads.mkdir(parents=True, exist_ok=True)

    token = os.environ.get("HF_TOKEN") or None
    print(f"Connecting to {args.space} ...", flush=True)
    client = Client(
        args.space,
        token=token,
        verbose=True,
        download_files=str(downloads),
    )

    api = client.view_api(print_info=False, return_format="dict")
    (output_path.parent / "space_api.json").write_text(
        json.dumps(api, indent=2, default=str),
        encoding="utf-8",
    )

    start_ep = endpoint_name(api, "start_session", required=False)
    preprocess_ep = endpoint_name(api, "preprocess_image")
    generate_ep = endpoint_name(api, "image_to_3d")
    extract_ep = endpoint_name(api, "extract_glb")
    end_ep = endpoint_name(api, "end_session", required=False)

    if start_ep:
        try:
            client.predict(api_name=start_ep)
        except Exception as exc:
            print(f"start_session skipped: {exc}", flush=True)

    print(f"Preprocessing {image_path} ...", flush=True)
    processed = client.predict(
        handle_file(str(image_path)),
        api_name=preprocess_ep,
    )
    processed_input = as_file_input(processed)

    print(
        f"Generating 3D at {args.resolution}, seed={args.seed}. "
        "ZeroGPU can keep this request queued for a while.",
        flush=True,
    )
    client.predict(
        processed_input,
        args.seed,
        args.resolution,
        7.5, 0.7, 12, 5.0,
        7.5, 0.5, 12, 3.0,
        1.0, 0.0, 12, 3.0,
        api_name=generate_ep,
    )

    print("Extracting GLB ...", flush=True)
    extracted = client.predict(
        args.decimation_target,
        args.texture_size,
        api_name=extract_ep,
    )
    glb = find_downloaded_glb(extracted)
    if glb is None:
        raise RuntimeError(f"Space returned no downloaded GLB path: {extracted!r}")

    shutil.copy2(glb, output_path)
    print(f"GLB ready: {output_path} ({output_path.stat().st_size} bytes)", flush=True)

    if end_ep:
        try:
            client.predict(api_name=end_ep)
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"TRELLIS workflow failed: {exc}", file=sys.stderr, flush=True)
        raise
