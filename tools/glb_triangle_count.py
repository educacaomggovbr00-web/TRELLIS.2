#!/usr/bin/env python3
"""Print the triangle count stored in a binary glTF (.glb)."""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path


def read_glb_json(path: Path) -> dict:
    data = path.read_bytes()
    if len(data) < 20 or data[:4] != b"glTF":
        raise ValueError(f"{path} is not a valid GLB")
    version, total_length = struct.unpack_from("<II", data, 4)
    if version != 2:
        raise ValueError(f"Unsupported GLB version: {version}")
    if total_length > len(data):
        raise ValueError("Truncated GLB")
    chunk_length, chunk_type = struct.unpack_from("<II", data, 12)
    if chunk_type != 0x4E4F534A:
        raise ValueError("First GLB chunk is not JSON")
    payload = data[20:20 + chunk_length].rstrip(b" \t\r\n\0")
    return json.loads(payload.decode("utf-8"))


def triangle_count(doc: dict) -> int:
    accessors = doc.get("accessors", [])
    total = 0
    for mesh in doc.get("meshes", []):
        for primitive in mesh.get("primitives", []):
            mode = primitive.get("mode", 4)
            if mode != 4:
                continue
            if "indices" in primitive:
                count = int(accessors[primitive["indices"]].get("count", 0))
            else:
                pos = primitive.get("attributes", {}).get("POSITION")
                count = int(accessors[pos].get("count", 0)) if pos is not None else 0
            total += count // 3
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("glb")
    args = parser.parse_args()
    print(triangle_count(read_glb_json(Path(args.glb))))


if __name__ == "__main__":
    main()
