"""Extract only official mtp.* byte ranges, without downloading a full checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import urllib.request


def get_json(url):
    with urllib.request.urlopen(url, timeout=40) as response:
        return json.load(response)


def read_range(url, start, size):
    request = urllib.request.Request(url + f"?download=true&offset={start}",
                                    headers={"Range": f"bytes={start}-{start+size-1}"})
    with urllib.request.urlopen(request, timeout=60) as response:
        if response.status != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {start}-{start+size-1}/"):
            raise RuntimeError("Server did not honor the bounded byte range")
        result = response.read(size + 1)
    if len(result) != size:
        raise RuntimeError("Incomplete tensor byte range")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-mib", type=int, default=128)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    info = get_json(f"https://huggingface.co/api/models/{args.repo}")
    revision = info["sha"]
    base = f"https://huggingface.co/{args.repo}/resolve/{revision}/"
    index = get_json(base + "model.safetensors.index.json")["weight_map"]
    selected = {k: v for k, v in index.items() if k.startswith("mtp.")}
    if not selected:
        raise RuntimeError("Official checkpoint contains no mtp.* tensors")
    headers = {}
    for shard in sorted(set(selected.values())):
        length = struct.unpack("<Q", read_range(base + shard, 0, 8))[0]
        if length > 16*2**20:
            raise RuntimeError("Unexpectedly large safetensors header")
        headers[shard] = (8+length, json.loads(read_range(base + shard, 8, length)))
    output_header, offset = {}, 0
    for key, shard in selected.items():
        entry = headers[shard][1][key]
        size = entry["data_offsets"][1] - entry["data_offsets"][0]
        output_header[key] = {**entry, "data_offsets": [offset, offset+size]}
        offset += size
    if offset > args.max_mib*2**20:
        raise RuntimeError(f"MTP head exceeds the {args.max_mib} MiB extraction budget")
    output_header["__metadata__"] = {"format": "pt", "repo": args.repo, "revision": revision}
    raw_header = json.dumps(output_header, separators=(",", ":")).encode()
    raw_header += b" " * (-len(raw_header) % 8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_suffix(args.output.suffix + ".partial")
    receipt = {"repo": args.repo, "revision": revision, "tensor_bytes": offset, "tensors": {}}
    print(f"Extracting {len(selected)} official tensors ({offset/2**20:.2f} MiB)", flush=True)
    with partial.open("xb") as stream:
        stream.write(struct.pack("<Q", len(raw_header)))
        stream.write(raw_header)
        for key, shard in selected.items():
            data_start, header = headers[shard]
            lo, hi = header[key]["data_offsets"]
            digest = hashlib.sha256()
            for start in range(lo, hi, 4*2**20):
                chunk = read_range(base + shard, data_start+start, min(hi-start, 4*2**20))
                digest.update(chunk)
                stream.write(chunk)
            receipt["tensors"][key] = {"sha256": digest.hexdigest(), "source_shard": shard,
                                       "source_data_offsets": [lo, hi], "shape": header[key]["shape"]}
            print(key, flush=True)
    partial.replace(args.output)
    receipt["output_sha256"] = hashlib.sha256(args.output.read_bytes()).hexdigest()
    args.output.with_suffix(".provenance.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(f"Saved {args.output.name}", flush=True)


if __name__ == "__main__":
    main()
