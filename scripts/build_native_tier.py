"""Build the small Colibri ABI with installed MSVC; all output stays in project."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--colibri-source", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = Path(args.colibri_source).resolve()
    header = source / "c/tier.h"
    if not header.is_file():
        raise FileNotFoundError(header)
    vswhere = Path(r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe")
    vs = subprocess.check_output([str(vswhere), "-latest", "-products", "*",
        "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property", "installationPath"], text=True).strip()
    setup = Path(vs) / "VC/Auxiliary/Build/vcvars64.bat"
    bridge = root / "src/native/tier_bridge.c"
    output = root / "data/native"
    output.mkdir(parents=True, exist_ok=True)
    dll = output / "colibri_tier.dll"
    # Both paths originate in explicit arguments or the installed toolchain.
    types = root / "src/native/freestanding"
    command = f'call "{setup}" >nul && cl /nologo /O2 /LD /TC /GS- /Zl /I"{types}" /I"{header.parent}" "{bridge}" /Fe:"{dll}" /link /INCREMENTAL:NO /NODEFAULTLIB /NOENTRY /MACHINE:X64'
    batch = output / "build_tier.cmd"
    batch.write_text("@echo off\n" + command + "\n", encoding="utf-8")
    subprocess.run(f'cmd.exe /d /c "{batch}"', cwd=output, check=True)
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    dll.with_suffix(".json").write_text(json.dumps(dict(abi=1, upstream=str(source), revision=revision,
        tier_sha256=sha(header), bridge_sha256=sha(bridge), dll_sha256=sha(dll),
        compiler_setup=str(setup), freestanding_types_sha256=sha(types / "stdint.h"),
        capability="CPU admission policy; not native model/CUDA inference"), indent=2), encoding="utf-8")
    print(dll)


if __name__ == "__main__":
    main()
