# initial weight export script
import argparse
import struct
import numpy as np
import safetensors.numpy
from pathlib import Path


def sanitize_name(name: str) -> str:
    return name.replace("/", ".").replace(":", "_")


def extract_ternary_g128(arr: np.ndarray) -> tuple:
    """Extract ternary values and scales from G128 quantized weight.

    Returns:
        magnitude: packed as uint64_t[2] per block (128 bits, little-endian)
        sign: packed as uint64_t[2] per block (128 bits, little-endian)
        scales: FP16 per block
    """
    total_elements = arr.size
    num_blocks = total_elements // 128

    # Reshape into blocks of 128
    arr_flat = arr.flat[:num_blocks * 128].reshape(num_blocks, 128)

    # Extract magnitude: 1 if non-zero, 0 otherwise
    magnitude = (np.abs(arr_flat) > 0).astype(np.uint8)

    # Extract sign: 1 if negative, 0 otherwise
    sign = (arr_flat < 0).astype(np.uint8)

    # Extract scales: max abs value per block as FP16
    scales = np.max(np.abs(arr_flat), axis=1).astype(np.float16)

    # Pack into 2x uint64_t per block (128 bits = 16 bytes)
    # Each block: 128 bits packed into 2 × uint64
    def pack_bits(bits: np.ndarray) -> np.ndarray:
        # bits shape: (num_blocks, 128)
        # Pack 32 bits at a time using packbits, then reshape
        packed = np.packbits(bits, axis=1, bitorder='little')
        # packed shape: (num_blocks, 16) - 16 bytes
        # Reinterpret as 2 × uint64 (8 bytes each)
        packed_u64 = packed.view(np.uint64)
        return packed_u64.reshape(num_blocks, 2)

    magnitude_packed = pack_bits(magnitude)
    sign_packed = pack_bits(sign)

    return magnitude_packed, sign_packed, scales


def pack_weight(name: str, arr, out_path: Path) -> None:
    name_bytes = name.encode("utf-8")
    name_len = len(name_bytes)
    num_dims = len(arr.shape)
    dtype_code = 1  # 0 = float16, 1 = float32

    total_elements = arr.size
    data_bytes = arr.nbytes

    header_size = 4 + 4 + name_len + 4 + num_dims * 8 + 4

    with open(out_path, "wb") as f:
        f.write(struct.pack("<I", header_size))
        f.write(struct.pack("<I", name_len))
        f.write(name_bytes)
        f.write(struct.pack("<I", num_dims))
        for dim in arr.shape:
            f.write(struct.pack("<Q", dim))
        f.write(struct.pack("<I", dtype_code))
        f.write(arr.tobytes())

    print(f"Exported: {name}")
    print(f"  Shape: {list(arr.shape)}  dtype: {arr.dtype}  numel: {total_elements:,}")
    print(f"  Data bytes: {data_bytes:,}  Header: {header_size}")
    print(f"  Total file size: {header_size + data_bytes:,} bytes")
    print(f"  Saved to: {out_path}")


def pack_weight_ternary_g128(name: str, arr_shape: tuple, num_blocks: int,
                            magnitude, sign, scales,
                            out_path_mag: Path, out_path_sign: Path, out_path_scales: Path) -> None:
    name_bytes = name.encode("utf-8")
    name_len = len(name_bytes)
    dtype_code = 3  # 3 = packed ternary G128 (128 bits per block)

    shape = arr_shape
    num_dims = len(shape)
    total_elements = num_blocks * 128
    elements_per_block = 128

    header_size = 4 + 4 + name_len + 4 + num_dims * 8 + 4 + 4 + 4

    with open(out_path_mag, "wb") as f:
        f.write(struct.pack("<I", header_size))
        f.write(struct.pack("<I", name_len))
        f.write(name_bytes)
        f.write(struct.pack("<I", num_dims))
        for dim in shape:
            f.write(struct.pack("<Q", dim))
        f.write(struct.pack("<I", dtype_code))
        f.write(struct.pack("<I", total_elements))
        f.write(struct.pack("<I", elements_per_block))
        f.write(magnitude.tobytes())

    with open(out_path_sign, "wb") as f:
        f.write(struct.pack("<I", header_size))
        f.write(struct.pack("<I", name_len))
        f.write(name_bytes)
        f.write(struct.pack("<I", num_dims))
        for dim in shape:
            f.write(struct.pack("<Q", dim))
        f.write(struct.pack("<I", dtype_code))
        f.write(struct.pack("<I", total_elements))
        f.write(struct.pack("<I", elements_per_block))
        f.write(sign.tobytes())

    with open(out_path_scales, "wb") as f:
        f.write(struct.pack("<I", header_size))
        f.write(struct.pack("<I", name_len))
        f.write(name_bytes)
        f.write(struct.pack("<I", num_dims))
        for dim in shape:
            f.write(struct.pack("<Q", dim))
        f.write(struct.pack("<I", 0))  # dtype_code 0 = FP16 for scales
        f.write(struct.pack("<I", total_elements))
        f.write(struct.pack("<I", elements_per_block))
        f.write(scales.tobytes())

    print(f"Exported: {name}")
    print(f"  Shape: {list(shape)}  numel: {total_elements:,}  blocks: {num_blocks:,}")
    print(f"  Magnitude: {magnitude.nbytes:,} bytes ({magnitude.shape})")
    print(f"  Sign: {sign.nbytes:,} bytes ({sign.shape})")
    print(f"  Scales: {scales.nbytes:,} bytes (FP16)")
    print(f"  Saved magnitude to: {out_path_mag}")
    print(f"  Saved sign to: {out_path_sign}")
    print(f"  Saved scales to: {out_path_scales}")


LAYER_WEIGHTS = [
    "input_layernorm.weight",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
    "self_attn.o_proj.weight",
    "post_attention_layernorm.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
]


def export_batch(model_dir: Path, layers: list, packed: bool = False):
    safetensors_path = model_dir / "model.safetensors"
    print(f"Loading {safetensors_path} ...")
    tensors = safetensors.numpy.load_file(str(safetensors_path))

    total = 0
    for layer_idx in layers:
        for weight_suffix in LAYER_WEIGHTS:
            name = f"model.layers.{layer_idx}.{weight_suffix}"
            if name not in tensors:
                print(f"SKIP (not found): {name}")
                continue
            arr = tensors[name].astype(np.float32)
            safe_name = sanitize_name(name).replace(".", "_")

            if packed:
                if arr.size % 128 != 0:
                    print(f"SKIP {name}: size {arr.size} not divisible by 128")
                    continue
                magnitude, sign, scales = extract_ternary_g128(arr)
                num_blocks = arr.size // 128
                out_path_mag = model_dir / f"weight_{safe_name}_magnitude.bin"
                out_path_sign = model_dir / f"weight_{safe_name}_sign.bin"
                out_path_scales = model_dir / f"weight_{safe_name}_scales.bin"
                pack_weight_ternary_g128(name, arr.shape, num_blocks,
                                         magnitude, sign, scales,
                                         out_path_mag, out_path_sign, out_path_scales)
            else:
                out_path = model_dir / f"weight_{safe_name}.bin"
                pack_weight(name, arr, out_path)
            total += 1
    print(f"\nBatch export complete: {total} weights exported")


def main():
    parser = argparse.ArgumentParser(description="Export a model weight to binary float32.")
    parser.add_argument("weight_name", nargs="?", help="Full weight name (e.g. model.layers.0.mlp.up_proj.weight)")
    parser.add_argument("--model-dir", default="models/Ternary-Bonsai-1.7B-unpacked",
                        help="Path to model directory")
    parser.add_argument("--packed", action="store_true",
                        help="Export as packed ternary G128 (magnitude + sign + scales)")
    parser.add_argument("--layers", type=str, default=None,
                        help="Export all layer weights. Comma-sep list (e.g. '0-7' or '0,3,5'). "
                             "Includes embed_tokens and model.norm automatically.")
    parser.add_argument("--embed", action="store_true",
                        help="Export embed_tokens (use with --layers)")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)

    if args.layers is not None:
        # Parse layer range
        parts = args.layers.replace(" ", "").split(",")
        layers = []
        for p in parts:
            if "-" in p:
                start, end = p.split("-")
                layers.extend(range(int(start), int(end) + 1))
            else:
                layers.append(int(p))
        layers = sorted(set(layers))
        packed = args.packed
        print(f"Batch export for layers: {layers}, packed={packed}")

        # Export embed_tokens first
        if args.embed or True:
            name = "model.embed_tokens.weight"
            tensors = safetensors.numpy.load_file(str(model_dir / "model.safetensors"))
            if name in tensors:
                arr = tensors[name].astype(np.float32)
                safe_name = sanitize_name(name).replace(".", "_")
                if packed:
                    if arr.size % 128 == 0:
                        magnitude, sign, scales = extract_ternary_g128(arr)
                        num_blocks = arr.size // 128
                        out_path_mag = model_dir / f"weight_{safe_name}_magnitude.bin"
                        out_path_sign = model_dir / f"weight_{safe_name}_sign.bin"
                        out_path_scales = model_dir / f"weight_{safe_name}_scales.bin"
                        pack_weight_ternary_g128(name, arr.shape, num_blocks,
                                                 magnitude, sign, scales,
                                                 out_path_mag, out_path_sign, out_path_scales)
                    else:
                        print(f"SKIP (not divisible by 128): {name}")
                else:
                    out_path = model_dir / f"weight_{safe_name}.bin"
                    pack_weight(name, arr, out_path)
            else:
                print(f"SKIP (not found): {name}")

        # Export model.norm
        name = "model.norm.weight"
        tensors = safetensors.numpy.load_file(str(model_dir / "model.safetensors"))
        if name in tensors:
            arr = tensors[name].astype(np.float32)
            safe_name = sanitize_name(name).replace(".", "_")
            if packed:
                if arr.size % 128 == 0:
                    magnitude, sign, scales = extract_ternary_g128(arr)
                    num_blocks = arr.size // 128
                    out_path_mag = model_dir / f"weight_{safe_name}_magnitude.bin"
                    out_path_sign = model_dir / f"weight_{safe_name}_sign.bin"
                    out_path_scales = model_dir / f"weight_{safe_name}_scales.bin"
                    pack_weight_ternary_g128(name, arr.shape, num_blocks,
                                             magnitude, sign, scales,
                                             out_path_mag, out_path_sign, out_path_scales)
                else:
                    print(f"SKIP (not divisible by 128): {name}")
            else:
                out_path = model_dir / f"weight_{safe_name}.bin"
                pack_weight(name, arr, out_path)
        else:
            print(f"SKIP (not found): {name}")

        export_batch(model_dir, layers, packed)
        return

    if not args.weight_name:
        parser.print_help()
        return

    safetensors_path = model_dir / "model.safetensors"

    print(f"Loading {safetensors_path} ...")
    tensors = safetensors.numpy.load_file(str(safetensors_path))

    name = args.weight_name
    if name not in tensors:
        print(f"ERROR: weight '{name}' not found.")
        print("Available weights:")
        for n in sorted(tensors.keys()):
            print(f"  {n}")
        return

    arr = tensors[name].astype(np.float32)
    safe_name = sanitize_name(name).replace(".", "_")

    if arr.size % 128 != 0:
        raise ValueError(f"Array size {arr.size} is not divisible by 128")

    if args.packed:
        magnitude, sign, scales = extract_ternary_g128(arr)
        num_blocks = arr.size // 128
        out_path_mag = model_dir / f"weight_{safe_name}_magnitude.bin"
        out_path_sign = model_dir / f"weight_{safe_name}_sign.bin"
        out_path_scales = model_dir / f"weight_{safe_name}_scales.bin"
        pack_weight_ternary_g128(name, arr.shape, num_blocks,
                          magnitude, sign, scales,
                          out_path_mag, out_path_sign, out_path_scales)
    else:
        out_path = model_dir / f"weight_{safe_name}.bin"
        pack_weight(name, arr, out_path)


if __name__ == "__main__":
    main()
