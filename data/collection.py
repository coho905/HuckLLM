import os
import argparse
import numpy as np
from transformers import GPT2Tokenizer
from datasets import load_dataset


FLUSH_EVERY = 100_000  # flush to disk every 100k tokens


def stream_tokenize(dataset, tokenizer, output_path, target_tokens, text_field="text"):
    """Stream a dataset, tokenize on the fly, write to disk in batches."""
    buffer = []
    total = 0
    docs = 0
    
    print(f"  Target: {target_tokens:,} tokens")
    print(f"  Output: {output_path}")
    
    with open(output_path, "wb") as f:
        for example in dataset:
            text = example[text_field]
            if not text or len(text.strip()) == 0:
                continue
            tokens = tokenizer.encode(text)
            buffer.extend(tokens)
            docs += 1
            if len(buffer) >= FLUSH_EVERY:
                if total + len(buffer) >= target_tokens:
                    remaining = target_tokens - total
                    buffer = buffer[:remaining]
                
                f.write(np.array(buffer, dtype=np.uint16).tobytes())
                total += len(buffer)
                buffer = []
                if docs % 1000 == 0:
                    print(f"    {docs:,} docs | {total:,} tokens ({total/target_tokens*100:.1f}%)")
                if total >= target_tokens:
                    break
        
        if buffer and total < target_tokens:
            remaining = target_tokens - total
            buffer = buffer[:remaining]
            f.write(np.array(buffer, dtype=np.uint16).tobytes())
            total += len(buffer)
    
    size_gb = os.path.getsize(output_path) / 1e9
    print(f"  Done: {total:,} tokens | {docs:,} docs | {size_gb:.2f} GB on disk")
    return total


def merge_and_split(output_dir, shard_paths, val_fraction=0.005):    
    print("\n=== Merging shards ===")
    train_path = os.path.join(output_dir, "train.bin")
    val_path = os.path.join(output_dir, "val.bin")    
    total_tokens = 0
    for path in shard_paths:
        n = os.path.getsize(path) // 2  # uint16 = 2 bytes
        total_tokens += n
        print(f"  {os.path.basename(path)}: {n:,} tokens")
    val_tokens = max(int(total_tokens * val_fraction), 100_000)
    train_tokens = total_tokens - val_tokens
    print(f"\n  Total: {total_tokens:,}")
    print(f"  Train: {train_tokens:,}")
    print(f"  Val:   {val_tokens:,}")    
    written = 0
    CHUNK = 10_000_000  # read 10M tokens at a time
    
    with open(train_path, "wb") as f_train, open(val_path, "wb") as f_val:
        for path in shard_paths:
            data = np.fromfile(path, dtype=np.uint16)
            for i in range(0, len(data), CHUNK):
                chunk = data[i:i+CHUNK]
                if written + len(chunk) <= train_tokens:
                    f_train.write(chunk.tobytes())
                elif written < train_tokens:
                    split = train_tokens - written
                    f_train.write(chunk[:split].tobytes())
                    f_val.write(chunk[split:].tobytes())
                else:
                    f_val.write(chunk.tobytes())
                
                written += len(chunk)
            
            del data
    train_gb = os.path.getsize(train_path) / 1e9
    val_gb = os.path.getsize(val_path) / 1e9
    print(f"\n  train.bin: {train_gb:.2f} GB")
    print(f"  val.bin:   {val_gb:.3f} GB")    
    for path in shard_paths:
        os.remove(path)
        print(f"  Deleted {os.path.basename(path)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default=os.path.expanduser("~/data"))
    parser.add_argument("--total_tokens", type=int, default=2_000_000_000)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    T = args.total_tokens
    targets = {
        "pg19":        int(T * 0.47),
        "tinystories": int(T * 0.13),
        "minipile":    int(T * 0.40),
    }
    
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    actual = {}
    shard_paths = []
    
    print("\n=== pg19 ===")
    ds = load_dataset("deepmind/pg19", split="train", streaming=True)
    path = os.path.join(args.output_dir, "pg19.bin")
    actual["pg19"] = stream_tokenize(ds, tokenizer, path, targets["pg19"])
    shard_paths.append(path)
    
    print("\n=== TinyStories ===")
    ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    path = os.path.join(args.output_dir, "tinystories.bin")
    actual["tinystories"] = stream_tokenize(ds, tokenizer, path, targets["tinystories"])
    shard_paths.append(path)
    
    print("\n=== Minipile (30% of train split) ===")
    ds = load_dataset("JeanKaddour/minipile", split="train[:30%]", streaming=True)
    path = os.path.join(args.output_dir, "minipile.bin")
    actual["minipile"] = stream_tokenize(ds, tokenizer, path, targets["minipile"])
    shard_paths.append(path)
    
    merge_and_split(args.output_dir, shard_paths)
    total = sum(actual.values())
    print("\n=== Summary ===")
    for name, count in actual.items():
        pct = count / total * 100
        print(f"  {name:15s}: {count:>13,} tokens ({pct:.1f}%)")
    print(f"  {'TOTAL':15s}: {total:>13,} tokens")
    print(f"\n  Output files:")
    print(f"    {args.output_dir}/train.bin")
    print(f"    {args.output_dir}/val.bin")
    print(f"\n  Ready for training!")


if __name__ == "__main__":
    main()