"""File IO compatibility only; preserve the original SentencePiece model bytes."""
from pathlib import Path


def load_tokenizer(model_dir):
    import sentencepiece as spm
    from transformers import AutoTokenizer
    original = spm.SentencePieceProcessor.LoadFromFile

    def load_unicode_path(processor, filename):
        # sentencepiece 0.1.99 on Windows cannot open non-ASCII filenames.
        # Python reads the exact same verified protobuf and hands it to C++.
        return processor.LoadFromSerializedProto(Path(filename).read_bytes())

    spm.SentencePieceProcessor.LoadFromFile = load_unicode_path
    try:
        return AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, local_files_only=True)
    finally:
        spm.SentencePieceProcessor.LoadFromFile = original
