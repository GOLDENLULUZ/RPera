import base64
import gzip
import hashlib
from functools import lru_cache
from pathlib import Path

import tiktoken


ENCODING_NAME = "o200k_base"
_VOCABULARY_HASH = "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d"
_PATTERN = "|".join(
    [
        r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
        r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
        r"\p{N}{1,3}",
        r" ?[^\s\p{L}\p{N}]+[\r\n/]*",
        r"\s*[\r\n]+",
        r"\s+(?!\S)",
        r"\s+",
    ]
)


@lru_cache(maxsize=1)
def _encoding() -> tiktoken.Encoding:
    vocabulary = Path(__file__).with_name("data") / "o200k_base.tiktoken.gz"
    vocabulary_data = gzip.decompress(vocabulary.read_bytes())
    if hashlib.sha256(vocabulary_data).hexdigest() != _VOCABULARY_HASH:
        raise RuntimeError("内置 o200k_base 词表校验失败")
    mergeable_ranks = {
        base64.b64decode(token): int(rank)
        for line in vocabulary_data.splitlines()
        for token, rank in [line.split()]
    }
    return tiktoken.Encoding(
        name=ENCODING_NAME,
        pat_str=_PATTERN,
        mergeable_ranks=mergeable_ranks,
        special_tokens={"<|endoftext|>": 199_999, "<|endofprompt|>": 200_018},
    )


def count_narrative_tokens(content: str) -> int:
    return len(_encoding().encode(content, disallowed_special=()))
