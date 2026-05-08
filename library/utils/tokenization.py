import abc
import json
import os
import re
from collections import OrderedDict
from typing import Any, Dict, List, Optional


def load_vocab(path: str) -> List[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Vocab file not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return list(data.keys())


class BaseVocabulary(abc.ABC):
    @property
    @abc.abstractmethod
    def vocab_size(self) -> int:
        ...

    @abc.abstractmethod
    def tokenize(self, x: Any) -> List[int]:
        ...

    @abc.abstractmethod
    def detokenize(self, ids: List[int], **kwargs: Any) -> Any:
        ...


CONTROL_PATTERN = re.compile(r"<\|[A-Z]+\|>")


class TextVocabulary(BaseVocabulary):
    EOS_TOKEN_STR = "<|EOS|>"
    PAD_TOKEN_STR = "<|PAD|>"
    BOS_TOKEN_STR = "<|BOS|>"
    UNK_TOKEN_STR = "<|UNK|>"

    EOS_ID = 0
    PAD_ID = 1
    BOS_ID = 2
    UNK_ID = 3

    def __init__(self, vocab_path: Optional[str] = None):
        self.tokens: List[str] = []
        if vocab_path is not None:
            if not os.path.exists(vocab_path):
                raise FileNotFoundError(vocab_path)
            with open(vocab_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.tokens = [t for t in data.keys() if t not in (self.EOS_TOKEN_STR, self.PAD_TOKEN_STR)]
        self._rebuild()

    def __getstate__(self) -> Dict[str, Any]:
        state = dict(self.__dict__)

        # The trie is a derived runtime cache. Pickling it can recurse very
        # deeply with large BPE/multilingual vocabularies.
        state.pop("root", None)

        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)

        # Do not call _rebuild() here. That may reorder/rederive token ids.
        # The checkpoint should preserve the exact saved token_map/index_map.
        self._build_trie()
    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def _rebuild(self) -> None:
        seen = set()
        uniq = []
        specials = (self.EOS_TOKEN_STR, self.PAD_TOKEN_STR, self.BOS_TOKEN_STR, self.UNK_TOKEN_STR)

        for t in self.tokens:
            if t not in seen and t not in specials:
                uniq.append(t)
                seen.add(t)
        self.tokens = uniq

        control = [t for t in self.tokens if CONTROL_PATTERN.fullmatch(t)]
        normal = [t for t in self.tokens if not CONTROL_PATTERN.fullmatch(t)]
        control.sort(key=len, reverse=True)
        normal.sort(key=len, reverse=True)
        ordered = control + normal

        self.token_map = {
            self.EOS_TOKEN_STR: self.EOS_ID,
            self.PAD_TOKEN_STR: self.PAD_ID,
            self.BOS_TOKEN_STR: self.BOS_ID,
            self.UNK_TOKEN_STR: self.UNK_ID,
        }

        # start AFTER the special ids
        start = max(self.token_map.values()) + 1
        for i, t in enumerate(ordered, start=start):
            self.token_map[t] = i

        self.index_map = {i: t for t, i in self.token_map.items()}
        self._vocab_size = len(self.token_map)
        self._build_trie()

    def add_tokens(self, new_tokens: List[str]) -> None:
        self.tokens.extend([t for t in new_tokens if t not in (self.EOS_TOKEN_STR, self.PAD_TOKEN_STR)])
        self._rebuild()

    def token_to_id(self, s: str) -> int:
        ids = self.tokenize(s)
        if not ids:
            raise ValueError("Empty input")
        return ids[0]

    def tokenize(self, text: str) -> List[int]:
        if not isinstance(text, str):
            text = str(text)

        out: List[int] = []
        n = len(text)
        i = 0

        while i < n:
            m = CONTROL_PATTERN.match(text, i)
            if m:
                tok = m.group(0)
                tid = self.token_map.get(tok)
                if tid is not None:
                    out.append(tid)
                    i = m.end()
                    continue

            node = self.root
            last_id = None
            last_pos = i
            j = i

            while j < n:
                ch = text[j]
                nxt = node.children.get(ch)
                if nxt is None:
                    break
                node = nxt
                j += 1
                if node.token_id is not None:
                    last_id, last_pos = node.token_id, j

            if last_id is not None:
                out.append(last_id)
                i = last_pos
            else:
                out.append(self.UNK_ID)  # could not match character to vocab
                i += 1

        return out

    def detokenize(self, ids: List[int], clean: bool = True) -> str:
        parts: List[str] = []
        for tid in ids:
            if clean and tid in (self.EOS_ID, self.PAD_ID, self.BOS_ID):
                continue
            tok = self.index_map.get(int(tid))
            if tok is None:
                raise ValueError(f"Bad token id {tid}")
            parts.append(tok)
        return "".join(parts)

    def compute_bytes_per_token(self, include_specials: bool = False) -> float:
        toks = list(self.token_map.keys()) if include_specials else [
            t for t in self.token_map.keys() if t not in (self.EOS_TOKEN_STR, self.PAD_TOKEN_STR, self.BOS_TOKEN_STR)
        ]
        if not toks:
            raise ValueError("Vocabulary is empty")
        bpt = sum(len(t.encode("utf-8")) for t in toks) / len(toks)
        self.bytes_per_token = float(bpt)
        return float(bpt)

    def save_json(self, file_path: str, indent: int = 4) -> None:
        os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
        export_map = OrderedDict(self.token_map)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(export_map, f, ensure_ascii=False, indent=indent, sort_keys=False)

    class _TrieNode:
        __slots__ = ("children", "token_id")

        def __init__(self):
            self.children: Dict[str, "TextVocabulary._TrieNode"] = {}
            self.token_id: Optional[int] = None

    def _build_trie(self) -> None:
        self.root = self._TrieNode()
        for tok, tid in self.token_map.items():
            node = self.root
            for ch in tok:
                node = node.children.setdefault(ch, self._TrieNode())
            node.token_id = tid


def export_vocab_state(vocab):
    return {
        "tokens": list(vocab.tokens),
        "token_map": dict(vocab.token_map),
        "index_map": dict(vocab.index_map),
    }


def rebuild_vocab(vocab_state):
    vocab = TextVocabulary()

    if "tokens" in vocab_state:
        vocab.tokens = list(vocab_state["tokens"])
        vocab._rebuild()
        return vocab

    specials = {
        vocab.EOS_TOKEN_STR,
        vocab.PAD_TOKEN_STR,
        vocab.BOS_TOKEN_STR,
        vocab.UNK_TOKEN_STR,
    }

    token_map = dict(vocab_state["token_map"])

    vocab.tokens = [
        tok for tok, _ in sorted(token_map.items(), key=lambda kv: int(kv[1]))
        if tok not in specials
    ]

    vocab._rebuild()
    return vocab