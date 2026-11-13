"""Decode the eight locked V25 sparse vectorizers without unpickling.

Only an explicit primitive pickle grammar plus inline numeric joblib arrays
is accepted. No pickle runtime, imports from the payload, object construction,
setstate, reducer invocation, memmap or sidecar files occur during decoding.
The separately trusted sklearn class is built from validated values afterward.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from io import BytesIO
import math
from pathlib import Path
import pickletools
import re
import zlib


_LOCKS = {
    "distort/vectorizer-0.joblib": (709505, "141df312a79bbaf717af891394e0a7556c0d186a9272641c06c69ebd1c0f09d4"),
    "distort/vectorizer-1.joblib": (690266, "26723d9fa1d83084533d0b401e93c519812421dcfcaf0aab5c538c61aacf6f41"),
    "svc4/vectorizer-0.joblib": (2532156, "81739e3bcf94930db252f151310301dc50e52b805602139fbeb84a3d87aa866e"),
    "svc4/vectorizer-1.joblib": (2276970, "c930c731c24cf492eb0ea8ffe5b9cdbaba806f5a74da0afbb91d9ee2ad06ae57"),
    "svc4/vectorizer-2.joblib": (510409, "3b0940a3e3679b7b4fe961ea7742bf68f21a0759c7a0c882c12f5feba60cde23"),
    "svc4/vectorizer-3.joblib": (544486, "a322a5aa10e7383026d1f3f9393125cb7712a9404cf8d8597ab1c099fffc43aa"),
    "tfidf/vectorizer-0.joblib": (612734, "3f5a5325c9ebe8be92916e8dffb8dd98b2d3682c921769304db395b612b585e7"),
    "tfidf/vectorizer-1.joblib": (357394, "fc28806e37a28893cc3ed362096b4e29c587b5086f92419a4e5d06b1d3433405"),
}
_MAX_SOURCE = 16 * 1024 * 1024
_MAX_DECODED = 48 * 1024 * 1024
_MAX_FEATURES = 400000
_MAX_OPS = 8000000
_MAX_MEMO = 2000000
_MAX_STACK = 10000
_MAX_TOKEN = 4096
_TFIDF = "sklearn.feature_extraction.text"
_GLOBALS = {
    (_TFIDF, "TfidfVectorizer"): "vectorizer",
    (_TFIDF, "TfidfTransformer"): "transformer",
    ("joblib.numpy_pickle", "NumpyArrayWrapper"): "array_wrapper",
    ("numpy", "ndarray"): "ndarray",
    ("numpy", "float32"): "float32",
    ("numpy", "float64"): "float64",
    ("numpy", "dtype"): "dtype_constructor",
    ("numpy._core.multiarray", "scalar"): "integer_scalar",
    ("numpy.core.multiarray", "scalar"): "integer_scalar",
}
_TOKEN_PATTERNS = frozenset({r"(?u)\b\w\w+\b", r"(?u)\b\w+\b|[^\w\s]",
                            r"\w+|[^\w\s]", r"[^\s]+", r"\S+"})
_OPERATIONS = {p.code.encode("latin1"): p for p in pickletools.opcodes}


class RejectedVectorizer(ValueError):
    """A rejected fingerprint, serialization, or vectorizer schema."""


def _require(ok, message):
    if not ok:
        raise RejectedVectorizer(message)


@dataclass(frozen=True)
class _Global:
    name: str


@dataclass(eq=False)
class _Record:
    kind: str
    state: object = None


@dataclass(frozen=True)
class _Array:
    code: str
    endian: str
    size: int
    data: bytes


_MARK = object()


def _source(raw):
    _require(type(raw) is bytes and 2 <= len(raw) <= _MAX_SOURCE, "unexpected serialized size")
    if raw.startswith((b"\x80\x04", b"\x80\x05")):
        return raw
    # Only standard zlib/gzip accepted. Reject bzip, xz, legacy companions,
    # concatenated streams, oversized output, and dictionary streams.
    if raw[:2] == b"\x1f\x8b":
        mode = 31
    elif raw[0] == 0x78 and int.from_bytes(raw[:2], "big") % 31 == 0:
        mode = 15
    else:
        raise RejectedVectorizer("unknown compression or pickle protocol")
    try:
        codec = zlib.decompressobj(mode)
        data = codec.decompress(raw, _MAX_DECODED + 1)
    except zlib.error as exc:
        raise RejectedVectorizer("bad compressed stream") from exc
    _require(len(data) <= _MAX_DECODED and codec.eof and not codec.unused_data and not codec.unconsumed_tail,
             "oversized, truncated or trailing compressed data")
    _require(data.startswith((b"\x80\x04", b"\x80\x05")), "unknown pickle protocol")
    return data


class _Machine:
    def __init__(self, raw):
        self.stream = BytesIO(_source(raw))
        self.frame_end = None
        self.stack, self.memo, self.marks = [], [], []
        self.array_count = 0

    def _read(self, n):
        _require(type(n) is int and 0 <= n <= _MAX_DECODED, "unbounded stream read")
        data = self.stream.read(n)
        _require(len(data) == n, "truncated stream")
        return data

    def _pop_mark(self):
        _require(self.marks, "missing MARK")
        start = self.marks.pop()
        _require(self.stack[start] is _MARK, "corrupt MARK")
        values = self.stack[start + 1:]
        del self.stack[start:]
        return values

    def _build(self):
        state, target = self.stack.pop(), self.stack[-1]
        _require(type(target) is _Record and target.state is None, "invalid or repeated BUILD target")
        if target.kind.startswith("dtype:"):
            _require(type(state) is tuple and len(state) == 8 and type(state[0]) is int and state[0] == 3
                     and state[1] in {"<", ">", "="} and state[2:] == (None, None, None, -1, -1, 0),
                     "only primitive numeric dtype is permitted")
            target.state = state[1]
        elif target.kind == "array_wrapper":
            _require(type(state) is dict and set(state) == {"subclass", "shape", "order", "dtype", "allow_mmap",
                                                            "numpy_array_alignment_bytes"}, "invalid ndarray wrapper")
            dtype, shape = state["dtype"], state["shape"]
            _require(state["subclass"] == _Global("ndarray") and state["order"] in ("C", "F")
                     and type(state["allow_mmap"]) is bool and state["numpy_array_alignment_bytes"] == 16
                     and type(dtype) is _Record and dtype.kind in {"dtype:f4", "dtype:f8"} and dtype.state is not None
                     and type(shape) is tuple and len(shape) == 1 and type(shape[0]) is int
                     and 1 <= shape[0] <= _MAX_FEATURES and self.array_count == 0, "invalid or excess inline numeric array")
            pos = self.stream.tell()
            _require(self.frame_end in (None, pos), "inline array begins inside a pickle frame")
            self.frame_end = None
            padding = self._read(1)[0]
            _require(padding == 16 - (pos + 1) % 16 and self._read(padding) == b"\xff" * padding,
                     "bad joblib ndarray alignment")
            code = dtype.kind[-2:]
            self.stack[-1] = _Array(code, dtype.state, shape[0], self._read(shape[0] * int(code[-1])))
            target.state = "consumed"
            self.array_count += 1
        else:
            _require(target.kind in {"vectorizer", "transformer"} and type(state) is dict
                     and len(state) <= 40 and all(type(k) is str for k in state), "invalid sklearn state")
            target.state = state

    def _reduce(self):
        args, fn = self.stack.pop(), self.stack.pop()
        _require(type(args) is tuple and type(fn) is _Global, "invalid reducer operands")
        if fn.name == "dtype_constructor":
            _require(len(args) == 3 and type(args[0]) is str and args[0] in {"f4", "f8", "i4", "i8"}
                     and args[1] is False and args[2] is True, "unapproved dtype arguments")
            self.stack.append(_Record("dtype:" + args[0]))
        elif fn.name == "integer_scalar":
            _require(len(args) == 2 and type(args[0]) is _Record and args[0].kind in {"dtype:i4", "dtype:i8"}
                     and args[0].state in {"<", ">", "="} and type(args[1]) is bytes
                     and len(args[1]) == int(args[0].kind[-1]), "unapproved NumPy scalar")
            self.stack.append(int.from_bytes(args[1], "big" if args[0].state == ">" else "little", signed=True))
        else:
            raise RejectedVectorizer("unapproved reducer")

    def _pairs(self, items):
        target = self.stack[-1]
        _require(type(target) is dict and len(items) % 2 == 0 and len(target) + len(items) // 2 <= _MAX_FEATURES,
                 "invalid dictionary opcode")
        for i in range(0, len(items), 2):
            k = items[i]
            _require(type(k) is str and k not in target, "non-string or duplicate dictionary key")
            target[k] = items[i + 1]

    def run(self):
        _require(self._read(1) == b"\x80" and self._read(1) in (b"\x04", b"\x05"), "invalid protocol header")
        for _ in range(_MAX_OPS):
            pos = self.stream.tell()
            if self.frame_end == pos:
                self.frame_end = None
            key = self._read(1)
            _require(key in _OPERATIONS, "unknown serialization opcode")
            op = _OPERATIONS[key].name
            if op == "FRAME":
                size = int.from_bytes(self._read(8), "little")
                _require(self.frame_end is None and 0 < size <= _MAX_DECODED, "invalid pickle frame")
                self.frame_end = self.stream.tell() + size
                continue
            if op in {"BININT", "BININT1", "BININT2", "BINFLOAT", "BINGET", "LONG_BINGET", "LONG1",
                      "SHORT_BINUNICODE", "BINUNICODE", "SHORT_BINBYTES"}:
                arg = _OPERATIONS[key].arg.reader(self.stream)
                if type(arg) is str:
                    _require(len(arg) <= _MAX_TOKEN, "oversized string")
                if type(arg) is bytes:
                    _require(len(arg) <= 8, "oversized scalar bytes")
                if type(arg) is int:
                    _require(-(2**63) <= arg < 2**64, "oversized integer")
                self.stack.append(self.memo[arg] if op in {"BINGET", "LONG_BINGET"} else arg)
            elif op == "MARK":
                self.marks.append(len(self.stack))
                self.stack.append(_MARK)
            elif op == "MEMOIZE":
                _require(len(self.memo) < _MAX_MEMO, "memo budget exceeded")
                self.memo.append(self.stack[-1])
            elif op == "STACK_GLOBAL":
                name, module = self.stack.pop(), self.stack.pop()
                _require(type(name) is type(module) is str and (module, name) in _GLOBALS, "unapproved serialized global")
                self.stack.append(_Global(_GLOBALS[(module, name)]))
            elif op == "NEWOBJ":
                args, cls = self.stack.pop(), self.stack.pop()
                _require(type(args) is tuple and not args and type(cls) is _Global
                         and cls.name in {"vectorizer", "transformer", "array_wrapper"}, "unapproved object construction")
                self.stack.append(_Record(cls.name))
            elif op == "REDUCE":
                self._reduce()
            elif op == "BUILD":
                _require(self.frame_end is None or self.stream.tell() <= self.frame_end, "pickle frame crossed")
                self._build()
            elif op == "SETITEMS":
                self._pairs(self._pop_mark())
            elif op == "SETITEM":
                v, k = self.stack.pop(), self.stack.pop()
                self._pairs([k, v])
            elif op in {"TUPLE", "FROZENSET", "ADDITEMS"}:
                items = self._pop_mark()
                if op == "TUPLE":
                    self.stack.append(tuple(items))
                else:
                    _require(all(type(v) is str for v in items), "set of non-string values")
                    if op == "FROZENSET":
                        self.stack.append(frozenset(items))
                    else:
                        _require(type(self.stack[-1]) is set and len(self.stack[-1]) + len(items) <= _MAX_FEATURES,
                                 "invalid set opcode")
                        self.stack[-1].update(items)
            elif op in {"TUPLE1", "TUPLE2", "TUPLE3"}:
                n = int(op[-1])
                values = tuple(self.stack[-n:])
                _require(len(values) == n and all(v is not _MARK for v in values), "invalid tuple operands")
                del self.stack[-n:]
                self.stack.append(values)
            elif op in {"EMPTY_DICT", "EMPTY_TUPLE", "EMPTY_SET", "NONE", "NEWTRUE", "NEWFALSE"}:
                self.stack.append({"EMPTY_DICT": {}, "EMPTY_TUPLE": (), "EMPTY_SET": set(),
                                   "NONE": None, "NEWTRUE": True, "NEWFALSE": False}[op])
            elif op == "STOP":
                _require(not self.marks and len(self.stack) == 1 and self.frame_end in (None, self.stream.tell())
                         and not self.stream.read(1), "invalid end or trailing serialized bytes")
                return self.stack[0]
            else:
                raise RejectedVectorizer("unapproved serialization opcode: " + op)
            _require(self.frame_end is None or self.stream.tell() <= self.frame_end, "pickle frame crossed")
            _require(len(self.stack) <= _MAX_STACK and len(self.marks) <= 16, "serialization stack budget exceeded")
        raise RejectedVectorizer("serialization operation budget exceeded")


_PARAMS = frozenset({"input", "encoding", "decode_error", "strip_accents", "preprocessor", "tokenizer", "analyzer",
                     "lowercase", "token_pattern", "stop_words", "max_df", "min_df", "max_features", "ngram_range",
                     "vocabulary", "binary", "dtype", "norm", "use_idf", "smooth_idf", "sublinear_tf"})
_FLAGS = {"lowercase", "binary", "use_idf", "smooth_idf", "sublinear_tf"}


def decode_vectorizer(data: bytes):
    """Return validated vectorizer parameters, vocabulary, and IDF array."""
    import numpy as np

    try:
        result = _Machine(data).run()
    except (IndexError, ValueError, TypeError, OverflowError) as exc:
        if isinstance(exc, RejectedVectorizer):
            raise
        raise RejectedVectorizer("malformed serialized vectorizer") from exc
    _require(type(result) is _Record and result.kind == "vectorizer" and type(result.state) is dict,
             "expected a plain TF-IDF vectorizer")
    state = result.state
    _require(_PARAMS | {"_tfidf", "vocabulary_", "fixed_vocabulary_", "_sklearn_version"} <= set(state)
             <= _PARAMS | {"_tfidf", "vocabulary_", "fixed_vocabulary_", "_sklearn_version", "stop_words_", "_stop_words_id"},
             "unexpected TF-IDF vectorizer state keys")
    p = {k: state[k] for k in _PARAMS}
    _require(p["input"] == "content" and p["encoding"] == "utf-8" and p["decode_error"] in {"strict", "ignore", "replace"}
             and p["strip_accents"] in {None, "unicode", "ascii"} and p["preprocessor"] is p["tokenizer"] is None
             and p["analyzer"] in {"word", "char", "char_wb"} and p["norm"] in {None, "l1", "l2"}
             and p["stop_words"] in (None, "english") and p["token_pattern"] in _TOKEN_PATTERNS
             and all(type(p[k]) is bool for k in _FLAGS) and p["use_idf"]
             and state["fixed_vocabulary_"] is False and p["vocabulary"] is None, "unapproved TF-IDF behavior")
    _require(type(p["ngram_range"]) is tuple and len(p["ngram_range"]) == 2
             and all(type(v) is int for v in p["ngram_range"]) and 1 <= p["ngram_range"][0] <= p["ngram_range"][1] <= 8
             and (p["max_features"] is None or type(p["max_features"]) is int and 1 <= p["max_features"] <= _MAX_FEATURES)
             and all((type(p[k]) is int and 1 <= p[k] <= _MAX_FEATURES)
                     or (type(p[k]) is float and math.isfinite(p[k]) and 0 <= p[k] <= 1)
                     for k in ("min_df", "max_df")), "unapproved TF-IDF numerical parameters")
    vocab = state["vocabulary_"]
    _require(type(vocab) is dict and 0 < len(vocab) <= _MAX_FEATURES and all(type(v) is int for v in vocab.values())
             and set(vocab.values()) == set(range(len(vocab))), "invalid vocabulary axis")
    _require("_stop_words_id" not in state or type(state["_stop_words_id"]) is int, "invalid unused stop word ID")
    _require("stop_words_" not in state or type(state["stop_words_"]) in (set, frozenset), "invalid unused stop words")
    version = state["_sklearn_version"]
    _require(type(version) is str and re.fullmatch(r"[0-9]+\.[0-9]+(?:\.[0-9]+)?", version) is not None,
             "unapproved sklearn serialization version")
    transformer = state["_tfidf"]
    _require(type(transformer) is _Record and transformer.kind == "transformer" and type(transformer.state) is dict,
             "invalid fitted TF-IDF transformer")
    inner = transformer.state
    expected = {"norm", "use_idf", "smooth_idf", "sublinear_tf"}
    _require(set(inner) == expected | {"n_features_in_", "idf_", "_sklearn_version"}
             and all(type(inner[k]) is type(p[k]) and inner[k] == p[k] for k in expected)
             and inner["_sklearn_version"] == version and type(inner["n_features_in_"]) is int
             and inner["n_features_in_"] == len(vocab), "invalid fitted TF-IDF transformer axes")
    a = inner["idf_"]
    _require(type(a) is _Array and a.size == len(vocab) and p["dtype"] == _Global({"f4": "float32", "f8": "float64"}[a.code]),
             "IDF/vocabulary/dtype mismatch")
    dtype = np.float32 if a.code == "f4" else np.float64
    idf = np.frombuffer(a.data, dtype=(">" if a.endian == ">" else "<") + a.code).astype(dtype, copy=True)
    _require(np.isfinite(idf).all() and (idf >= 1).all() and (idf <= 64).all(), "invalid IDF range")
    p["dtype"] = dtype
    return p, vocab, idf, version


def _locked_data(path):
    path = Path(path)
    key = "/".join(path.parts[-2:])
    _require(path.parent.parent.name == "core" and key in _LOCKS, "unapproved prefit vectorizer path")
    size, fingerprint = _LOCKS[key]
    with path.open("rb") as source:
        data = source.read(size + 1)
    _require(len(data) == size and hashlib.sha256(data).hexdigest() == fingerprint, "prefit vectorizer bytes changed")
    return data, key, fingerprint


def inspect_prefit_vectorizer(path):
    """Inspect a pinned file and return only aggregate structural evidence."""
    data, key, fingerprint = _locked_data(path)
    p, vocab, idf, version = decode_vectorizer(data)
    return {"member": key, "sha256": fingerprint, "analyzer": p["analyzer"], "ngram_range": list(p["ngram_range"]),
            "features": len(vocab), "idf_dtype": str(idf.dtype), "source_sklearn_version": version,
            "unpickling_performed": False, "serialized_callable_invoked": False}


def build_vectorizer(parameters, vocabulary, idf):
    """Reconstruct from decoded values; never populate arbitrary attributes."""
    from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer

    model = TfidfVectorizer(**parameters)
    model.vocabulary_ = vocabulary
    model.fixed_vocabulary_ = False
    transformer = TfidfTransformer(**{k: parameters[k] for k in ("norm", "use_idf", "smooth_idf", "sublinear_tf")})
    transformer.n_features_in_ = len(vocabulary)
    transformer.idf_ = idf
    model._tfidf = transformer
    return model


def load_prefit_vectorizer(path):
    """Replacement for the eight source `joblib.load` calls only."""
    data, _, _ = _locked_data(path)
    parameters, vocabulary, idf, _ = decode_vectorizer(data)
    return build_vectorizer(parameters, vocabulary, idf)
