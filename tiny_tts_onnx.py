"""
TinyTTS ONNX — CPU inference with no PyTorch/transformers/numba.
Downloads models from HuggingFace Hub on first use.
"""
import os
import re
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from g2p_en import G2p
from tokenizers import Tokenizer

_REPO = "backtracking/tiny-tts"

# Phoneme symbols from tiny-tts (English subset + punctuation + special)
_EN_SYMBOLS = [
    "aa", "ae", "ah", "ao", "aw", "ay", "b", "ch", "d", "dh",
    "eh", "er", "ey", "f", "g", "hh", "ih", "iy", "jh", "k",
    "l", "m", "n", "ng", "ow", "oy", "p", "r", "s", "sh", "t",
    "th", "uh", "uw", "V", "w", "y", "z", "zh",
]
_PUNCTUATION = ["!", "?", "…", ",", ".", "'", "-", "¿", "¡", "SP", "UNK"]
_SYMBOLS = ["_"] + sorted(set(_EN_SYMBOLS)) + _PUNCTUATION
_SYM_TO_ID = {s: i for i, s in enumerate(_SYMBOLS)}

# ARPAbet → tiny-tts symbol mapping (excluding stress markers)
_ARPA_MAP = {
    "AA": "aa", "AE": "ae", "AH": "ah", "AO": "ao", "AW": "aw", "AY": "ay",
    "B": "b", "CH": "ch", "D": "d", "DH": "dh", "EH": "eh", "ER": "er",
    "EY": "ey", "F": "f", "G": "g", "HH": "hh", "IH": "ih", "IY": "iy",
    "JH": "jh", "K": "k", "L": "l", "M": "m", "N": "n", "NG": "ng",
    "OW": "ow", "OY": "oy", "P": "p", "R": "r", "S": "s", "SH": "sh",
    "T": "t", "TH": "th", "UH": "uh", "UW": "uw", "V": "v", "W": "w",
    "Y": "y", "Z": "z", "ZH": "zh",
}
_ARPABET_SET = set(_ARPA_MAP.keys())

# English tone offset (from tiny_tts.text.symbols: ZH=6 + JP=1)
_EN_TONE_OFFSET = 7
_UNK_ID = _SYM_TO_ID["UNK"]
_PAD_ID = _SYM_TO_ID["_"]  # blank/padding symbol

# Punctuation mapping from tiny-tts
_PUNCT_MAP = {
    "：": ",", "；": ",", "，": ",", "。": ".", "！": "!",
    "？": "?", "\n": ".", "·": ",", "、": ",", "…": "…",
    "v": "V",
}

_TEXT_NORM_REPLACEMENTS = [
    (r"(\d+)\.(\d+)", r"\1 point \2"),
    (r"(\d+)\s*-\s*(\d+)", r"\1 to \2"),
]


def _insert_blanks(lst, item):
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result


def _parse_arpabet(ph):
    """Parse an ARPAbet phone string into (symbol, tone)."""
    m = re.match(r'^([A-Z]+)(\d)$', ph)
    if m:
        return m.group(1), int(m.group(2)) + 1
    return ph, 0


def _map_phoneme(symbol):
    """Map a phoneme/grapheme to a tiny-tts symbol."""
    if symbol in _PUNCT_MAP:
        symbol = _PUNCT_MAP[symbol]
    if symbol in _SYM_TO_ID:
        return symbol
    return "UNK"


class TinyTTSOnnx:
    """TinyTTS inference via ONNX Runtime, with no PyTorch/transformers."""

    def __init__(self):
        self.sample_rate = 44100
        self.frame_rate = 44100.0 / 512.0
        self.predefined_voices = ["MALE", "FEMALE"]

        model_path = hf_hub_download(_REPO, "tinytts_fp16.onnx")
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = os.cpu_count() or 2
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            model_path, sess_options=opts, providers=["CPUExecutionProvider"]
        )

        # Load BERT tokenizer for word-level tokenization (matches tiny-tts pipeline)
        try:
            tok_path = hf_hub_download("bert-base-uncased", "tokenizer.json")
            self.tokenizer = Tokenizer.from_file(tok_path)
        except Exception:
            self.tokenizer = None

        self.g2p = G2p()

    def _tokenize_words(self, text: str) -> list[str]:
        """Tokenize text into words using BERT WordPiece tokenizer,
        group subwords back into whole words for CMU dict lookup."""
        words = []
        if self.tokenizer:
            encoded = self.tokenizer.encode(text)
            tokens = encoded.tokens
            current = ""
            for t in tokens:
                if t in ("[CLS]", "[SEP]"):
                    continue
                if t.startswith("##"):
                    current += t[2:]
                else:
                    if current:
                        words.append(current)
                    current = t
            if current:
                words.append(current)
        else:
            words = text.split()
        return [w for w in words if w]

    def _text_to_ids(self, text: str):
        text = text.strip().lower()
        for pat, repl in _TEXT_NORM_REPLACEMENTS:
            text = re.sub(pat, repl, text)

        words = self._tokenize_words(text)
        phones = []
        tones = []

        for word in words:
            if not word:
                continue
            try:
                arpa_phones = self.g2p(word)
            except Exception:
                arpa_phones = []

            for ph in arpa_phones:
                if ph == " ":
                    continue
                if ph in _ARPABET_SET or re.match(r'^[A-Z]+\d$', ph):
                    arpa_base, tone = _parse_arpabet(ph)
                    tts_ph = _ARPA_MAP.get(arpa_base, arpa_base.lower())
                    phones.append(_map_phoneme(tts_ph))
                    tones.append(tone)
                elif ph in _PUNCTUATION or ph in _PUNCT_MAP:
                    tts_ph = _PUNCT_MAP.get(ph, ph)
                    phones.append(_map_phoneme(tts_ph))
                    tones.append(0)
                else:
                    phones.append(_map_phoneme(ph))
                    tones.append(0)

        phones = ["_"] + phones + ["_"]
        tones = [0] + tones + [0]

        phone_ids = [_SYM_TO_ID.get(ph, _UNK_ID) for ph in phones]
        tone_ids = [t + _EN_TONE_OFFSET for t in tones]
        lang_ids = [2] * len(phone_ids)

        phone_ids = _insert_blanks(phone_ids, _PAD_ID)
        tone_ids = _insert_blanks(tone_ids, 0)
        lang_ids = _insert_blanks(lang_ids, 0)

        return phone_ids, tone_ids, lang_ids

    def generate(self, text: str, voice: str = "MALE"):
        phone_ids, tone_ids, lang_ids = self._text_to_ids(text)
        T = len(phone_ids)

        sid = 0
        x = np.array(phone_ids, dtype=np.int64)[None, :]
        x_len = np.array([T], dtype=np.int64)
        tone = np.array(tone_ids, dtype=np.int64)[None, :]
        lang = np.array(lang_ids, dtype=np.int64)[None, :]
        bert = np.zeros((1, 1024, T), dtype=np.float32)
        ja_bert = np.zeros((1, 768, T), dtype=np.float32)
        sid_arr = np.array([sid], dtype=np.int64)
        noise_scale = np.array([0.667], dtype=np.float32)
        noise_scale_w = np.array([0.8], dtype=np.float32)
        length_scale = np.array([1.0], dtype=np.float32)

        audio = self.session.run(
            None,
            {
                "x": x,
                "x_lengths": x_len,
                "sid": sid_arr,
                "tone": tone,
                "language": lang,
                "bert": bert.astype(np.float16),
                "ja_bert": ja_bert.astype(np.float16),
                "noise_scale": noise_scale.astype(np.float16),
                "noise_scale_w": noise_scale_w.astype(np.float16),
                "length_scale": length_scale.astype(np.float16),
            },
        )[0]

        return audio[0, 0].astype(np.float32)

    def stream(self, text: str, voice: str = "MALE"):
        yield self.generate(text, voice)
