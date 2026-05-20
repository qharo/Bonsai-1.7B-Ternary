"""
TinyTTS ONNX — CPU inference, no PyTorch/transformers/numba.
Model and tokenizer loaded from models/tinytts/ (local, not HF Hub).
"""
import os, re
import numpy as np
import onnxruntime as ort
from g2p_en import G2p
from tokenizers import Tokenizer

_MODEL_DIR = os.path.join(os.path.dirname(__file__), "models", "tinytts")

# ── Full multilingual phoneme table (from tiny_tts.text.symbols) ──

_ZH_SYMBOLS = ["E","En","a","ai","an","ang","ao","b","c","ch","d","e","ei","en","eng","er","f","g","h","i","i0","ia","ian","iang","iao","ie","in","ing","iong","ir","iu","j","k","l","m","n","o","ong","ou","p","q","r","s","sh","t","u","ua","uai","uan","uang","ui","un","uo","v","van","ve","vn","w","x","y","z","zh","AA","EE","OO"]
_JA_SYMBOLS = ["N","a","a:","b","by","ch","d","dy","e","e:","f","g","gy","h","hy","i","i:","j","k","ky","m","my","n","ny","o","o:","p","py","q","r","ry","s","sh","t","ts","ty","u","u:","w","y","z","zy"]
_EN_SYMBOLS = ["aa","ae","ah","ao","aw","ay","b","ch","d","dh","eh","er","ey","f","g","hh","ih","iy","jh","k","l","m","n","ng","ow","oy","p","r","s","sh","t","th","uh","uw","V","w","y","z","zh"]
_KR_SYMBOLS = ['ᄌ','ᅥ','ᆫ','ᅦ','ᄋ','ᅵ','ᄅ','ᅴ','ᄀ','ᅡ','ᄎ','ᅪ','ᄑ','ᅩ','ᄐ','ᄃ','ᅢ','ᅮ','ᆼ','ᅳ','ᄒ','ᄆ','ᆯ','ᆷ','ᄂ','ᄇ','ᄉ','ᆮ','ᄁ','ᅬ','ᅣ','ᄄ','ᆨ','ᄍ','ᅧ','ᄏ','ᆸ','ᅭ','(','ᄊ',')','ᅲ','ᅨ','ᄈ','ᅱ','ᅯ','ᅫ','ᅰ','ᅤ','~','\\','[',']','/','^',':','ㄸ','*']
_ES_SYMBOLS = ["N","Q","a","b","d","e","f","g","h","i","j","k","l","m","n","o","p","s","t","u","v","w","x","y","z","ɑ","æ","ʃ","ʑ","ç","ɯ","ɪ","ɔ","ɛ","ɹ","ð","ə","ɫ","ɥ","ɸ","ʊ","ɾ","ʒ","θ","β","ŋ","ɦ","ɡ","r","ɲ","ʝ","ɣ","ʎ","ˈ","ˌ","ː"]
_FR_SYMBOLS = ["\u0303","œ","ø","ʁ","ɒ","ʌ","ɜ","ɐ"]
_DE_SYMBOLS = ["ʏ","̩"]
_RU_SYMBOLS = ["ɭ","ʲ","ɕ","\"","ɵ","^","ɬ"]

_PUNCTUATION = ["!", "?", "…", ",", ".", "'", "-", "¿", "¡", "SP", "UNK"]

_NORMAL_SYMBOLS = sorted(set(_ZH_SYMBOLS + _JA_SYMBOLS + _EN_SYMBOLS + _KR_SYMBOLS + _ES_SYMBOLS + _FR_SYMBOLS + _DE_SYMBOLS + _RU_SYMBOLS))
_SYMBOLS = ["_"] + _NORMAL_SYMBOLS + _PUNCTUATION
_SYM_TO_ID = {s: i for i, s in enumerate(_SYMBOLS)}

# ── Constants ──
_UNK_ID = _SYM_TO_ID["UNK"]
_PAD_ID = _SYM_TO_ID["_"]
_EN_TONE_OFFSET = 7  # num_zh_tones(6) + num_ja_tones(1)

# ── ARPAbet → tiny-tts mapping ──
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

_PUNCT_MAP = {
    "：": ",", "；": ",", "，": ",", "。": ".", "！": "!",
    "？": "?", "\n": ".", "·": ",", "、": ",", "…": "…", "v": "V",
}


def _insert_blanks(lst, item):
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result


def _parse_arpabet(ph):
    m = re.match(r'^([A-Z]+)(\d)$', ph)
    if m:
        return m.group(1), int(m.group(2)) + 1
    return ph, 0


def _map_phoneme(symbol):
    if symbol in _PUNCT_MAP:
        symbol = _PUNCT_MAP[symbol]
    if symbol in _SYM_TO_ID:
        return symbol
    return "UNK"


class TinyTTSOnnx:

    def __init__(self):
        self.sample_rate = 44100
        self.frame_rate = 44100.0 / 512.0
        self.predefined_voices = ["MALE", "FEMALE"]

        onnx_path = os.path.join(_MODEL_DIR, "tinytts_fp16.onnx")
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = os.cpu_count() or 2
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            onnx_path, sess_options=opts, providers=["CPUExecutionProvider"]
        )

        tok_path = os.path.join(_MODEL_DIR, "tokenizer.json")
        if os.path.exists(tok_path):
            self.tokenizer = Tokenizer.from_file(tok_path)
        else:
            self.tokenizer = None

        self.g2p = G2p()

    def _tokenize_words(self, text: str) -> list[str]:
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
        text = re.sub(r"(\d+)\.(\d+)", r"\1 point \2", text)
        text = re.sub(r"(\d+)\s*-\s*(\d+)", r"\1 to \2", text)

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

    def phonemize(self, text: str):
        """
        Convert text to phoneme IDs (without padding/blanks).
        This is FAST (~1ms per token) and can run during LLM generation.
        Returns tuple: (phone_ids, tone_ids, lang_ids) - lists of integers
        """
        text = text.strip().lower()
        text = re.sub(r"(\d+)\.(\d+)", r"\1 point \2", text)
        text = re.sub(r"(\d+)\s*-\s*(\d+)", r"\1 to \2", text)

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

        return phone_ids, tone_ids, lang_ids

    def generate_from_phones(self, phone_ids, tone_ids, lang_ids, voice: str = "MALE"):
        """
        Generate audio from pre-computed phoneme IDs.
        This skips the g2p phonemization step for faster inference.
        Args:
            phone_ids: list of phone IDs (already padded with blanks)
            tone_ids: list of tone IDs (already padded with blanks)
            lang_ids: list of language IDs (already padded with blanks)
        """
        T = len(phone_ids)

        x = np.array(phone_ids, dtype=np.int64)[None, :]
        x_len = np.array([T], dtype=np.int64)
        tone = np.array(tone_ids, dtype=np.int64)[None, :]
        lang = np.array(lang_ids, dtype=np.int64)[None, :]
        bert = np.zeros((1, 1024, T), dtype=np.float16)
        ja_bert = np.zeros((1, 768, T), dtype=np.float16)
        sid_arr = np.array([0], dtype=np.int64)
        noise_scale = np.array([0.667], dtype=np.float16)
        noise_scale_w = np.array([0.8], dtype=np.float16)
        length_scale = np.array([1.0], dtype=np.float16)

        audio = self.session.run(None, {
            "x": x, "x_lengths": x_len, "sid": sid_arr,
            "tone": tone, "language": lang,
            "bert": bert, "ja_bert": ja_bert,
            "noise_scale": noise_scale,
            "noise_scale_w": noise_scale_w,
            "length_scale": length_scale,
        })[0]

        return audio[0, 0].astype(np.float32)

    def generate(self, text: str, voice: str = "MALE"):
        phone_ids, tone_ids, lang_ids = self._text_to_ids(text)
        return self.generate_from_phones(phone_ids, tone_ids, lang_ids, voice)

    def stream(self, text: str, voice: str = "MALE"):
        yield self.generate(text, voice)
