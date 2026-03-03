import re
import unicodedata
from typing import List, Tuple


class EuropeanNameNormalizer:
    """
    Pragmatic normalization for European personal names for *matching* (not display).

    Produces:
      - primary: NFKC + casefold + punctuation/whitespace normalization
      - ascii_fallback: primary + diacritic stripping + common expansions
      - optional German transliteration: ä->ae, ö->oe, ü->ue (and uppercase variants)

    Notes:
      - Keep original input elsewhere; this is only for lookup keys.
      - ASCII fallback increases recall when sources disagree on diacritics.
    """

    _APOSTROPHES = {
        "\u2019", "\u2018", "\u02BC", "\uFF07", "`",
    }

    _HYPHENS = {
        "\u2010", "\u2011", "\u2012", "\u2013", "\u2014", "\u2212", "\u00AD",
    }

    # Common Latin letter expansions frequently seen in European names
    _EXPAND = {
        "ß": "ss",
        "ẞ": "ss",
        "æ": "ae",
        "Æ": "ae",
        "œ": "oe",
        "Œ": "oe",
        "ø": "o",
        "Ø": "o",
        "ð": "d",
        "Ð": "d",
        "þ": "th",
        "Þ": "th",
        "ł": "l",
        "Ł": "l",
        "đ": "d",
        "Đ": "d",
    }

    # German-style transliteration (requested)
    _DE = {
        "ä": "ae",
        "ö": "oe",
        "ü": "ue",
        "Ä": "ae",
        "Ö": "oe",
        "Ü": "ue",
        # ß is already handled in _EXPAND
    }

    def __init__(self, *, keep_apostrophe: bool = False, de_transliteration: bool = False) -> None:
        self.keep_apostrophe = keep_apostrophe
        self.de_transliteration = de_transliteration

    def variants(self, raw: str) -> Tuple[str, ...]:
        """
        Return candidate lookup keys, ordered from most faithful -> most permissive.
        """
        primary = self.primary(raw)
        if not primary:
            return tuple()

        ascii_key = self.ascii_fallback(primary)

        out: List[str] = []
        for v in (primary, ascii_key):
            if v and v not in out:
                out.append(v)
        return tuple(out)

    def primary(self, raw: str) -> str:
        if not raw:
            return ""

        s = raw.strip()
        if not s:
            return ""

        # Compatibility normalize first
        s = unicodedata.normalize("NFKC", s)

        # Normalize apostrophes/hyphens to stable forms
        for ch in self._APOSTROPHES:
            s = s.replace(ch, "'")
        for ch in self._HYPHENS:
            s = s.replace(ch, "-")

        # Optional DE transliteration BEFORE casefold (so Ä/Ö/Ü handled too)
        if self.de_transliteration:
            for k, v in self._DE.items():
                s = s.replace(k, v)

        # Case-insensitive matching
        s = s.casefold()

        # Separators/punctuation -> spaces (or keep apostrophe if configured)
        s = s.replace("-", " ")
        s = re.sub(r"[.,/\\]+", " ", s)
        if self.keep_apostrophe:
            s = re.sub(r"\s*'\s*", "'", s)
        else:
            s = s.replace("'", " ")

        # Drop other punctuation/symbols (keep word chars, spaces, combining marks)
        s = re.sub(r"[^\w\s\u0300-\u036f]+", " ", s, flags=re.UNICODE)

        # Collapse whitespace
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def ascii_fallback(self, normalized_primary: str) -> str:
        if not normalized_primary:
            return ""

        s = normalized_primary

        # Apply common expansions first (primary is already casefolded)
        for k, v in self._EXPAND.items():
            s = s.replace(k.casefold(), v)

        # Strip diacritics: decompose -> remove combining marks -> recompose
        s = unicodedata.normalize("NFD", s)
        s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
        s = unicodedata.normalize("NFC", s)

        # Conservative cleanup
        s = re.sub(r"[^\w\s]+", " ", s, flags=re.UNICODE)
        s = re.sub(r"\s+", " ", s).strip()
        return s