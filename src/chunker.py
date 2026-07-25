import os
import re
import pdfplumber
from langdetect import detect, DetectorFactory, LangDetectException

DetectorFactory.seed = 0

RE_ARTICLE = re.compile(
    r"^(?:ARTICLE|Article)\s+"
    r"(?P<no>[IVXLCDM]+|\d+)"
    r"[\s.\-\u2013:]*"
    r"(?P<title>.+)?$"
)

RE_CLAUSE = re.compile(
    r"^\s*"
    r"(?P<no>\d+|[A-Za-z])"
    r"\s*(?:\.|\))"
    r"(?:\s+(?P<text>.+))?"
)

RE_SUBCLAUSE = re.compile(
    r"^\s{0,8}"
    r"\((?P<no>[a-z]|[ivx]+|\d+)\)"
    r"\s+(?P<text>.+)",
    re.I
)

RE_SUBCLAUSE_MID = re.compile(
    r"\((?P<no>[a-z]|[ivx]+|\d+)\)"
    r"\s+(?P<text>.+)",
    re.I
)

RE_PART = re.compile(r"^(?:PART|Part)\s+\w+")
RE_ANNEX = re.compile(r"^(?:ANNEX|Annexure|SCHEDULE)\b")
RE_DOC_CODE = re.compile(r"\b(?:A/RES/|CCPR/|CAT/)\S+")
RE_PAGE_NUM = re.compile(r"\n\s*\d+\s*\n")


class NonEnglishDocumentError(Exception):
    pass


class Chunk:
    def __init__(self, chunk_type, doc_name, path_parts, **kwargs):
        self.chunk_type = chunk_type
        self.doc_name = doc_name
        self.path = "/".join(str(p) for p in path_parts)
        for k, v in kwargs.items():
            setattr(self, k, v)

    def to_dict(self):
        result = {}
        for attr, value in self.__dict__.items():
            if value is not None:
                result[attr] = value
        return result


def load_pdf_text(filepath):
    full_text = []
    try:
        with pdfplumber.open(filepath) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    full_text.append(page_text)
    except Exception as exc:
        raise RuntimeError(f"Failed to load PDF '{filepath}': {exc}") from exc

    return "\n".join(full_text)


def fix_hyphenated_breaks(text):
    return re.sub(r"-\s*\n\s*", "", text)


def remove_noise(text):
    text = RE_PAGE_NUM.sub("\n", text)
    text = RE_DOC_CODE.sub("", text)
    return text


def normalize_spaces(text):
    return re.sub(r"[ \t]+", " ", text)


def normalize_midline_breaks(text):
    text = re.sub(
        r"(?<=[a-z][.;])\s+(?=\d+\.\s+[A-Z])",
        "\n",
        text,
    )
    text = re.sub(
        r"(?<=[.;])\s+(?=[A-Z]\.\s+[A-Z])",
        "\n",
        text,
    )
    return text


def clean_text(text):
    text = fix_hyphenated_breaks(text)
    text = remove_noise(text)
    text = normalize_spaces(text)
    text = normalize_midline_breaks(text)
    return text.strip()


def detect_language(text, doc_name):
    sample = text[:2000]
    clean_sample = re.sub(r"[\d\W_]+", " ", sample).strip()

    if not clean_sample:
        raise NonEnglishDocumentError(
            f"'{doc_name}' is not in English, please insert an English document"
        )

    try:
        lang = detect(clean_sample)
    except LangDetectException as exc:
        raise NonEnglishDocumentError(
            f"'{doc_name}' is not in English, please insert an English document"
        ) from exc

    if lang != "en":
        raise NonEnglishDocumentError(
            f"'{doc_name}' is not in English, please insert an English document"
        )


def normalize_article_no(raw):
    roman_values = {
        "I": 1, "V": 5, "X": 10, "L": 50,
        "C": 100, "D": 500, "M": 1000,
    }

    if raw.isdigit():
        return raw, raw

    try:
        total = 0
        prev = 0
        for ch in reversed(raw.upper()):
            curr = roman_values[ch]
            if curr < prev:
                total -= curr
            else:
                total += curr
            prev = curr
        return str(total), raw
    except (KeyError, TypeError):
        return raw, raw


def normalize_clause_no(raw):
    raw_stripped = raw.strip()
    if raw_stripped.isalpha() and len(raw_stripped) == 1:
        return str(ord(raw_stripped.upper()) - ord("A") + 1)
    return str(int(raw_stripped))


def _alpha_to_int(s):
    return ord(s) - ord("a") + 1


_ROMAN_TO_INT = {
    "i": 1, "ii": 2, "iii": 3, "iv": 4,
    "v": 5, "vi": 6, "vii": 7, "viii": 8,
    "ix": 9, "x": 10,
}


def normalize_subclause_no(raw):
    raw_lower = raw.lower().strip()

    if raw_lower in _ROMAN_TO_INT:
        return str(_ROMAN_TO_INT[raw_lower]), raw

    if raw_lower.isdigit():
        return raw_lower, raw

    if raw_lower.isalpha() and len(raw_lower) == 1:
        return str(_alpha_to_int(raw_lower)), raw

    return raw_lower, raw


def parse_document(text, doc_name):
    lines = text.split("\n")
    chunks = []

    current_article = None
    current_clause = None
    current_subclause = None
    text_buffer = []
    blank_line_count = 0
    parsing_started = False
    parsing_done = False

    doc_name_upper = doc_name.replace(".pdf", "").upper()

    def flush_buffer(target):
        if text_buffer:
            joined = " ".join(t.strip() for t in text_buffer if t.strip())
            if joined:
                target["text"] = joined
            text_buffer.clear()

    def _flush_article_to_chunks():
        nonlocal current_article, current_clause, current_subclause
        if current_article is None or current_article.get("_appended"):
            return
        finalise_clause()
        flush_buffer(current_article)
        has_text = current_article.get("text") is not None
        current_article["is_structural"] = not has_text
        path_parts = [doc_name_upper, current_article["no"]]
        chunk = Chunk(
            "article", doc_name, path_parts,
            article_no=current_article["no"],
            article_no_display=current_article["no_display"],
            article_title=current_article.get("title"),
            is_structural=current_article["is_structural"],
            text=current_article.get("text"),
            depth=0,
        )
        chunks.append(chunk)
        current_article["_appended"] = True

    def finalise_article():
        nonlocal current_article, current_clause, current_subclause
        if current_article is None:
            return
        if current_clause is not None:
            finalise_clause()
        if current_subclause is not None:
            finalise_subclause()
        _flush_article_to_chunks()
        current_article = None

    def _flush_clause_to_chunks():
        nonlocal current_clause, current_subclause
        if current_clause is None or current_clause.get("_appended"):
            return
        finalise_subclause()
        flush_buffer(current_clause)
        has_text = current_clause.get("text") is not None
        current_clause["is_structural"] = not has_text
        path_parts = [
            doc_name_upper,
            current_clause["parent_article"],
            current_clause["no"],
        ]
        chunk = Chunk(
            "clause", doc_name, path_parts,
            clause_no=current_clause["no"],
            parent_article_no=current_clause["parent_article"],
            is_structural=current_clause["is_structural"],
            text=current_clause.get("text"),
            depth=1,
        )
        chunks.append(chunk)
        current_clause["_appended"] = True

    def finalise_clause():
        nonlocal current_clause, current_subclause
        if current_clause is None:
            return
        _flush_clause_to_chunks()
        current_clause = None

    def finalise_subclause():
        nonlocal current_subclause
        if current_subclause is None:
            return

        flush_buffer(current_subclause)

        path_parts = [
            doc_name_upper,
            current_subclause["parent_article"],
        ]
        kwargs = {
            "subclause_no": current_subclause["no"],
            "subclause_no_display": current_subclause["no_display"],
            "parent_article_no": current_subclause["parent_article"],
            "text": current_subclause.get("text"),
        }
        if current_subclause.get("parent_clause") is not None:
            kwargs["parent_clause_no"] = current_subclause["parent_clause"]
            path_parts.append(current_subclause["parent_clause"])
        path_parts.append(current_subclause["no"])

        kwargs["depth"] = 2

        chunk = Chunk(
            "subclause", doc_name, path_parts,
            **kwargs
        )
        chunks.append(chunk)
        current_subclause = None

    for line in lines:
        stripped = line.strip()

        if RE_ANNEX.search(stripped):
            parsing_done = True
            break

        if not parsing_started:
            article_match = RE_ARTICLE.match(stripped)
            if article_match and article_match.group("no") in ("1", "I"):
                parsing_started = True
            else:
                continue

        if parsing_done:
            break

        if RE_PART.match(stripped):
            continue

        if not stripped:
            blank_line_count += 1
            if blank_line_count >= 2 and current_subclause is not None:
                finalise_subclause()
            continue
        else:
            blank_line_count = 0

        article_match = RE_ARTICLE.match(stripped)
        if article_match:
            finalise_article()
            raw_no = article_match.group("no")
            stored_no, display_no = normalize_article_no(raw_no)
            current_article = {
                "no": stored_no,
                "no_display": display_no,
                "title": article_match.group("title"),
            }
            continue

        clause_match = RE_CLAUSE.match(stripped)
        if clause_match:
            if current_clause is not None:
                subclause_match = RE_SUBCLAUSE.match(stripped)
                if subclause_match:
                    if current_article is not None:
                        _flush_article_to_chunks()
                    if current_clause is not None and not current_clause.get("_leadin_flushed"):
                        flush_buffer(current_clause)
                        current_clause["_leadin_flushed"] = True
                    if current_clause is not None:
                        _flush_clause_to_chunks()
                    finalise_subclause()
                    raw_no = subclause_match.group("no")
                    stored_no, display_no = normalize_subclause_no(raw_no)
                    current_subclause = {
                        "no": stored_no,
                        "no_display": display_no,
                        "parent_article": current_article["no"] if current_article else None,
                        "parent_clause": current_clause["no"],
                    }
                    continue

            raw_no = clause_match.group("no")
            clause_text_body = clause_match.group("text")

            inner_clause = RE_CLAUSE.match(clause_text_body) if clause_text_body else None
            if inner_clause:
                text_buffer.append(raw_no + ".")
                raw_no = inner_clause.group("no")
                clause_text_body = inner_clause.group("text")

            if current_article is not None:
                _flush_article_to_chunks()
            if current_subclause is not None:
                finalise_subclause()
            finalise_clause()
            stored_no = normalize_clause_no(raw_no)
            current_clause = {
                "no": stored_no,
                "parent_article": current_article["no"] if current_article else None,
            }
            embedded_sc = RE_SUBCLAUSE.match(clause_text_body) if clause_text_body else None
            if not embedded_sc and clause_text_body:
                embedded_sc = RE_SUBCLAUSE_MID.search(clause_text_body)
            if embedded_sc:
                prefix = clause_text_body[:embedded_sc.start()]
                if prefix.strip():
                    text_buffer.append(prefix.strip())
                current_clause["_leadin_flushed"] = True
                _flush_clause_to_chunks()
                raw_sc_no = embedded_sc.group("no")
                stored_sc_no, display_sc_no = normalize_subclause_no(raw_sc_no)
                current_subclause = {
                    "no": stored_sc_no,
                    "no_display": display_sc_no,
                    "parent_article": current_article["no"] if current_article else None,
                    "parent_clause": stored_no,
                }
                if embedded_sc.group("text"):
                    text_buffer.append(embedded_sc.group("text"))
            else:
                current_clause["text_body"] = clause_text_body
                if clause_text_body:
                    text_buffer.append(clause_text_body)
            continue

        subclause_match = RE_SUBCLAUSE.match(stripped)
        if subclause_match:
            if current_subclause is not None:
                finalise_subclause()

            if current_article is not None:
                _flush_article_to_chunks()
            if current_clause is not None and not current_clause.get("_leadin_flushed"):
                flush_buffer(current_clause)
                current_clause["_leadin_flushed"] = True
            if current_clause is not None:
                _flush_clause_to_chunks()
            finalise_subclause()
            raw_no = subclause_match.group("no")
            stored_no, display_no = normalize_subclause_no(raw_no)
            current_subclause = {
                "no": stored_no,
                "no_display": display_no,
                "parent_article": current_article["no"] if current_article else None,
                "parent_clause": current_clause["no"] if current_clause else None,
            }
            if subclause_match.group("text"):
                text_buffer.append(subclause_match.group("text"))
            continue

        if current_subclause is not None:
            text_buffer.append(stripped)
        elif current_clause is not None:
            mid_sc = RE_SUBCLAUSE_MID.search(stripped)
            if mid_sc and current_article is not None:
                before = stripped[:mid_sc.start()]
                if before.strip():
                    text_buffer.append(before.strip())
                if not current_clause.get("_leadin_flushed"):
                    flush_buffer(current_clause)
                    current_clause["_leadin_flushed"] = True
                if current_article is not None:
                    _flush_article_to_chunks()
                if current_clause is not None:
                    _flush_clause_to_chunks()
                finalise_subclause()
                raw_no = mid_sc.group("no")
                stored_no, display_no = normalize_subclause_no(raw_no)
                current_subclause = {
                    "no": stored_no,
                    "no_display": display_no,
                    "parent_article": current_article["no"] if current_article else None,
                    "parent_clause": current_clause["no"] if current_clause else None,
                }
                if mid_sc.group("text"):
                    text_buffer.append(mid_sc.group("text"))
            else:
                text_buffer.append(stripped)
        elif current_article is not None:
            text_buffer.append(stripped)

    finalise_subclause()
    finalise_clause()
    finalise_article()

    return [c.to_dict() for c in chunks]


def chunk_document(filepath):
    doc_name = os.path.basename(filepath)

    raw_text = load_pdf_text(filepath)

    cleaned = clean_text(raw_text)

    if not cleaned:
        raise RuntimeError(f"No text could be extracted from '{doc_name}'")

    detect_language(cleaned, doc_name)

    chunks = parse_document(cleaned, doc_name)
    if not chunks:
        raise ValueError(
            f"'{doc_name}' contains no treaty structure (no articles found)"
        )
    return chunks
