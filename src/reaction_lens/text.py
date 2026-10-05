"""Versioned sentence segmentation with original character offsets."""
import re

POLICY = "v1: title as one segment; conservative punctuation/newline abstract spans; original character offsets; no text truncation; encode each segment as (text, empty abstract) with frozen proximity SPECTER2"
ABBREVIATIONS = {"e.g.", "i.e.", "et al.", "fig.", "figs.", "dr.", "vs.", "mr.", "mrs.", "prof.", "no.", "approx."}

def segments(row):
    """Heuristic boundaries, not biological evidence annotations; preserve all text."""
    result = []
    for field in ("title", "abstract"):
        text = row.get(field, "")
        cuts = [0]
        if field == "abstract":
            for match in re.finditer(r"(?<=[.!?])\s+|\n+", text):
                prefix = text[:match.start()].lower()
                if "\n" not in match.group() and (any(prefix.endswith(a) for a in ABBREVIATIONS)
                        or re.search(r"(?:^|\s)[a-z]\.$", prefix)):
                    continue
                cuts.append(match.end())
        cuts.append(len(text))
        for start, end in zip(cuts, cuts[1:]):
            # Include inter-sentence whitespace in offsets; encoder text is stripped.
            if text[start:end].strip():
                result.append({"field": field, "start": start, "end": end})
    if not result:
        raise ValueError("Empty publication")
    return result

def segment_texts(row, spans):
    return [row[s["field"]][s["start"]:s["end"]].strip() for s in spans]
