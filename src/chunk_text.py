def searchable_chunk_text(chunk):
    section = str(chunk.get("section") or "").strip()
    text = str(chunk.get("chunk_text") or "").strip()
    if section:
        return f"Section: {section}\n\n{text}"
    return text
