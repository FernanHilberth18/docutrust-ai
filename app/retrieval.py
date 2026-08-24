import math
import re
import time
from collections import Counter

from sklearn.feature_extraction.text import HashingVectorizer

from app.schemas import Citation, QueryResponse

TOKEN_PATTERN = re.compile(r"[a-záéíóúñü0-9]{2,}", re.IGNORECASE)
SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+")
STOPWORDS = {
    "de",
    "del",
    "el",
    "la",
    "los",
    "las",
    "un",
    "una",
    "que",
    "qué",
    "como",
    "cómo",
    "cuál",
    "cual",
    "es",
    "son",
    "en",
    "por",
    "para",
    "con",
    "se",
    "y",
    "o",
    "the",
    "is",
    "are",
    "what",
    "how",
}
QUERY_EXPANSIONS = {
    "horario": ["atiende", "atención", "horas"],
    "soporte": ["mesa", "ayuda", "servicio"],
    "vacaciones": ["días", "permiso"],
    "vpn": ["acceso", "remoto", "conectarse"],
    "tarda": ["tiempo", "horas", "plazo"],
    "respuesta": ["atiende", "tiempo", "objetivo"],
}


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


def expand_question(question: str) -> str:
    tokens = tokenize(question)
    additions = [term for token in tokens for term in QUERY_EXPANSIONS.get(token, [])]
    return " ".join([question, *additions])


def split_sentences(text: str) -> list[str]:
    protected = text.replace("a.m.", "a·m·").replace("p.m.", "p·m·")
    return [
        sentence.replace("a·m·", "a.m.").replace("p·m·", "p.m.")
        for sentence in SENTENCE_PATTERN.split(protected)
    ]


def bm25_scores(question: str, chunks: list[dict]) -> list[float]:
    tokenized = [tokenize(chunk["text"]) for chunk in chunks]
    query_terms = set(tokenize(question)) - STOPWORDS
    if not query_terms or not tokenized:
        return [0.0] * len(chunks)
    average_length = sum(len(tokens) for tokens in tokenized) / len(tokenized)
    frequencies = Counter(term for tokens in tokenized for term in set(tokens))
    scores = []
    for tokens in tokenized:
        counts = Counter(tokens)
        score = 0.0
        for term in query_terms:
            document_frequency = frequencies[term]
            if not document_frequency:
                continue
            inverse_frequency = math.log(
                1 + (len(tokenized) - document_frequency + 0.5) / (document_frequency + 0.5)
            )
            frequency = counts[term]
            denominator = frequency + 1.5 * (1 - 0.75 + 0.75 * len(tokens) / max(average_length, 1))
            score += inverse_frequency * (frequency * 2.5 / denominator)
        scores.append(score)
    return scores


def hybrid_search(
    question: str,
    chunks: list[dict],
    *,
    top_k: int,
    document_ids: list[str] | None = None,
) -> list[dict]:
    candidates = [
        chunk for chunk in chunks if not document_ids or chunk["document_id"] in document_ids
    ]
    if not candidates:
        return []
    texts = [chunk["text"] for chunk in candidates]
    expanded_question = expand_question(question)
    lexical = bm25_scores(expanded_question, candidates)
    max_lexical = max(lexical) or 1.0
    vectorizer = HashingVectorizer(
        n_features=4096,
        alternate_sign=False,
        ngram_range=(1, 2),
        norm="l2",
        lowercase=True,
    )
    matrix = vectorizer.transform([*texts, expanded_question])
    vector_scores = (matrix[:-1] @ matrix[-1].T).toarray().ravel()

    ranked = []
    for chunk, lexical_score, vector_score in zip(candidates, lexical, vector_scores, strict=True):
        combined = 0.65 * (lexical_score / max_lexical) + 0.35 * float(vector_score)
        ranked.append({**chunk, "score": round(combined, 6)})
    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked[:top_k]


def best_sentences(question: str, evidence: list[dict], limit: int = 2) -> list[tuple[str, dict]]:
    question_terms = set(tokenize(question)) - STOPWORDS
    candidates: list[tuple[float, str, dict]] = []
    for rank, chunk in enumerate(evidence):
        for sentence in split_sentences(chunk["text"]):
            sentence = sentence.strip()
            if len(sentence) < 25:
                continue
            sentence_terms = set(tokenize(sentence))
            overlap = len(question_terms.intersection(sentence_terms))
            score = overlap * 2 + chunk["score"] - rank * 0.05
            candidates.append((score, sentence, chunk))
    candidates.sort(key=lambda item: item[0], reverse=True)
    selected: list[tuple[str, dict]] = []
    seen = set()
    for _, sentence, chunk in candidates:
        normalized = sentence.lower()
        if normalized in seen:
            continue
        selected.append((sentence, chunk))
        seen.add(normalized)
        if len(selected) == limit:
            break
    return selected


def answer_question(
    question: str,
    chunks: list[dict],
    *,
    top_k: int,
    min_confidence: float,
    document_ids: list[str] | None = None,
) -> QueryResponse:
    started_at = time.perf_counter()
    ranked = hybrid_search(question, chunks, top_k=top_k, document_ids=document_ids)
    warnings: list[str] = []
    flagged = [item for item in ranked if item["untrusted"]]
    if flagged:
        warnings.append(
            "Se omitió contenido con posibles instrucciones incrustadas en el documento."
        )
    safe_evidence = [item for item in ranked if not item["untrusted"]]
    top_score = safe_evidence[0]["score"] if safe_evidence else 0.0
    latency = round((time.perf_counter() - started_at) * 1000, 2)
    if top_score < min_confidence:
        return QueryResponse(
            answer=(
                "No encontré evidencia suficiente en los documentos para responder con confianza. "
                "Intenta reformular la pregunta o cargar una fuente relevante."
            ),
            grounded=False,
            confidence="insufficient",
            score=round(top_score, 4),
            citations=[],
            warnings=warnings,
            latency_ms=latency,
        )

    selected = best_sentences(question, safe_evidence)
    if not selected:
        selected = [(safe_evidence[0]["text"][:500], safe_evidence[0])]
    answer_parts = [
        f"{sentence} [{chunk['document_title']}, p. {chunk['page']}]"
        for sentence, chunk in selected
    ]
    citation_chunks = []
    seen_chunks = set()
    for _, chunk in selected:
        if chunk["id"] not in seen_chunks:
            citation_chunks.append(chunk)
            seen_chunks.add(chunk["id"])
    confidence = "high" if top_score >= 0.55 else "medium" if top_score >= 0.25 else "low"
    citations = [
        Citation(
            document_id=chunk["document_id"],
            document_title=chunk["document_title"],
            page=chunk["page"],
            chunk_id=chunk["id"],
            excerpt=chunk["text"][:500],
            score=round(chunk["score"], 4),
            untrusted_content=chunk["untrusted"],
        )
        for chunk in citation_chunks
    ]
    latency = round((time.perf_counter() - started_at) * 1000, 2)
    return QueryResponse(
        answer=" ".join(answer_parts),
        grounded=True,
        confidence=confidence,
        score=round(top_score, 4),
        citations=citations,
        warnings=warnings,
        latency_ms=latency,
    )
