import json
import tempfile
from pathlib import Path

from app.parsers import parse_document
from app.retrieval import answer_question
from app.store import DocumentStore

ROOT = Path(__file__).parent.parent


def main() -> None:
    content = (ROOT / "sample_documents" / "manual_soporte.txt").read_bytes()
    pages, mime_type = parse_document("manual_soporte.txt", content)
    with tempfile.TemporaryDirectory() as directory:
        store = DocumentStore(Path(directory))
        store.add_document(
            filename="manual_soporte.txt",
            title="Manual de soporte",
            content=content,
            pages=pages,
            mime_type=mime_type,
        )
        cases = json.loads((ROOT / "evaluation" / "questions.json").read_text(encoding="utf-8"))
        results = []
        for case in cases:
            response = answer_question(case["question"], store.chunks, top_k=5, min_confidence=0.12)
            terms_ok = all(
                term.lower() in response.answer.lower() for term in case["expected_terms"]
            )
            passed = response.grounded == case["should_answer"] and terms_ok
            results.append(
                {
                    "question": case["question"],
                    "passed": passed,
                    "grounded": response.grounded,
                    "citations": len(response.citations),
                    "score": response.score,
                }
            )
        passed_count = sum(result["passed"] for result in results)
        report = {
            "passed": passed_count,
            "total": len(results),
            "accuracy_percent": round(passed_count / len(results) * 100, 2),
            "results": results,
        }
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
