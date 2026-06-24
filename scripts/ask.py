import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.api.dependencies import get_chat_service


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask a question against the RAG knowledge base.")
    parser.add_argument("question", help="Question to ask")
    parser.add_argument("--top-k", type=int, default=None, help="Number of chunks to retrieve")
    args = parser.parse_args()

    service = get_chat_service()
    result = service.ask(question=args.question, top_k=args.top_k)
    print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
