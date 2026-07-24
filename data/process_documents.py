"""Process a folder of documents and extract entities and relations using LangChain.

This script walks through all files in a given folder, loads text content,
splits the text into chunks, and uses a language model to extract named
entities and relations between entities. The results are written to a JSON file.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List


# LangChain imports (ensure the package is installed in your environment)
try:
    from langchain.chains import LLMChain
    from langchain.prompts import PromptTemplate
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.chat_models import ChatOpenAI
except Exception:  # pragma: no cover
    raise SystemExit("LangChain is required. Install: pip install langchain[openai]")

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file


def _get_llm(model_name: str, temperature: float):
    # Try ChatOpenAI first, fall back to OpenAI if needed
    try:
        llm = ChatOpenAI(model_name=model_name, temperature=temperature)
    except Exception:
        from langchain.llms import OpenAI as _OpenAI
        llm = _OpenAI(model=model_name, temperature=temperature)  # type: ignore
    return llm


def _log(message: str) -> None:
    from datetime import datetime
    print(f"[ProcessDocs] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}")


def _build_chain(model_name: str, temperature: float) -> LLMChain:
    template = (
        "Você é um analista de NLP avançado. Dado o seguinte texto, "
        "extraia todas as entidades relevantes e as relações entre elas. "
        "Retorne apenas um JSON com as chaves 'entities' e 'relations'. "
        "Entidades: cada item tem 'text' e 'type'. Relações: cada item tem 'type', "
        "'subject' e 'object' (valor text do sujeito e do objeto). Texto:\n{text}"
    )
    prompt = PromptTemplate(template=template, input_variables=["text"])
    llm = _get_llm(model_name, temperature)
    return LLMChain(llm=llm, prompt=prompt, verbose=False)


def _read_text_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def _load_documents(folder: Path) -> List[Dict[str, str]]:
    texts: List[Dict[str, str]] = []
    allowed_exts = {
        ".txt", ".md", ".log", ".csv", ".json", ".xml", ".html", ".htm",
        ".mdx", ".rst", ".tex", ".yaml", ".yml",
    }
    _log(f"Scanning folder: {folder} for text files with extensions: {sorted(list(allowed_exts))}")
    for root, _, files in os.walk(folder.as_posix()):
        for fname in files:
            p = Path(root) / fname
            if p.suffix.lower() not in allowed_exts:
                continue
            # skip this script file itself to avoid reading code as content
            if p.name == Path(__file__).name:
                continue
            text = _read_text_file(p)
            if text.strip():
                texts.append({"path": str(p), "text": text})
    _log(f"Found {len(texts)} documents to analyze.")
    return texts


def _parse_json(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*?\}", text, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return None


def _normalize_data(chunk_result: Dict[str, Any]) -> Dict[str, Any]:
    entities = []
    relations = []
    if not chunk_result:
        return {"entities": entities, "relations": relations}
    raw_entities = chunk_result.get("entities", [])
    raw_relations = chunk_result.get("relations", [])
    for e in raw_entities:
        if not isinstance(e, dict):
            continue
        text = e.get("text") or e.get("name") or ""
        etype = e.get("type") or e.get("entity_type") or "UNKNOWN"
        if text:
            entities.append({"text": text, "type": etype})
    for r in raw_relations:
        if not isinstance(r, dict):
            continue
        rel_type = r.get("type") or r.get("relation_type") or "UNKNOWN"
        subject = r.get("subject") or r.get("subject_text") or ""
        obj = r.get("object") or r.get("object_text") or ""
        if subject or obj:
            relations.append({"type": rel_type, "subject": subject, "object": obj})
    return {"entities": entities, "relations": relations}


def process_document(doc: Dict[str, str], chain: LLMChain) -> Dict[str, Any]:
    text = doc.get("text", "")
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    chunks = splitter.split_text(text)
    all_entities: List[Dict[str, str]] = []
    all_relations: List[Dict[str, str]] = []
    seen = set()
    for chunk in chunks:
        raw = chain.run(text=chunk)
        parsed = _parse_json(raw)
        normalized = _normalize_data(parsed) if parsed else {"entities": [], "relations": []}
        for e in normalized.get("entities", []):
            key = (e.get("text"), e.get("type"))
            if key not in seen:
                seen.add(key)
                all_entities.append(e)
        all_relations.extend(normalized.get("relations", []))
    return {"entities": all_entities, "relations": all_relations}


def analyze_documents_list(
    documents: List[Dict[str, str]],
    model: str = "gpt-4",
    temperature: float = 0.0,
    chunk_size: int = 2000,
    chunk_overlap: int = 200,
) -> Dict[str, Any]:
    """Process a list of in-memory documents and return entities/relations."""
    _log(f"Initializing language model: {model} with temperature {temperature}")
    chain = _build_chain(model, temperature)
    results: Dict[str, Any] = {}
    total_entities = 0
    total_relations = 0
    for doc in documents:
        data = process_document(doc, chain)
        key = doc.get("path", "inline_document")
        results[key] = data
        _log(f"Document {key}: {len(data.get('entities', []))} entities, {len(data.get('relations', []))} relations")
        total_entities += len(data.get("entities", []))
        total_relations += len(data.get("relations", []))
    return {
        "files": results,
        "summary": {
            "files_processed": len(documents),
            "total_entities": total_entities,
            "total_relations": total_relations,
        },
    }


def analyze_folder(
    folder: str,
    model: str = "gpt-4",
    temperature: float = 0.0,
    chunk_size: int = 2000,
    chunk_overlap: int = 200,
) -> Dict[str, Any]:
    """Load documents from a folder and analyze them."""
    docs = _load_documents(Path(folder))
    return analyze_documents_list(docs, model, temperature, chunk_size, chunk_overlap)


def main(folder: str, output: str, model: str, temperature: float, chunk_size: int, chunk_overlap: int) -> None:
    folder_path = Path(folder)
    if not folder_path.exists() or not folder_path.is_dir():
        raise SystemExit(f"Folder inválido: {folder}")
    docs = _load_documents(folder_path)
    if not docs:
        raise SystemExit("Nenhum documento encontrado na pasta especificada.")
    _log(f"Starting processing folder: {folder}")
    chain = _build_chain(model, temperature)
    results: Dict[str, Any] = {}
    total_entities = 0
    total_relations = 0
    for doc in docs:
        data = process_document(doc, chain)
        results[doc["path"]] = data
        total_entities += len(data.get("entities", []))
        total_relations += len(data.get("relations", []))
        _log(f"Processed {doc['path']}: {len(data.get('entities', []))} entities, {len(data.get('relations', []))} relations")
    output_data = {"files": results, "summary": {"files_processed": len(docs), "total_entities": total_entities, "total_relations": total_relations}}
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    _log(f"Done. Output saved to {output}")


if __name__ == "__main__":
    folder_path = os.environ.get("PROCESS_DOCS_FOLDER", "c:\\00 - Source Codes\\finbrain-agent\\.env")
    output = os.environ.get("PROCESS_DOCS_OUTPUT", "data/ner_relations_output.json")
    model = os.environ.get("PROCESS_DOCS_MODEL", "gpt-5-nano")
    temperature = float(os.environ.get("PROCESS_DOCS_TEMPERATURE", "0.0"))
    chunk_size = int(os.environ.get("PROCESS_DOCS_CHUNK_SIZE", "2000"))
    chunk_overlap = int(os.environ.get("PROCESS_DOCS_CHUNK_OVERLAP", "200"))
    main(folder_path, output, model, temperature, chunk_size, chunk_overlap)