"""
Tool de busca de artigos científicos no PubMed, usando a API E-utilities do NCBI
diretamente (gratuita, sem necessidade de API key para uso moderado).
Muito mais simples que rodar um servidor MCP local: são duas chamadas HTTP.

Documentação oficial: https://www.ncbi.nlm.nih.gov/books/NBK25501/
"""

import logging
import time
import requests
from typing import Optional
from langchain_core.tools import tool

logger = logging.getLogger("pubmed_tool")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s pubmed_tool: %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# Opcional: registrar um e-mail e (se tiver) uma API key aumenta o rate limit.
# Sem API key: ~3 requisições/segundo. Com key gratuita: ~10/segundo.
# Para obter uma key gratuita: https://www.ncbi.nlm.nih.gov/account/settings/
NCBI_API_KEY = None  # ex: "1234567890abcdef..."
NCBI_EMAIL = None    # ex: "seuemail@exemplo.com" (recomendado por boas práticas da API)


def _base_params():
    params = {}
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY
    if NCBI_EMAIL:
        params["email"] = NCBI_EMAIL
    return params


@tool
def search_pubmed(query: str, max_results: int = 8) -> str:
    """Busca artigos científicos no PubMed (base biomédica do NCBI) sobre um tema.
    Use para perguntas do tipo 'existe evidência científica sobre X para TEA' ou
    'o que a literatura diz sobre Y'. Retorna título, autores, revista, ano e resumo
    de cada artigo encontrado."""

    call_start = time.monotonic()
    logger.info(f"CHAMADA INICIADA | query='{query}' | max_results={max_results}")

    # 1. Busca os IDs dos artigos (esearch)
    search_params = {
        **_base_params(),
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json",
        "sort": "relevance",
    }

    t0 = time.monotonic()
    try:
        search_resp = requests.get(f"{EUTILS_BASE}/esearch.fcgi", params=search_params, timeout=15)
        search_resp.raise_for_status()
    except requests.exceptions.Timeout:
        logger.error(f"esearch TIMEOUT após {time.monotonic() - t0:.2f}s | query='{query}'")
        return "Erro: a busca no PubMed excedeu o tempo limite (esearch)."
    except requests.exceptions.RequestException as e:
        logger.error(f"esearch FALHOU após {time.monotonic() - t0:.2f}s | erro={e}")
        return f"Erro ao consultar o PubMed (esearch): {e}"

    esearch_elapsed = time.monotonic() - t0
    logger.info(f"esearch OK | {esearch_elapsed:.2f}s | status={search_resp.status_code}")

    esearch_json = search_resp.json()
    id_list = esearch_json.get("esearchresult", {}).get("idlist", [])
    total_count = esearch_json.get("esearchresult", {}).get("count", "?")
    logger.info(f"esearch retornou {len(id_list)} ids (de {total_count} resultados totais no PubMed)")

    if not id_list:
        logger.warning(f"NENHUM RESULTADO para query='{query}' | tempo total={time.monotonic() - call_start:.2f}s")
        return "Nenhum artigo encontrado para essa busca."

    time.sleep(0.34)  # respeita rate limit (~3 req/s sem API key)

    # 2. Busca os detalhes (efetch, formato XML com abstract)
    fetch_params = {
        **_base_params(),
        "db": "pubmed",
        "id": ",".join(id_list),
        "rettype": "abstract",
        "retmode": "xml",
    }

    t1 = time.monotonic()
    try:
        fetch_resp = requests.get(f"{EUTILS_BASE}/efetch.fcgi", params=fetch_params, timeout=15)
        fetch_resp.raise_for_status()
    except requests.exceptions.Timeout:
        logger.error(f"efetch TIMEOUT após {time.monotonic() - t1:.2f}s | ids={id_list}")
        return "Erro: a busca no PubMed excedeu o tempo limite (efetch)."
    except requests.exceptions.RequestException as e:
        logger.error(f"efetch FALHOU após {time.monotonic() - t1:.2f}s | erro={e}")
        return f"Erro ao consultar o PubMed (efetch): {e}"

    efetch_elapsed = time.monotonic() - t1
    logger.info(f"efetch OK | {efetch_elapsed:.2f}s | tamanho_resposta={len(fetch_resp.text)} chars")

    t2 = time.monotonic()
    result = _parse_pubmed_xml(fetch_resp.text)
    parse_elapsed = time.monotonic() - t2

    total_elapsed = time.monotonic() - call_start
    logger.info(
        f"CHAMADA CONCLUÍDA | total={total_elapsed:.2f}s "
        f"(esearch={esearch_elapsed:.2f}s, efetch={efetch_elapsed:.2f}s, parse={parse_elapsed:.2f}s)"
    )

    if total_elapsed > 5:
        logger.warning(f"CHAMADA LENTA (>5s): considere revisar rede/rate-limit/max_results")

    return result


def _parse_pubmed_xml(xml_text: str) -> str:
    """Parser simples do XML retornado pelo efetch, extraindo os campos principais."""
    import xml.etree.ElementTree as ET

    root = ET.fromstring(xml_text)
    articles = []

    for article in root.findall(".//PubmedArticle"):
        title_el = article.find(".//ArticleTitle")
        title = "".join(title_el.itertext()).strip() if title_el is not None else "Sem título"

        abstract_parts = article.findall(".//AbstractText")
        abstract = " ".join("".join(p.itertext()).strip() for p in abstract_parts) or "Sem resumo disponível."

        journal_el = article.find(".//Journal/Title")
        journal = journal_el.text if journal_el is not None else "Revista não identificada"

        year_el = article.find(".//PubDate/Year")
        year = year_el.text if year_el is not None else "Ano não identificado"

        authors = []
        for author in article.findall(".//AuthorList/Author"):
            last_name = author.find("LastName")
            initials = author.find("Initials")
            if last_name is not None:
                name = last_name.text
                if initials is not None:
                    name += f" {initials.text}"
                authors.append(name)

        pmid_el = article.find(".//PMID")
        pmid = pmid_el.text if pmid_el is not None else None
        url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None

        articles.append({
            "title": title,
            "authors": authors[:5],  # limita para não poluir o contexto do agente
            "journal": journal,
            "year": year,
            "abstract": abstract[:1000],  # trunca resumos muito longos
            "url": url,
        })

    if not articles:
        logger.warning("PARSE: nenhum artigo pôde ser extraído do XML retornado (resposta pode ter mudado de formato)")
        return "Nenhum artigo pôde ser processado a partir da resposta do PubMed."

    logger.info(f"PARSE OK: {len(articles)} artigos extraídos do XML")

    import json
    return json.dumps(articles, ensure_ascii=False, indent=2)