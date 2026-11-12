"""Ricerca sul web senza dipendenze esterne.

Cerca su DuckDuckGo (versione HTML), scarica le prime pagine e ne estrae il
testo. Il risultato viene passato al modello nel blocco <|context|>.
"""

from __future__ import annotations

import html
import re
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from html.parser import HTMLParser

USER_AGENT = "Mozilla/5.0 (compatible; Milo/1.0)"
_SKIP_TAGS = {"script", "style", "noscript", "nav", "footer", "header", "aside", "form", "svg", "template"}
_BLOCK_TAGS = {"p", "br", "div", "li", "h1", "h2", "h3", "h4", "tr", "section", "article"}


@dataclass
class Result:
    title: str
    url: str
    snippet: str
    text: str = ""


def _get(url: str, timeout: float, limit: int = 1_500_000) -> str:
    if urllib.parse.urlparse(url).scheme not in ("http", "https"):
        raise ValueError("URL non supportato")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept-Language": "it,en;q=0.8"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        if "html" not in resp.headers.get("Content-Type", "text/html"):
            return ""
        charset = resp.headers.get_content_charset() or "utf-8"
        return resp.read(limit).decode(charset, errors="replace")


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in _SKIP_TAGS:
            self.skip += 1
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in _SKIP_TAGS and self.skip:
            self.skip -= 1
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)


def html_to_text(page: str) -> str:
    parser = _TextExtractor()
    try:
        parser.feed(page)
    except Exception:
        pass
    text = "".join(parser.parts)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    lines = [ln.strip() for ln in text.split("\n")]
    # Le righe cortissime sono quasi sempre menu e pulsanti.
    return "\n".join(ln for ln in lines if len(ln) > 40)


def _strip_tags(s: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", s)).strip()


def search(query: str, max_results: int = 3, timeout: float = 8.0) -> list[Result]:
    """Risultati (titolo, url, riassunto) da DuckDuckGo."""
    page = _get("https://html.duckduckgo.com/html/?" + urllib.parse.urlencode({"q": query}), timeout)
    links = re.findall(r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', page, re.S)
    snippets = re.findall(r'class="result__snippet"[^>]*>(.*?)</a>', page, re.S)

    results: list[Result] = []
    for i, (href, title) in enumerate(links):
        href = html.unescape(href)
        # I link passano da un redirect: l'indirizzo vero è nel parametro "uddg".
        target = urllib.parse.parse_qs(urllib.parse.urlparse(href).query).get("uddg", [href])[0]
        if target.startswith("//"):
            target = "https:" + target
        if "duckduckgo.com/y.js" in target:  # annunci
            continue
        results.append(Result(_strip_tags(title), target, _strip_tags(snippets[i]) if i < len(snippets) else ""))
        if len(results) >= max_results:
            break
    return results


def fetch_text(url: str, timeout: float = 8.0) -> str:
    try:
        return html_to_text(_get(url, timeout))
    except Exception:
        return ""


def search_and_read(query: str, max_results: int = 3, chars_per_page: int = 1200) -> list[Result]:
    """Cerca e scarica in parallelo il testo delle pagine trovate."""
    results = search(query, max_results)
    with ThreadPoolExecutor(max_workers=max(1, len(results))) as pool:
        for r, text in zip(results, pool.map(fetch_text, [r.url for r in results])):
            r.text = text[:chars_per_page]
    return results


def format_context(results: list[Result]) -> str:
    """Testo da mettere nel blocco <|context|>. Stesso formato dei dati di addestramento."""
    blocks = []
    for i, r in enumerate(results, 1):
        body = r.text or r.snippet
        blocks.append(f"[{i}] {r.title}\n{r.url}\n{body}".strip())
    return "\n\n".join(blocks)
