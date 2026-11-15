"""Бизнес-логика консультанта МТС Energytool: RAG по базе знаний, диалог, заявки.

Views работают только с функциями этого модуля. Поиск и генерация адаптированы
из solution.py (гибридный поиск BM25 + эмбеддинги, реранкер, все пространства имён базы).
"""

import json
import os
import re
import sys
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import PurePosixPath

import httpx
import numpy as np
from django.conf import settings
from rank_bm25 import BM25Okapi

from .models import ChatSession, Lead, Message


def env(*names, default=""):
    for name in names:
        value = os.getenv(name, "").strip()
        if value and value != name:
            return value
    return default


def number(name, default, lower, upper, kind=int):
    try:
        return max(lower, min(upper, kind(env(name))))
    except ValueError:
        return default


BASE_URL = env("LLM_BASE_URL", "BASE_URL", "OPENAI_BASE_URL").rstrip("/")
LLM_KEY = env("LLM_API_KEY", "OPENAI_API_KEY")
EMB_KEY = env("EMBEDDER_API_KEY") or LLM_KEY
RERANK_KEY = env("RERANKER_API_KEY") or LLM_KEY
LLM_MODEL = env("LLM_MODEL", "GEN_MODEL", "OPENAI_MODEL", default="qwen/qwen3.8-27b")
EMB_MODEL = env("EMB_MODEL", default="baai/bge-m3")
RERANKER_MODEL = env("RERANKER_MODEL", default="qwen/qwen3-reranker-8b")
TOP_K = number("TOP_K", 6, 1, 30)
RERANKER_TOP_N = number("RERANKER_TOP_N", 6, 1, 30)
MAX_TOKENS = number("MAX_TOKENS", 2048, 256, 8192)
MAX_WORKERS = number("MAX_WORKERS", 4, 1, 8)
TEMPERATURE = number("TEMPERATURE", 0.0, 0.0, 2.0, float)

MAX_CHUNK = 1400
PER_NAMESPACE = 3
HISTORY_LIMIT = 20
MAX_MESSAGE_LEN = 2000

FALLBACK = ("Сейчас не получается ответить по базе знаний. "
            "Предлагаю связаться со специалистом МТС Energytool: оставьте ФИО, телефон и название компании.")

SYSTEM_PROMPT = """Ты — ИИ-консультант сайта МТС Energytool (учёт и аналитика энергоресурсов, оптовый рынок электроэнергии, \
облачная платформа учёта, поиск коммерческих потерь, распознавание показаний по фото, солнечные станции). \
Клиенты пишут простыми словами, часто без терминов: отвечай понятно, без жаргона.

Правила:
1. Факты о продуктах, ценах, тарифах, сроках, характеристиках и порогах берёшь ТОЛЬКО из фрагментов базы знаний ниже. \
Собственные знания об энергетике не используй. Ситуацию клиента берёшь из диалога.
2. Если в фрагментах нет ответа — прямо скажи об этом, ничего не придумывай, предложи связаться со специалистом.
3. Консультируешь только по продуктам Energytool и связанным задачам клиента. Вопросы по технической документации, \
настройке приборов и систем учёта, юридические заключения о правомерности схем, договорные условия — вне твоей компетенции: \
скажи об этом и предложи специалиста.
4. Если запрос неясен или данных не хватает — дай то, что можно сказать по базе, и задай ОДИН уточняющий вопрос \
(тип объекта, задача, мощность, регион, приборы и т. п. — что важно именно здесь).
5. Подбирай подходящее решение под задачу клиента и объясняй простым языком, почему оно подходит. \
Если клиенту решение не подходит или проект может быть невыгоден — скажи честно.
6. Не гарантируй экономию и сроки, не называй точную стоимость, если её нет в базе. Не называй компании-клиентов, \
производителей оборудования и подрядчиков, не раскрывай внутренние алгоритмы и пороговые правила аналитики. \
МТС не является энергосбытовой компанией.
7. Если клиент готов к контакту или нужен расчёт по объекту — предложи связаться с менеджером и попроси недостающее из: \
ФИО, телефон, название компании. Если клиент не хочет оставлять телефон — не настаивай, помоги по существу. \
Не обещай точное время звонка и имя специалиста. Когда все три данных получены — поблагодари и скажи, \
что заявка передана менеджеру.
8. Диалог: отвечай на последнюю реплику пользователя с учётом истории; не повторяй уже заданные вопросы.
9. Отвечай по-русски, кратко: как правило до 150 слов, без таблиц и длинных списков, без ссылок на «фрагменты» и номера документов.
10. Служебный блок. Если клиент в этом диалоге называл ФИО, телефон или компанию, в самом конце ответа добавь блок \
<lead>{"full_name": "...", "phone": "...", "company": "..."}</lead> со ВСЕМИ известными за диалог значениями \
(неизвестные — пустая строка). Клиент этот блок не видит. Если ничего из этого не называлось — блок не добавляй."""

SUMMARY_PROMPT = """Составь краткое резюме диалога клиента с консультантом МТС Energytool для менеджера: \
задача клиента, что известно об объекте и параметрах, какие решения обсуждались, какие вопросы остались открытыми. \
Только факты из диалога, без выдумок. Не более 120 слов, простым текстом без разметки."""


# ---------------------------------------------------------------- HTTP

CLIENT = httpx.Client(timeout=150)


def post(endpoint, key, body, timeout=120, attempts=4):
    if not BASE_URL or not key:
        raise ValueError("Не заданы адрес API или ключ")
    for attempt in range(attempts):
        try:
            response = CLIENT.post(BASE_URL + endpoint, json=body, timeout=timeout,
                                   headers={"Authorization": "Bearer " + key})
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            code = exc.response.status_code
            if code not in (408, 429) and code < 500 or attempt == attempts - 1:
                raise
        except httpx.TransportError:
            if attempt == attempts - 1:
                raise
        time.sleep(1.5 * (attempt + 1))


# ---------------------------------------------------------------- база знаний

WORD = re.compile(r"[0-9a-zа-я]+")


def tokens(text):
    words = WORD.findall(text.lower().replace("ё", "е"))
    return [w[:5] if len(w) > 5 else w for w in words]


def namespace_of(name):
    """Пространство имён документа — папка, в которой он лежит (00-reference, 10-energy, 20-products)."""
    parts = PurePosixPath(name).parts
    return parts[-2] if len(parts) > 1 else "root"


def split_text(text):
    """Делит документ на фрагменты по заголовкам Markdown; длинные секции — по абзацам."""
    sections, heading, buf, title = [], "", [], ""
    for line in text.splitlines():
        match = re.match(r"^(#{1,4})\s+(.*)", line)
        if match:
            if "".join(buf).strip():
                sections.append((heading, "\n".join(buf).strip()))
            buf = []
            if len(match.group(1)) == 1:
                title = heading = match.group(2).strip()
            else:
                heading = (title + " > " if title else "") + match.group(2).strip()
        else:
            buf.append(line)
    if "".join(buf).strip():
        sections.append((heading, "\n".join(buf).strip()))

    pieces = []
    for heading, body in sections:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
        current = ""
        for paragraph in paragraphs:
            while len(paragraph) > MAX_CHUNK:
                cut = paragraph.rfind("\n", 0, MAX_CHUNK)
                cut = cut if cut > MAX_CHUNK // 2 else MAX_CHUNK
                if current:
                    pieces.append((heading, current))
                    current = ""
                pieces.append((heading, paragraph[:cut].strip()))
                paragraph = paragraph[cut:].strip()
            if current and len(current) + len(paragraph) + 2 > MAX_CHUNK:
                pieces.append((heading, current))
                current = ""
            current = (current + "\n\n" + paragraph).strip()
        if current:
            pieces.append((heading, current))
    return pieces


def load_chunks(path):
    chunks = []
    with zipfile.ZipFile(path) as archive:
        for name in sorted(archive.namelist()):
            if name.startswith("__MACOSX/") or PurePosixPath(name).name.startswith("."):
                continue
            if PurePosixPath(name).suffix.lower() not in (".md", ".txt"):
                continue
            text = archive.read(name).decode("utf-8-sig", errors="replace")
            if not text.strip():
                continue
            ns = namespace_of(name)
            for heading, body in split_text(text):
                label = f"{ns}/{PurePosixPath(name).name}" + (f" — {heading}" if heading else "")
                chunks.append({"ns": ns, "label": label, "text": body, "full": label + "\n" + body})
    if not chunks:
        raise ValueError("База знаний пуста")
    return chunks


def embed(texts):
    result = post("/embeddings", EMB_KEY, {"model": EMB_MODEL, "input": texts})
    vectors = [None] * len(texts)
    for i, item in enumerate(result["data"]):
        v = np.asarray(item["embedding"], dtype=float)
        vectors[int(item.get("index", i))] = v / (np.linalg.norm(v) or 1.0)
    if any(v is None for v in vectors):
        raise ValueError("Не все эмбеддинги получены")
    return np.array(vectors)


def embed_chunks(chunks):
    batches = [list(range(i, min(i + 16, len(chunks)))) for i in range(0, len(chunks), 16)]

    def run(ids):
        return ids, embed([chunks[i]["full"][:2500] for i in ids])

    vectors = [None] * len(chunks)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        for ids, vecs in pool.map(run, batches):
            for i, v in zip(ids, vecs):
                vectors[i] = v
    return np.array(vectors)


def rank_map(scores):
    order = np.argsort(-np.asarray(scores), kind="stable")
    return {int(i): r for r, i in enumerate(order)}


class Index:
    def __init__(self, chunks):
        self.chunks = chunks
        self.bm25 = BM25Okapi([tokens(c["full"]) for c in chunks], k1=1.4, b=0.75)
        self.vectors = None
        try:
            self.vectors = embed_chunks(chunks)
        except Exception as exc:
            print(f"Эмбеддинги базы недоступны ({type(exc).__name__}); только BM25", file=sys.stderr)
        self.namespaces = sorted({c["ns"] for c in chunks})

    def search(self, last, users):
        history = " ".join(users[:-1][-2:])
        query = last + (" " + history if history else "")
        lexical = self.bm25.get_scores(tokens(last + " " + last + " " + history))
        fused = np.zeros(len(self.chunks))
        for i, r in rank_map(lexical).items():
            fused[i] += 1.0 / (60 + r)
        if self.vectors is not None:
            try:
                dense = self.vectors @ embed([query])[0]
                for i, r in rank_map(dense).items():
                    fused[i] += 1.0 / (60 + r)
            except Exception as exc:
                print(f"Эмбеддинг запроса: {type(exc).__name__}", file=sys.stderr)
        order = [int(i) for i in np.argsort(-fused, kind="stable")]

        # Кандидаты: общий топ + лучшие фрагменты каждого пространства имён.
        candidates = order[:max(TOP_K * 3, 18)]
        best_by_ns = {}
        for i in order:
            ids = best_by_ns.setdefault(self.chunks[i]["ns"], [])
            if len(ids) < PER_NAMESPACE:
                ids.append(i)
        for ids in best_by_ns.values():
            candidates += [i for i in ids if i not in candidates]

        ranked = candidates
        try:
            result = post("/rerank", RERANK_KEY, {
                "model": RERANKER_MODEL, "query": query,
                "documents": [self.chunks[i]["full"][:2500] for i in candidates],
                "top_n": len(candidates)})
            entries = result.get("results", result.get("data", []))
            entries = sorted(entries, key=lambda e: float(e.get("relevance_score", e.get("score", 0))), reverse=True)
            reranked = [candidates[int(e["index"])] for e in entries if 0 <= int(e["index"]) < len(candidates)]
            if reranked:
                ranked = reranked
        except Exception as exc:
            print(f"Реранкер: {type(exc).__name__}; порядок гибридного поиска", file=sys.stderr)

        selected = ranked[:max(TOP_K, RERANKER_TOP_N)]
        # Ни одно пространство имён не остаётся без представителя.
        for ns in self.namespaces:
            if not any(self.chunks[i]["ns"] == ns for i in selected):
                extra = next((i for i in ranked if self.chunks[i]["ns"] == ns), None)
                if extra is not None:
                    selected.append(extra)
        return [self.chunks[i] for i in selected]


_index = None
_index_lock = threading.Lock()


def get_index():
    """Индекс строится один раз на процесс при первом обращении (или при прогреве)."""
    global _index
    if _index is None:
        with _index_lock:
            if _index is None:
                _index = Index(load_chunks(settings.KB_PATH))
    return _index


def warmup():
    threading.Thread(target=get_index, daemon=True).start()


# ---------------------------------------------------------------- генерация

LEAD_BLOCK = re.compile(r"<lead>(.*?)(?:</lead>|$)", re.S)


def clean(answer):
    answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.S)
    answer = re.sub(r"^.*?</think>", "", answer, flags=re.S)
    return answer.replace("**", "").strip()


def split_lead_block(answer):
    """Отделяет служебный блок <lead> от текста ответа. Возвращает (текст, поля контакта)."""
    contact = {}
    match = LEAD_BLOCK.search(answer)
    if match:
        try:
            data = json.loads(match.group(1))
            if isinstance(data, dict):
                contact = {k: str(data.get(k) or "").strip() for k in ("full_name", "phone", "company")}
        except ValueError:
            pass
        answer = LEAD_BLOCK.sub("", answer)
    return answer.strip(), contact


def chat_completion(messages, max_tokens=None):
    payload = {"model": LLM_MODEL, "temperature": TEMPERATURE,
               "max_tokens": max_tokens or MAX_TOKENS, "messages": messages}
    result = post("/chat/completions", LLM_KEY, payload, timeout=150)
    return clean(result["choices"][0]["message"].get("content") or "")


def generate_reply(dialog, known_contact):
    """dialog — список {'role', 'content'}, последняя запись — реплика пользователя.

    Возвращает (текст ответа, найденные поля контакта, использованные источники).
    """
    users = [m["content"] for m in dialog if m["role"] == "user"]
    sources = get_index().search(users[-1], users)
    context = "\n\n".join(f"[{c['label']}]\n{c['text']}" for c in sources)
    known = "; ".join(f"{label}: {known_contact[key]}" for key, label in
                      (("full_name", "ФИО"), ("phone", "телефон"), ("company", "компания")) if known_contact.get(key))
    system = SYSTEM_PROMPT + "\n\nУже известные контакты клиента: " + (known or "нет") + \
        "\n\nФрагменты базы знаний:\n\n" + context
    answer = ""
    for attempt in range(2):
        try:
            answer = chat_completion([{"role": "system", "content": system}] + dialog)
            if answer:
                break
        except Exception as exc:
            print(f"LLM попытка {attempt + 1}: {type(exc).__name__}", file=sys.stderr)
    text, contact = split_lead_block(answer)
    return text or FALLBACK, contact, [c["label"] for c in sources]


def summarize(dialog):
    transcript = "\n".join(("Клиент: " if m["role"] == "user" else "Консультант: ") + m["content"] for m in dialog)
    try:
        summary = chat_completion([{"role": "system", "content": SUMMARY_PROMPT},
                                   {"role": "user", "content": transcript}], max_tokens=600)
        if summary:
            return summary
    except Exception as exc:
        print(f"Резюме: {type(exc).__name__}", file=sys.stderr)
    return "Резюме не сформировано. Реплики клиента: " + " | ".join(
        m["content"] for m in dialog if m["role"] == "user")


# ---------------------------------------------------------------- диалоги и заявки

def get_or_create_session(session_id=None):
    if session_id:
        found = ChatSession.objects.filter(pk=session_id).first()
        if found:
            return found
    return ChatSession.objects.create()


def new_session():
    """«Новый диалог»: контекст сбрасывается, старая сессия остаётся в базе."""
    return ChatSession.objects.create()


def history(session):
    return [{"role": m.role, "content": m.content, "sources": m.sources, "elapsed_ms": m.elapsed_ms}
            for m in session.messages.all()]


def normalize_phone(value):
    digits = re.sub(r"\D", "", value or "")
    return value.strip() if 10 <= len(digits) <= 15 else ""


def merge_contact(session, found):
    """Пополняет контакты сессии новыми значениями; пустые не затирают известные."""
    values = {"full_name": (found.get("full_name") or "").strip()[:200],
              "phone": normalize_phone(found.get("phone"))[:50],
              "company": (found.get("company") or "").strip()[:200]}
    for field, value in values.items():
        if value:
            setattr(session, "contact_" + field.replace("full_", ""), value)
    session.save()


def create_lead_if_ready(session, dialog):
    """Заявка создаётся один раз, когда собраны ФИО, телефон и компания."""
    if not (session.contact_name and session.contact_phone and session.contact_company):
        return False
    if hasattr(session, "lead"):
        return False
    Lead.objects.create(session=session, full_name=session.contact_name, phone=session.contact_phone,
                        company=session.contact_company, summary=summarize(dialog))
    return True


def handle_user_message(session, text):
    """Полный цикл одной реплики: сохранить, найти источники, ответить, обновить заявку."""
    text = (text or "").strip()[:MAX_MESSAGE_LEN]
    if not text:
        raise ValueError("Пустое сообщение")
    started = time.time()
    dialog = [{"role": m.role, "content": m.content} for m in session.messages.all()][-HISTORY_LIMIT:]
    dialog.append({"role": "user", "content": text})
    Message.objects.create(session=session, role=Message.USER, content=text)

    known = {"full_name": session.contact_name, "phone": session.contact_phone, "company": session.contact_company}
    try:
        answer, found, sources = generate_reply(dialog, known)
    except Exception as exc:
        print(f"Ответ: {type(exc).__name__}: {exc}", file=sys.stderr)
        answer, found, sources = FALLBACK, {}, []
    elapsed_ms = int((time.time() - started) * 1000)

    Message.objects.create(session=session, role=Message.ASSISTANT, content=answer,
                           sources=sources, elapsed_ms=elapsed_ms)
    merge_contact(session, found)
    lead_created = create_lead_if_ready(session, dialog + [{"role": "assistant", "content": answer}])
    session.save()  # обновляет updated_at
    return {"answer": answer, "sources": sources, "elapsed_ms": elapsed_ms, "lead_created": lead_created}
