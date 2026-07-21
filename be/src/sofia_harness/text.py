from __future__ import annotations
import re, unicodedata

def normalize_search(text:str)->str:
    text=unicodedata.normalize("NFC",text)
    text=re.sub(r"-\s*\n\s*","",text)
    return re.sub(r"\s+"," ",text).strip()

def reconstruct_articles(page:dict)->list[dict]:
    groups={}
    for item in page.get("regions",[]):
        region=item.get("region",item); article_id=region.get("article_id") or f"{page['page_id']}:ungrouped"
        text=item.get("accepted_text") or item.get("text") or (item.get("read_a") or {}).get("verbatim_text") or ""
        groups.setdefault(article_id,[]).append((region.get("reading_order",9999),region.get("type"),text,region.get("confidence"),region.get("id")))
    output=[]
    for article_id,rows in groups.items():
        rows.sort(); title=next((text for _,kind,text,_,_ in rows if kind=="headline" and text),None)
        verbatim="\n\n".join(text for _,_,text,_,_ in rows if text)
        confidence=min((c for *_,c,_ in rows if c is not None),default=None)
        output.append({"article_id":article_id,"title":title,"verbatim_text":verbatim,"normalized_text":normalize_search(verbatim),"confidence":confidence,"region_ids":[r[-1] for r in rows]})
    return output

ENTITY_PATTERN=re.compile(r"\b[А-ЯѢ][а-яѣъь]+(?:\s+[А-Я][а-яѣъь]+){1,3}\b")
QUOTED_UPPER_ENTITY_PATTERN=re.compile(r"[„\"](?P<name>[А-ЯѢ]{2,}(?:\s+[А-ЯѢ]{2,}){1,3})[“\"]")
DATE_PATTERN=re.compile(r"\b(?:[0-3]?\d[.\-/][01]?\d[.\-/](?:18|19|20)\d{2}|(?:18|19|20)\d{2}\s*г\.)\b")

def extract_entities(text:str):
    found=[]; seen=set()
    for match in ENTITY_PATTERN.finditer(text):found.append(("named_entity",match.group(),match.start(),match.end()))
    for match in QUOTED_UPPER_ENTITY_PATTERN.finditer(text):
        start,end=match.span("name"); key=(start,end)
        if key not in seen: found.append(("named_entity",match.group("name"),start,end));seen.add(key)
    for match in DATE_PATTERN.finditer(text):found.append(("date",match.group(),match.start(),match.end()))
    return found
