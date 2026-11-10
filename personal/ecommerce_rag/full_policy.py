"""Read-only, explicitly scoped answers from a reviewed official policy snapshot."""
import json
import re

from common import WORKSPACE, file_hash


POLICY_FILE = WORKSPACE / "data/ecommerce_rag/policies-us-v1/clauses.json"
SITE = re.compile(r"(?<![a-z0-9.])(?:www\.)?(amazon\.[a-z]+(?:\.[a-z]+)?)(?![a-z0-9.])", re.I)
US_ALIAS = re.compile(r"美国亚马逊|亚马逊美国站")
RETURNS = re.compile(r"(?:一般|通常|标准)?\s*(?:的)?\s*(?:一般|通常|标准)?\s*退货"
                     r"(?:政策|规则|期限|时限|窗口)?(?:是什么|是多少|多久|多少天|有哪些|是怎样的|怎么规定的|呢)?")
REFUNDS = re.compile(r"(?:的)?\s*(?:退货后)?\s*退款\s*(?:通常|一般)?\s*(?:需要)?(?:多久|多少天)(?:能到账|到账)?")


def policy_answer(question, domain=None):
    """Return None for unrelated turns. Personalized outcomes stay at the business boundary."""
    sites = {match.casefold() for match in SITE.findall(question)}
    if US_ALIAS.search(question):
        sites.add("amazon.com")
    if not sites and not domain:
        return None
    subject = SITE.sub("", US_ALIAS.sub("", question)).strip(" ：:，,？?。！!")
    subject = re.sub(r"^(?:请问|想问一下|当前|现行)\s*", "", subject)
    kind = "general_returns" if RETURNS.fullmatch(subject) else "general_refunds" if REFUNDS.fullmatch(subject) else None
    if not kind:
        if re.search(r"退货政策|退货规则|退货期限|保修政策|保修期限", subject):
            return {"action": "not_found", "answer": "现有政策快照只覆盖标准退货期限和一般退款时效。"
                    "无法核实这次所问的具体情形，请查看对应站点官方条款。",
                    "answer_source": "policy_clause_unavailable"}
        return None
    if domain:
        sites.add(domain.casefold())
    if len(sites) != 1:
        return {"action": "clarify", "answer": "你提到了不同站点，请确认要查哪一个站点的规则。",
                "answer_source": "policy_scope_conflict"}
    site = sites.pop()
    if site != "amazon.com":
        return {"action": "not_found", "answer": f"目前没有 {site} 的已核对政策快照。",
                "answer_source": "policy_scope_unavailable"}
    if not POLICY_FILE.is_file():
        return {"action": "not_found", "answer": "本地政策快照不可用，请查看该站点的官方帮助。",
                "answer_source": "policy_snapshot_unavailable"}
    snapshot = json.loads(POLICY_FILE.read_text("utf-8"))
    if snapshot["marketplace"] != site:
        raise ValueError("Policy snapshot marketplace differs from the request")
    clause = snapshot["clauses"][kind]
    citation = f"{snapshot['version']}:{kind}"
    page = {"page_id": citation, "doc_id": snapshot["version"], "page": None, "rank": 1,
            "score": 1.0, "text_source": "official_policy_snapshot", "marketplace": site,
            "source_url": clause["url"], "section": clause["section"], "observed_on": snapshot["observed_on"]}
    return {"action": "answer", "answer": f"根据 {snapshot['observed_on']} 核对的官方快照，{clause['text']}\n官方来源：{clause['url']}",
            "answer_source": "official_policy_snapshot", "citations": [citation], "retrieved_pages": [page],
            "retrieval_trace": {"scope": site, "snapshot": snapshot["version"], "observed_on": snapshot["observed_on"],
                                "snapshot_sha256": file_hash(POLICY_FILE), "source": clause["url"]}}
