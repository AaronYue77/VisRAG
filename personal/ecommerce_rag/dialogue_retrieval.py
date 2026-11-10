"""BM25 page retrieval only. No dense model, reranker, or retrieval retry."""
import re
import time
from stage_a_index import BM25Retriever
from core import tokenize

KEYWORDS={"width":"width 宽度", "height":"height 高度", "length":"length 长度", "weight":"weight 重量",
 "color":"color 颜色", "material":"material 材质", "brand":"brand 品牌", "model":"model_number 型号",
 "assembly":"assembly assemble 组装 安装", "cleaning":"wipe clean cloth cleaning 清洁 擦拭",
 "capacity":"weight capacity 承重", "age":"ages age youth kids children 适用年龄",
 "features":"features reclining storage assembly 产品特点", "appearance":"color arm back image 图片 扶手",
 "comparison":"width dimensions weight 规格", "policy_history":"returns warranty 历史原文 保修 退货"}


class DialogueRetriever:
    def __init__(self,corpus):
        self.base=BM25Retriever(corpus,strategy="blocks",overlap=0)
        self.documents=self.base.documents

    def search(self, question, skus, topic, needs, top_k=2):
        started=time.perf_counter()
        lookup=self.base.by_id
        if topic=="fit":
            active=needs.get('_active_topic')
            if active in KEYWORDS and active not in {'fit','comparison','policy_history'}:topic=active
            elif needs.get('space'):topic="width"
            elif re.search(r"清洁|维护|打理",needs.get('focus','')):topic="cleaning"
            elif re.search(r"孩子|儿童|小孩",needs.get('purpose','')+needs.get('focus','')):topic="age"
            else:topic="features"
        query=KEYWORDS.get(topic,question[:350])
        scores=self.base.bm25.get_scores(tokenize(query))
        allowed={p['page_id'] for p in self.base.pages if p['sku'] in skus and p['store_id']=='amazon.com'}
        order=sorted((i for i,c in enumerate(self.base.chunks) if c['page_id'] in allowed),
                     key=lambda i:(-float(scores[i]),self.base.chunks[i]['chunk_id']))
        unique=[]
        seen=set()
        for i in order:
            pid=self.base.chunks[i]['page_id']
            if pid not in seen:unique.append(i);seen.add(pid)
        if len(skus)>1:
            reserve=[next(i for i in unique if lookup[self.base.chunks[i]['page_id']]['sku']==sku) for sku in skus]
            unique=reserve+[i for i in unique if i not in reserve]
        if len(skus)==1:
            top_k=1  # A single product uses its highest-ranked page; comparisons reserve one page per product.
        pages=[{**lookup[self.base.chunks[i]['page_id']],"rank":rank,"score":float(scores[i])}
               for rank,i in enumerate(unique[:top_k],1)]
        return pages,{"mode":"bm25","normalized_topic":topic,"keyword_query":query,"eligible_pages":len(allowed),
                      "top_k":top_k,"page_ranking":[self.base.chunks[i]['page_id'] for i in unique],
                      "note":"Canonical topic keywords, one retrieval pass; no generative rewrite or retry."},time.perf_counter()-started
