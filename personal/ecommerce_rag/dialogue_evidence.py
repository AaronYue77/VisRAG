"""Deterministic presentation and conflicts from CURRENT provided pages only."""
import math
import re

FIELDS={'width':('宽度','W'),'height':('高度','H'),'length':('长度','L'),'weight':('重量',None)}

def field_excerpt(quote,topic):
    if topic not in FIELDS:return quote
    # Trim only a fully verified quote, retaining an exact complete numeric field line.
    for line in quote.splitlines():
        if re.search(r'\b'+topic+r'\s*[:：]\s*\d+(?:\.\d+)?\s*(?:inches|inch|pounds|kg|cm|厘米)',line,re.I):return line.strip()
    return quote

def current_page_conflicts(pages,topic):
    """No title-only metadata or historical reply can introduce a factual warning."""
    topics=[topic] if topic in FIELDS else ['width'] if topic=='fit' else []
    result=[]
    for p in pages:
        for field in topics:
            label,axis=FIELDS[field]
            if axis is None:continue
            spec=re.search(r'[^\n]*\b'+field+r'\s*[:：]\s*(\d+(?:\.\d+)?)\s*(inches|inch)[^\n]*',p['text'],re.I)
            if not spec:continue
            # Explicit inch+axis notation only; D is not silently treated as length.
            alternatives=list(re.finditer(r'(\d+(?:\.\d+)?)\s*["”″]\s*'+axis+r'\b',p['text'],re.I))
            for other in alternatives:
                if math.isclose(float(spec[1]),float(other[1]),abs_tol=1e-6):continue
                line=p['text'][p['text'].rfind('\n',0,other.start())+1:p['text'].find('\n',other.end()) if '\n' in p['text'][other.end():] else len(p['text'])].strip()
                result.append({'sku':p['sku'],'page_id':p['page_id'],'page':p['page'],'field':field,
                    'spec_value':spec[1],'other_value':other[1],'unit':'inches','spec_quote':spec[0].strip(),'other_quote':line})
                break
    return result

def conflict_notice(conflicts):
    return '\n'.join(f"来源冲突：{c['sku']}同一提供页中{FIELDS[c['field']][0]}属性为{c['spec_value']} inches，尺寸文案为{c['other_value']} inches【第{c['page']}页】。保留两种原文，不能确认实物尺寸；请向商家核实。" for c in conflicts)
