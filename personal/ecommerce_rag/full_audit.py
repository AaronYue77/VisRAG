"""Human readable full-catalog records without inventing PDF pages for metadata hits."""
import json
from common import WORKSPACE


def append_record(out,row):
    lines=['='*90,'运行名称：'+row['run_name'],'问题 ID：'+row['question_id'],'问题：'+row['question'],
      '参考答案：'+row['reference_answer'],'参考证据：'+str(row.get('reference_pages',[])),
      '证据类型：'+row.get('evidence_kind','metadata_record')+'；metadata标识不是PDF物理页码。',
      '召回记录及排名：'+json.dumps(row.get('retrieved_pages',[]),ensure_ascii=False),
      '模型答案：'+row.get('answer','未执行生成'),'输出引用：'+str(row.get('citations',[])),
      '检索耗时（秒）：'+str(row.get('retrieval_seconds',0)),'生成耗时（秒）：'+str(row.get('generation_seconds',0)),
      '模型版本：'+str(row.get('model_revision')),'配置版本SHA：'+row['config_sha256'],
      '检索轨迹：'+json.dumps(row.get('retrieval_trace',{}),ensure_ascii=False),
      '评测：'+json.dumps(row.get('metrics',{}),ensure_ascii=False),'执行状态：'+row['status'],'']
    text='\n'.join(lines)+'\n'
    for path in (out/'逐题结果.txt',WORKSPACE/'记录/逐题结果汇总.txt'):
        with path.open('a',encoding='utf-8') as f:f.write(text)
