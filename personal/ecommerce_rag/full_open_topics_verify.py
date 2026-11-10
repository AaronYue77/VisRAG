"""Open-ended, new-topic real-model smoke checks; topics are tests, not allowlists."""
import json
from pathlib import Path
import time
from full_app import FullApplication
from common import WORKSPACE,write_json,read_jsonl,operation


def run():
    out=WORKSPACE/'reports/ecommerce_rag'/('open-topics-'+time.strftime('%Y%m%d-%H%M%S'))
    out.mkdir();app=FullApplication('B',report_root=out);s=app.session();checks=[]
    examples=[
      ('为什么地球会有四季？请简短说明。','天文'),
      ('印刷术对知识传播有什么影响？','历史'),
      ('怎样让炒鸡蛋更嫩？','烹饪'),
      ('请把 I will get back to you tomorrow 翻译成自然的中文。','翻译'),
      ('写四句关于雨后街道的小诗。','创作'),
      ('把刚才的小诗改成轻松活泼的语气。','创作追问'),
      ('这件事让我很纠结，朋友总迟到，我该怎么和他沟通？','人际交流'),
      ('这把游戏又输了，我该怎么复盘和调整心态？','游戏与情绪'),
      ('搜索算法和排序算法有什么区别？','算法概念'),
      ('查找字符串里重复出现的字符，给我一个Python实现。','编程'),
      ('这个函数的时间复杂度是多少？','编程追问'),
      ('一个国家为什么可能有多个时区？','地理'),
      ('周末在家整理书桌，东西太杂，怎么开始？','生活安排'),
      ('每天记住5个新单词，一周一共能记多少个？','简单计算'),
      ('用英文给第一次做演讲的人一句简短鼓励。','英语表达'),
      ('写一个正则表达式，提取示例字符串中的B07C5PRQHN这个编号。','含编号的编程数据')]
    try:
        for i,(q,topic) in enumerate(examples):
            if i==3:
                chosen=s.respond('B07C5PRQHN@amazon.ca');assert chosen['action']=='selected'
            v=s.respond(q);r=read_jsonl(s.out/'results.jsonl')[-1]
            ok=v['status']=='ok' and v['action']=='chat' and not v['citations'] and not v['retrieved_pages'] and not r.get('catalog_items') and r['retrieval_seconds']==0
            if i>=3:ok&=v['current_listing_keys']==['B07C5PRQHN@amazon.ca']
            checks.append({'topic':topic,'question':q,'passed':bool(ok),'action':v['action'],'answer':v['answer'],'routing':r['routing'],
                 'retrieval_seconds':r['retrieval_seconds'],'citations':v['citations'],'record_txt':v['record_txt']})
            write_json(out/'checks.json',checks)
            print(topic,v['action'],bool(ok),v['answer'][:110].replace('\n',' / '),flush=True)
        v=s.respond('这款商品的品牌原文是什么？');r=read_jsonl(s.out/'results.jsonl')[-1]
        checks.append({'topic':'商品取证控制','question':'这款商品的品牌原文是什么？','passed':v['status']=='ok' and v['action']=='answer' and 'AmazonBasics' in v['answer'] and bool(v['citations']),
           'action':v['action'],'answer':v['answer'],'citations':v['citations'],'record_txt':v['record_txt']})
        summary={'checks':len(checks),'passed':sum(c['passed'] for c in checks),'question_txt':str(s.out/'逐题结果.txt'),'conversation_txt':str(s.out/'对话记录.txt'),
          'note':'Real local Qwen outputs. These topics exercise the generic route and are not allowed-topic lists. Passing means routing/no unwanted lookup/context continuity; content needs separate review.'}
        write_json(out/'checks.json',checks);write_json(out/'summary.json',summary)
        (out/'开放话题实测.txt').write_text('开放话题真实模型检查\n'+json.dumps(summary,ensure_ascii=False,indent=2)+'\n\n'+'\n\n'.join(json.dumps(c,ensure_ascii=False,indent=2) for c in checks),'utf-8')
        operation('新话题开放对话真实模型验证','16个新主题/追问：开始不选商品，再保留商品上下文聊其他话题；每轮检查无商品检索/引用，最后商品问题重新取证。',
          json.dumps(summary,ensure_ascii=False),[out/'开放话题实测.txt',s.out/'逐题结果.txt',s.out/'对话记录.txt'],
          '不把测试主题变成回答白名单；不把路由通过称一般知识全正确。',status='完成' if summary['checks']==summary['passed'] else '存在失败待处理')
        print(json.dumps(summary,ensure_ascii=False),flush=True)
    finally:app.close()
    assert summary['checks']==summary['passed']
    return out


if __name__=='__main__':run()
