"""Local A-stage API/CLI. A routed RAG workflow, without Agent/tool execution."""
import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re
import threading
import time
import uuid

from stage_a import Reader, MODEL_REVISION, PROMPT_VERSION, snapshot
from stage_a_index import BM25Retriever, DEFAULT_CORPUS
from common import WORKSPACE, timestamp, file_hash, write_json, write_jsonl, operation, issue
from audit import append_question


class Application:
    def __init__(self, corpus=DEFAULT_CORPUS, reader_factory=Reader):
        self.corpus=Path(corpus)
        frozen=json.loads((self.corpus/"frozen-test-config.json").read_text("utf-8"))
        self.settings=frozen["settings"]
        self.retriever=BM25Retriever(self.corpus,self.settings["strategy"],self.settings["overlap"])
        self.reader=None
        self.reader_factory=reader_factory
        self.model_load_count=0
        self.lock=threading.Lock()  # One GPU request at a time on 8GB VRAM.
        self.sessions={}
        self.sessions_lock=threading.Lock()

    def create_run(self, prefix="abo-interactive"):
        name=prefix+"-"+time.strftime("%Y%m%d-%H%M%S")+"-"+uuid.uuid4().hex[:6]
        output=WORKSPACE/"reports/ecommerce_rag"/name
        output.mkdir(parents=True)
        config={**self.settings,"schema_version":"2.1","source_sha256":snapshot(output),
                "source_type":"official_ABO_derived_PDF","mode":"bm25","evaluation":"unlabelled_user_request",
                "context_policy":"Only explicit product/store/snapshot context; past answers are not factual evidence."}
        write_json(output/"config.json",config)
        return output

    def _row(self, question, request, output, found, context=None):
        config=json.loads((output/"config.json").read_text("utf-8"))
        return {"question_id":output.name+"-"+uuid.uuid4().hex[:8],"question":question,"request":request,
                "reference_answer":"未标注真实请求，不参与基准评分","reference_pages":[],
                "run_name":output.name,"run_directory":str(output.relative_to(WORKSPACE)).replace("\\","/"),
                "timestamp":timestamp(),"status":"ok","schema_version":"2.1",
                "source_sha256":config["source_sha256"],"config_sha256":file_hash(output/"config.json"),
                "routing":found["plan"],"retrieval_trace":found["trace"],"retrieval_seconds":found["seconds"],
                "generation_seconds":0.0,"model":None,"model_revision":None,"raw_output":"","citations":[],
                "retrieved_pages":[{k:p[k] for k in ("page_id","doc_id","page","sku","rank","score","text_source")} for p in found["pages"]],
                "metrics":{"evaluation":"unlabelled"}, **(context or {})}

    def _save(self, output, row):
        row["model_load_count"]=self.model_load_count
        with (output/"results.jsonl").open("a",encoding="utf-8",newline="\n") as stream:
            stream.write(json.dumps(row,ensure_ascii=False)+"\n");stream.flush()
        append_question(output,row)
        return {"action":row["action"],"answer":row["answer"],"citations":row["citations"],
                "record_txt":str(output/"逐题结果.txt"),"retrieved_pages":row["retrieved_pages"],
                "status":row["status"],"model_load_count":self.model_load_count}

    def local_reply(self, question, request, action, answer, *, output, context=None, reason="conversation", error=None):
        with self.lock:
            found={"plan":{"action":action,"reason":reason},"trace":{"mode":"conversation_rule","model_called":False},"seconds":0.0,"pages":[]}
            row=self._row(question,request,output,found,context)
            row.update(action=action,answer=answer,answer_source="conversation_rule",total_seconds=0.0)
            if error:row.update(status="error",error=error)
            return self._save(output,row)

    def ask(self,question,request=None,*,output=None,context=None):
        request=request or {}
        if not isinstance(request,dict) or set(request)-{"skus","store_id","as_of"}:
            raise ValueError("Only skus, store_id and as_of are public request context")
        with self.lock:
            started=time.perf_counter()
            output=output or self.create_run()
            found={"plan":{"action":"error","reason":"runtime_error"},"trace":{},"seconds":0.0,"pages":[]}
            row=None
            try:
                found=self.retriever.search(question,request,self.settings["top_k"])
                pages=found["pages"]
                row=self._row(question,request,output,found,context)
                if found["plan"]["action"]=="retrieve":
                    if self.reader is None:
                        self.reader=self.reader_factory()
                        self.model_load_count+=1
                    row.update(model="Qwen/Qwen3.5-2B",model_revision=MODEL_REVISION)
                    row.update(self.reader.read(question,request,pages,output/"images",self.settings["max_pixels"]))
                else:
                    row.update(action=found["plan"]["action"],answer=found["plan"]["answer"],answer_source="rule_router")
            except KeyboardInterrupt:
                row=row or self._row(question,request,output,found,context)
                row.update(status="cancelled",action="cancelled",answer="本轮已取消",answer_source="interrupted")
                row["total_seconds"]=time.perf_counter()-started
                self._save(output,row)
                raise
            except Exception as error:
                row=row or self._row(question,request,output,found,context)
                row.update(status="error",action="error",answer="本轮处理失败，记录已保存；可以重新提问或切换商品。",
                           answer_source="runtime_error",error=type(error).__name__+": "+str(error))
                row["issue_id"]=issue("商品交互失败："+row["error"],"检查输入、页面与模型运行状态。","保留失败题和会话，继续接收下一条；按错误记录排查。")
            row["total_seconds"]=time.perf_counter()-started
            result=self._save(output,row)
            operation("处理真实商品交互请求","复用冻结 A 配置；输入校验→规则路由→BM25→图片VLM；GPU串行。",row["action"]+"："+row["answer"],[output/"逐题结果.txt"],"真实无标签请求，不混入开发/测试准确率。")
            return result

    def respond(self, question, request=None, session_id=None):
        from stage_a_chat import ChatSession
        if session_id is not None and (not isinstance(session_id,str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}",session_id)):
            raise ValueError("Invalid session_id")
        with self.sessions_lock:
            key=session_id or uuid.uuid4().hex
            session=self.sessions.get(key)
            if session is None:
                # Context expires after 30 minutes idle; model remains shared by this application.
                for old, value in list(self.sessions.items()):
                    if time.monotonic()-value.last_used>1800 and value.lock.acquire(blocking=False):
                        try:
                            value.close("idle_timeout")
                            self.sessions.pop(old,None)
                        finally:value.lock.release()
                if len(self.sessions)>=64:raise ValueError("Too many active sessions")
                session=ChatSession(self,session_id=key)
                self.sessions[key]=session
        result=session.respond(question,request)
        if session.ended:
            with self.sessions_lock:self.sessions.pop(key,None)
        return result


def serve(corpus,port):
    app=Application(corpus);slots=threading.BoundedSemaphore(2)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def reply(self,status,value):
            body=json.dumps(value,ensure_ascii=False).encode("utf-8")
            self.send_response(status);self.send_header("Content-Type","application/json; charset=utf-8");self.send_header("Content-Length",str(len(body)));self.end_headers();self.wfile.write(body)
        def do_GET(self):
            self.reply(200,{"status":"ready","workflow":"routed_multimodal_rag","agent":False,"model_loaded":app.reader is not None}) if self.path=="/health" else self.reply(404,{"error":"not_found"})
        def do_POST(self):
            if self.path!="/ask":return self.reply(404,{"error":"not_found"})
            if not slots.acquire(blocking=False):return self.reply(429,{"error":"busy"})
            try:
                size=int(self.headers.get("Content-Length","0"))
                if not 0<size<=16384:raise ValueError("Request must be 1..16384 bytes")
                body=json.loads(self.rfile.read(size))
                if not isinstance(body,dict) or set(body)-{"question","request","session_id"}:raise ValueError("Expected question, optional request and session_id")
                result=app.respond(body.get("question"),body.get("request"),body.get("session_id"))
                self.reply(500 if result["action"]=="error" else 200,result)
            except (ValueError,TypeError) as error:self.reply(400,{"error":str(error)})
            except Exception as error:
                issue("本地API请求失败："+type(error).__name__+": "+str(error),"检查模型、输入页面和服务运行环境。","保留异常，修复后使用新交互记录。")
                self.reply(500,{"error":type(error).__name__})
            finally:slots.release()
    operation("启动本地电商 RAG API","标准 HTTP JSON 接口，localhost 绑定；健康检查、输入限制、GPU串行和最多2个在途请求。",f"http://127.0.0.1:{port}；不是云端上线。",verification="随后独立客户端验证 health、路由、真实图像问答及错误请求。",status="进行中")
    print(f"Listening on http://127.0.0.1:{port}",flush=True)
    server=ThreadingHTTPServer(("127.0.0.1",port),Handler)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:
        server.server_close()
        operation("停止本地RAG API","Ctrl+C结束本次本机服务并关闭监听。","当前服务已停止，模型进程退出释放GPU资源。",verification="仅停止本任务服务；可按启动命令重新运行。")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=["ask","serve","chat"])
    parser.add_argument("--question",default="")
    parser.add_argument("--sku",action="append",default=[])
    parser.add_argument("--corpus",default=str(DEFAULT_CORPUS))
    parser.add_argument("--port",type=int,default=8765)
    args=parser.parse_args()
    if args.command=="serve":serve(args.corpus,args.port)
    elif args.command=="chat":
        from stage_a_chat import run_chat
        run_chat(Application(args.corpus),initial_skus=args.sku)
    else:print(json.dumps(Application(args.corpus).ask(args.question,{"skus":args.sku}),ensure_ascii=False,indent=2))
