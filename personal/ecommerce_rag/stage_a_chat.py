"""Terminal/HTTP conversations with explicit product context and per-turn records."""
from datetime import date
import json
from pathlib import Path
import re
import sys
import threading
import time
import uuid

sys.path.insert(1,str(Path(__file__).resolve().parent.parent/"rag_common"))
from common import timestamp, write_json, operation, issue

EXIT_WORDS={"退出","结束","exit","quit","q"}
RESET_WORDS={"清空","重置","清空上下文","切换商品","reset"}
HELP_WORDS={"帮助","help","?","？","你是谁","你能做什么"}
LIST_WORDS={"商品列表","列表","list"}
GREETING=re.compile(r"(?:你好[呀啊]?|您好|嗨|哈喽|早上好|晚上好|hello|hi|hey)[\s，,!！。.?？~～]*",re.I)
THANKS=re.compile(r"(?:谢谢|感谢|多谢|thanks|thank you)[\s，,!！。.?？~～]*",re.I)


class ChatSession:
    def __init__(self, app, initial_skus=None, session_id=None):
        self.app=app
        self.session_id=session_id or uuid.uuid4().hex
        self.documents=app.retriever.documents
        self.known={d["sku"].upper() for d in self.documents}
        skus=sorted(set(s.upper() for s in (initial_skus or [])))
        if len(skus)>2 or not set(skus)<=self.known:
            raise ValueError("Initial context must contain one or two known product IDs")
        self.state={"skus":skus,"store_id":self.documents[0]["store_id"],"as_of":self.documents[0]["snapshot_date"]}
        self.output=app.create_run("abo-chat")
        self.transcript=self.output/"对话记录.txt"
        self.transcript.write_text("商品助手连续对话记录\n商品事实来自当前静态资料；历史回答不作为新的事实证据。\n", "utf-8")
        self.turn=0
        self.ended=False
        self.last_used=time.monotonic()
        self.lock=threading.RLock()
        self._save_state()

    def _save_state(self):
        write_json(self.output/"session.json",{"session_id":self.session_id,"turns":self.turn,
                   "current_request":self.state,"ended":self.ended,"model_load_count":self.app.model_load_count,
                   "updated_at":timestamp(),"memory_scope":"Product/store/snapshot only; no previous answers in model prompt."})

    def _mentions(self, question):
        upper=question.upper()
        values={sku for sku in self.known if re.search(r"(?<![A-Z0-9])"+re.escape(sku)+r"(?![A-Z0-9])",upper)}
        values.update(re.findall(r"\bB0[A-Z0-9]{8}\b",upper))
        return sorted(values)

    def _local(self, question, action, answer, request, context, reason):
        return self.app.local_reply(question,request,action,answer,output=self.output,context=context,reason=reason)

    def respond(self, question, request=None):
        if not isinstance(question,str):raise ValueError("question must be a string")
        if request is not None and (not isinstance(request,dict) or set(request)-{"skus","store_id","as_of"}):
            raise ValueError("Only skus, store_id and as_of are public request context")
        supplied=dict(request or {})
        if "skus" in supplied and (not isinstance(supplied["skus"],list) or any(not isinstance(s,str) for s in supplied["skus"])):
            raise ValueError("skus must be an array of strings")
        if "as_of" in supplied:date.fromisoformat(supplied["as_of"])
        with self.lock:
            if self.ended:raise ValueError("Session already ended")
            self.turn+=1
            self.last_used=time.monotonic()
            before={**self.state,"skus":list(self.state["skus"])}
            context={"session_id":self.session_id,"turn_number":self.turn,"context_before":before}
            text=question.strip()
            word=text.casefold()
            mentions=self._mentions(text)
            effective={**self.state,**supplied}
            # Explicit product changes replace old context; caller-provided conflicting IDs still clarify.
            if mentions and "skus" not in supplied:effective["skus"]=mentions
            effective["skus"]=sorted(set(s.upper() for s in effective["skus"]))
            try:
                if not text or len(text)>3000:
                    result=self._local(question,"clarify","请输入1到3000个字符的问题。",before,context,"question_length")
                elif word in EXIT_WORDS:
                    result=self._local(question,"exit","对话已结束。",before,context,"user_exit")
                    self.ended=True
                elif word in RESET_WORDS:
                    self.state={**self.state,"skus":[]}
                    result=self._local(question,"reset","已清空当前商品，请输入新的商品编号。",self.state,context,"reset_product")
                elif GREETING.fullmatch(text):
                    result=self._local(question,"greeting","你好，我可以帮你查询商品资料。请输入商品编号和问题；输入“商品列表”查看可选商品，输入“退出”结束。",before,context,"greeting")
                elif THANKS.fullmatch(text):
                    result=self._local(question,"thanks","不客气，你可以继续询问当前商品。",before,context,"thanks")
                elif word in HELP_WORDS:
                    result=self._local(question,"help","可问颜色、材质、尺寸、重量和图片信息；首次请提供商品编号。命令：商品列表、当前商品、重置、退出。当前只查询静态资料。",before,context,"help")
                elif word in LIST_WORDS:
                    products={d["sku"]:d["title"] for d in self.documents}
                    result=self._local(question,"catalog","可选商品：\n"+"\n".join(sku+" | "+title for sku,title in sorted(products.items())),before,context,"product_list")
                elif word in {"当前商品","当前型号","context"}:
                    result=self._local(question,"context","当前商品："+(", ".join(self.state["skus"]) or "尚未选择，请输入商品编号。"),before,context,"show_context")
                elif (not mentions and "skus" not in supplied and len(self.state["skus"])>1
                      and not re.search(r"分别|比较|两[个款者]|它们|这些",text)):
                    result=self._local(question,"clarify","当前有两个商品，请指定编号，或说明要分别比较两者。",before,context,"ambiguous_product")
                else:
                    selection=bool(mentions and re.fullmatch(r"(?:选择|选中|切换到|切换为|换成|select|use)?\s*[:：]?\s*"+re.escape(mentions[0]),text,re.I))
                    mismatch=bool(mentions and "skus" in supplied and not set(mentions)<=set(effective["skus"]))
                    # Selection reflects the user's intent even when this turn's model call fails.
                    if not mismatch and (mentions or "skus" in supplied):
                        if len(effective["skus"])<=2 and set(effective["skus"])<=self.known:
                            self.state=effective
                        else:self.state={**self.state,"skus":[]}
                    if selection and set(effective["skus"])<=self.known and not mismatch:
                        result=self._local(question,"selected","已选择商品 "+", ".join(effective["skus"])+"，现在可以直接问它的宽度、重量等。",effective,context,"select_product")
                    else:
                        result=self.app.ask(question,effective,output=self.output,context=context)
                    if result["status"]=="ok" and result["action"]!="clarify":
                        if set(effective["skus"])<=self.known and len(effective["skus"])<=2:
                            self.state=effective
                        elif mentions or "skus" in supplied:
                            self.state={**self.state,"skus":[]}
            except KeyboardInterrupt:
                self.close("keyboard_interrupt")
                raise
            except Exception as error:
                key=issue("连续对话本轮失败："+type(error).__name__+": "+str(error),"检查会话、输入或后端异常。","保留会话上下文与错误记录，继续接收下一条问题。")
                result=self.app.local_reply(question,effective,"error","本轮处理失败，可以继续提问。",output=self.output,context={**context,"issue_id":key},error=type(error).__name__+": "+str(error))
                result["issue_id"]=key
            result.update(session_id=self.session_id,current_skus=list(self.state["skus"]),turn_number=self.turn,
                          conversation_txt=str(self.transcript))
            with self.transcript.open("a",encoding="utf-8",newline="\n") as stream:
                stream.write(f"\n[{timestamp()}] 第{self.turn}轮\n你：{question}\n助手：{result['answer']}\n引用：{result['citations']}\n当前商品：{self.state['skus']}\n逐题记录：{result['record_txt']}\n")
            with (self.output/"turns.jsonl").open("a",encoding="utf-8",newline="\n") as stream:
                stream.write(json.dumps({"question":question,"context_before":before,"context_after":self.state,**result},ensure_ascii=False)+"\n")
            self._save_state()
            return result

    def close(self, reason):
        with self.lock:
            self.ended=True
            self._save_state()
            with self.transcript.open("a",encoding="utf-8",newline="\n") as stream:
                stream.write(f"\n会话结束：{reason}；本进程模型累计加载次数：{self.app.model_load_count}。\n")


def run_chat(app, initial_skus=None):
    session=ChatSession(app,initial_skus=initial_skus)
    print("商品助手已启动，输入“退出”结束。输入“商品列表”查看商品，输入“重置”清空当前商品。",flush=True)
    print("首次商品问答会加载模型，后续提问复用。当前商品："+(", ".join(session.state["skus"]) or "尚未选择"),flush=True)
    print("对话记录："+str(session.transcript),flush=True)
    operation("启动终端连续聊天","创建一次Application和独立商品会话；循环读取自然语言输入，复用冻结RAG和本地模型。",session.session_id,[session.transcript],status="进行中")
    reason="user_exit"
    try:
        while not session.ended:
            try:question=input("\n你：")
            except EOFError:
                reason="end_of_input";break
            if not question.strip():continue
            result=session.respond(question)
            print("助手："+result["answer"],flush=True)
            if result["citations"]:
                pages={p["page_id"]:p for p in result["retrieved_pages"]}
                print("依据："+"；".join(f"{pages[c]['sku']} 商品资料第{pages[c]['page']}页（{c}）" if c in pages else c for c in result["citations"]),flush=True)
            print("逐题记录："+result["record_txt"],flush=True)
    except KeyboardInterrupt:
        reason="keyboard_interrupt"
        print("\n助手：已停止对话。",flush=True)
    finally:
        session.close(reason)
        print("对话记录已保存："+str(session.transcript),flush=True)
        operation("结束终端连续聊天","退出词、EOF或Ctrl+C结束输入循环；每轮保留原始结果，关闭本次模型进程。",
                  f"{session.turn}轮，模型加载{app.model_load_count}次，结束原因{reason}。",[session.transcript,session.output/"逐题结果.txt"],"会话结束不删除历史；模型随Python进程退出释放。")
