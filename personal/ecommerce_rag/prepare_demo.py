"""Create explicitly synthetic Chinese product manuals for functional checks only."""
import sys
from pathlib import Path
import json

HERE = Path(__file__).resolve().parent
sys.path.insert(1, str(HERE.parent / "a_baseline"))
from common import WORKSPACE, write_json, write_jsonl, file_hash, operation


def main():
    from reportlab.pdfgen import canvas
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.lib import colors
    pdfmetrics.registerFont(TTFont("DemoChinese", "C:/Windows/Fonts/simhei.ttf"))
    root = WORKSPACE / "data/ecommerce_rag/demo-v1"
    pdf_root = root / "documents"
    pdf_root.mkdir(parents=True, exist_ok=True)
    records = []
    products = [
        {"doc_id": "dn-r100-2025", "sku": "DN-R100", "version": "2025.1", "valid_from": "2025-01-01", "valid_to": "2025-12-31", "wifi": "Wi-Fi 5 / AC1200", "wan": "1000 Mbps", "lan": "2 个千兆 LAN 口", "power": "12V / 1A", "warranty": "12 个月", "color": "blue", "ports": 2},
        {"doc_id": "dn-r100-2026", "sku": "DN-R100", "version": "2026.1", "valid_from": "2026-01-01", "valid_to": None, "wifi": "Wi-Fi 6 / AX1800", "wan": "1000 Mbps", "lan": "2 个千兆 LAN 口", "power": "12V / 1A", "warranty": "24 个月", "color": "blue", "ports": 2},
        {"doc_id": "dn-r200-2026", "sku": "DN-R200", "version": "2026.1", "valid_from": "2026-01-01", "valid_to": None, "wifi": "Wi-Fi 6 / AX3000", "wan": "2500 Mbps", "lan": "3 个千兆 LAN 口", "power": "12V / 2A", "warranty": "36 个月", "color": "red", "ports": 3},
    ]
    def header(c, title, page, version):
        c.setFillColor(colors.HexColor("#17354c"))
        c.rect(0, 760, 595, 82, fill=1, stroke=0)
        c.setFillColor(colors.white)
        c.setFont("DemoChinese", 19)
        c.drawString(35, 799, title)
        c.setFont("DemoChinese", 10)
        c.drawString(35, 778, "DemoMall 模拟商品资料 | 无实际商品或交易效力")
        c.setFillColor(colors.HexColor("#49586a"))
        c.setFont("DemoChinese", 9)
        c.drawString(35, 24, f"版本 {version} | PDF 第 {page} 页 | 合成功能测试数据")
    def line(c, text, y, size=13):
        c.setFillColor(colors.HexColor("#182f45"))
        c.setFont("DemoChinese", size)
        c.drawString(40, y, text)
    def table(c, values, top):
        for i, (key, value) in enumerate(values):
            y = top - i * 35
            c.setFillColor(colors.HexColor("#edf3f6") if i % 2 == 0 else colors.white)
            c.rect(35, y - 24, 525, 34, fill=1, stroke=0)
            line(c, key, y - 10, 12)
            c.drawString(245, y - 10, value)
    for product in products:
        path = pdf_root / f"{product['doc_id']}.pdf"
        c = canvas.Canvas(str(path), pagesize=(595, 842), invariant=1)
        header(c, product["sku"] + " 商品与接线说明", 1, product["version"])
        line(c, "一、规格与售后参数", 725, 15)
        table(c, [("无线规格", product["wifi"]), ("WAN 口最高速率", product["wan"]), ("LAN 接口", product["lan"]), ("电源适配器", product["power"]), ("PoE 供电", "不支持"), ("演示保修期", product["warranty"])], 690)
        line(c, "二、机身后侧接口图", 440, 15)
        c.setStrokeColor(colors.HexColor("#748da0"))
        c.setFillColor(colors.HexColor("#f0f4f7"))
        c.roundRect(45, 260, 505, 135, 12, fill=1, stroke=1)
        c.setFillColor(colors.black)
        c.circle(80, 330, 13, fill=1, stroke=0)
        c.setFont("DemoChinese", 11)
        c.drawString(66, 300, "电源")
        for i, label in enumerate(["WAN"] + [f"LAN{x+1}" for x in range(product["ports"])]):
            x = 140 + i * 88
            c.setFillColor(colors.blue if i == 0 and product["color"] == "blue" else colors.red if i == 0 else colors.HexColor("#555c68"))
            c.rect(x, 315, 65, 42, fill=1, stroke=0)
            c.setFillColor(colors.HexColor("#172e42"))
            c.drawCentredString(x + 32, 295, label)
        line(c, "连接方法：光猫的网线接 WAN；电脑网线接 LAN。", 220, 12)
        line(c, "接口颜色请以本页图示为准。", 195, 12)
        c.showPage()
        header(c, product["sku"] + " 故障排查与复位", 2, product["version"])
        line(c, "三、状态指示灯", 725, 15)
        table(c, [("绿色常亮", "网络连接正常"), ("橙色闪烁", "未连接互联网"), ("指示灯熄灭", "请检查电源适配器")], 680)
        line(c, "四、恢复出厂设置", 520, 15)
        line(c, "保持通电，按住 RESET 按钮 10 秒后松开。", 480)
        line(c, "恢复出厂设置会清除已有网络配置。", 445)
        line(c, "五、资料适用范围", 355, 15)
        line(c, f"适用型号：{product['sku']}；版本：{product['version']}。", 315)
        line(c, "商品参数不包含实时售价、库存、订单或物流信息。", 280, 12)
        line(c, "本文件未报告防水等级，不应根据常识补充。", 245, 12)
        c.save()
        records.append({"doc_id": product["doc_id"], "title": f"{product['sku']} 手册 {product['version']}", "pdf": str(path.relative_to(WORKSPACE)), "sha256": file_hash(path), "sku": product["sku"], "store_id": "demo_store", "doc_type": "product_manual", "version": product["version"], "valid_from": product["valid_from"], "valid_to": product["valid_to"], "source_type": "synthetic_demo", "category": "router"})
    path = pdf_root / "demo-store-policy-2026.pdf"
    c = canvas.Canvas(str(path), pagesize=(595, 842), invariant=1)
    header(c, "DemoMall 模拟服务规则", 1, "2026.1")
    line(c, "此为虚构商店的功能测试规则，不适用于真实交易。", 710, 13)
    line(c, "演示退货申请期限", 640, 16)
    line(c, "未启用商品可在签收后 10 天内提交演示退货申请。", 595, 13)
    line(c, "商品保修期", 520, 16)
    line(c, "保修期按对应型号与生效版本的商品手册执行。", 475, 13)
    line(c, "实时业务信息", 400, 16)
    line(c, "售价、库存、订单状态及物流进度需查询业务服务。", 355, 13)
    line(c, "知识库不能确认订单已退款、商品有货或包裹到达。", 320, 13)
    c.showPage()
    header(c, "DemoMall 服务问题分类", 2, "2026.1")
    table(c, [("商品参数与使用说明", "查询对应型号手册"), ("未明确商品型号", "先询问用户具体型号"), ("实时价格与库存", "需要业务查询服务"), ("订单与物流", "需要身份校验及业务服务"), ("资料未报告的功能", "明确说明无可用证据")], 700)
    c.save()
    records.append({"doc_id": "demo-store-policy-2026", "title": "DemoMall 模拟服务规则 2026.1", "pdf": str(path.relative_to(WORKSPACE)), "sha256": file_hash(path), "sku": "*", "store_id": "demo_store", "doc_type": "store_policy", "version": "2026.1", "valid_from": "2026-01-01", "valid_to": None, "source_type": "synthetic_demo", "category": "router"})
    write_json(root / "documents.json", records)
    raw = [
        ("DN-R100", "DN-R100 有几个 LAN 接口？", "2 个", ["dn-r100-2026:p0001"], ["2", "2 个", "两个"], "spec", "answer", None),
        ("DN-R100", "DN-R100 的电源适配器规格是多少？", "12V / 1A", ["dn-r100-2026:p0001"], ["12V / 1A", "12V 1A"], "spec", "answer", None),
        ("DN-R100", "请看接口图，DN-R100 接光猫的 WAN 接口是什么颜色？", "蓝色", ["dn-r100-2026:p0001"], ["蓝色", "蓝"], "visual", "answer", None),
        ("DN-R100", "DN-R100 橙色指示灯闪烁表示什么？", "未连接互联网", ["dn-r100-2026:p0002"], ["未连接互联网", "没有连接互联网"], "troubleshooting", "answer", None),
        ("DN-R100", "DN-R100 当前版本的演示保修期是多少个月？", "24 个月", ["dn-r100-2026:p0001"], ["24", "24 个月", "24个月"], "version", "answer", None),
        ("DN-R200", "DN-R200 的演示保修期是多少个月？", "36 个月", ["dn-r200-2026:p0001"], ["36", "36 个月", "36个月"], "spec", "answer", None),
        (["DN-R100", "DN-R200"], "对比 DN-R100 和 DN-R200，WAN 口最高速率分别是多少？", "DN-R100 为 1000 Mbps，DN-R200 为 2500 Mbps", ["dn-r100-2026:p0001", "dn-r200-2026:p0001"], ["DN-R100: 1000 Mbps; DN-R200: 2500 Mbps"], "comparison", "answer", None),
        (None, "DemoMall 模拟服务规则中，未启用商品签收后几天内可以提交退货申请？", "10 天", ["demo-store-policy-2026:p0001"], ["10", "10 天", "10天"], "policy", "answer", None),
        ("DN-R100", "DN-R100 通电时按住 RESET 多久可以恢复出厂设置？", "10 秒", ["dn-r100-2026:p0002"], ["10", "10 秒", "10秒"], "troubleshooting", "answer", None),
        ("DN-R200", "DN-R200 是否支持 PoE 供电？", "不支持", ["dn-r200-2026:p0001"], ["不支持", "不支持 PoE 供电"], "spec", "answer", None),
        ("DN-R100", "DN-R100 有 IP68 防水等级吗？", "NOT_FOUND", [], ["NOT_FOUND"], "unsupported", "not_found", None),
        (None, "我的订单现在到哪里了？", "当前 RAG 未接入订单与物流查询服务", [], ["当前 RAG 未接入订单与物流查询服务"], "dynamic", "requires_tool", None),
        ("DN-R100", "DN-R100 现在卖多少钱？", "当前 RAG 未接入实时价格与库存查询服务", [], ["当前 RAG 未接入实时价格与库存查询服务"], "dynamic", "requires_tool", None),
        (None, "这款路由器有几个 LAN 口？", "请提供具体商品型号", [], ["请提供具体商品型号"], "missing_sku", "clarify", None),
        ("DN-R900", "DN-R900 的 WAN 口速率是多少？", "知识库没有该商品型号的有效资料", [], ["知识库没有该商品型号的有效资料"], "unknown_sku", "not_found", None),
        ("DN-R100", "DN-R100 在 2025 年版本的演示保修期是多少个月？", "12 个月", ["dn-r100-2025:p0001"], ["12", "12 个月", "12个月"], "historical_version", "answer", "2025-06-01"),
    ]
    questions = []
    for i, (sku, query, answer, gold, aliases, kind, action, date) in enumerate(raw, start=1):
        request = {"store_id": "demo_store", "as_of": date or "2026-11-01"}
        if sku:
            request["skus"] = sku if isinstance(sku, list) else [sku]
        questions.append({"question_id": f"ecom-{i:02d}", "question": query, "reference_answer": answer, "answer_aliases": aliases,
                          "reference_pages": gold, "document_ids": sorted({p.split(":p")[0] for p in gold}), "split": "smoke_dev",
                          "question_type": kind, "expected_action": action, "request": request,
                          "gold_origin": "manually_authored_against_explicit_synthetic_demo", "page_numbering": "one_based_pdf_page"})
    write_jsonl(root / "questions.jsonl", questions)
    write_json(root / "protocol.json", {"source_type": "synthetic_demo", "not_real_customer_data": True, "purpose": "Functional smoke checks only; not a held-out business benchmark.", "questions": len(questions), "documents": len(records), "request_policy": "SKU/store/as_of are user request context, never inferred from gold evidence.", "all_questions_split": "smoke_dev", "future_business_test": "Real authorized product docs and independently reviewed questions required; group by model family/version and policy date."})
    operation("构建电商 RAG 模拟功能资料", "ReportLab 生成带参数表、接口颜色图和历史版本的 4 份中文 PDF；手工定义商品/店铺/有效期元数据与 16 道带动作标签的功能题。", "4 份 PDF、8 页、16 道 smoke_dev 功能题；全部标注为模拟资料，不冒充真实品牌、政策或客户数据。", [root / "documents.json", root / "questions.jsonl", root / "protocol.json"], "生成后需渲染检查图表与文字；SKU 上下文来自请求字段，不从参考页反推。", status="数据已生成，待渲染与运行验证")
    print(root)


if __name__ == "__main__":
    main()
