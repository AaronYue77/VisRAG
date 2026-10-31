"""Ten manually authored questions, checked against the exact downloaded PDF pages.

This is a development smoke suite, not an unbiased generalization benchmark.
English retrieval queries match the English documents; Chinese translations are for reading.
"""
from common import WORKSPACE, write_jsonl, write_json
from core import page_id

ITEMS = [
    ("attention", "What is the output dimension d_model of the Transformer sublayers and embeddings in Section 3.1?", "第 3.1 节中，Transformer 子层和嵌入的输出维度 d_model 是多少？", "512", [3], ["512", "512 dimensions"], "text"),
    ("attention", "How many parallel attention heads h are used in Section 3.2.2 of the Transformer paper?", "第 3.2.2 节使用多少个并行注意力头？", "8", [5], ["8", "8 heads", "eight"], "text"),
    ("attention", "In Section 3.3, what is the inner-layer dimensionality d_ff of the position-wise feed-forward network?", "第 3.3 节前馈网络内部维度 d_ff 是多少？", "2048", [5], ["2048", "2,048"], "text"),
    ("attention", "How many warmup_steps are used in the Transformer learning rate schedule in Section 5.3?", "第 5.3 节学习率预热步数是多少？", "4000", [7], ["4000", "4,000", "4000 steps"], "text"),
    ("attention", "In Table 3, how many million parameters does the Transformer base model have?", "表 3 中 Transformer base 模型参数量是多少百万？", "65", [9], ["65", "65 million", "65M"], "table"),
    ("clip", "In Section 2.1 of the CLIP paper, how many million image-text pairs are collected for WIT?", "CLIP 论文第 2.1 节中，WIT 收集了多少百万图文对？", "400", [2], ["400", "400 million", "400M"], "text"),
    ("clip", "In Table 1, what ImageNet zero-shot classification accuracy does CLIP achieve?", "表 1 中，CLIP 在 ImageNet 上的零样本分类准确率是多少？", "76.2%", [4], ["76.2", "76.2%", "76.2 percent"], "table"),
    ("clip", "In Section 2.4, what higher pixel resolution is used for the additional epoch of ViT-L/14 pre-training?", "第 2.4 节中，ViT-L/14 额外一轮预训练使用多大的像素分辨率？", "336", [4], ["336", "336 pixels", "336px", "336 x 336", "336×336"], "text"),
    ("clip", "In Section 2.4, how many days did the largest ResNet RN50x64 take to train?", "第 2.4 节中，最大的 ResNet RN50x64 训练用了多少天？", "18", [4], ["18", "18 days"], "text"),
    ("clip", "In Section 2.4, how many V100 GPUs were used to train the largest ResNet RN50x64?", "第 2.4 节中，最大的 ResNet RN50x64 使用了多少张 V100？", "592", [4], ["592", "592 GPUs", "592 V100 GPUs"], "text"),
]


def main():
    rows = []
    for index, (doc, query, query_zh, answer, pages, aliases, kind) in enumerate(ITEMS, start=1):
        rows.append({"question_id": f"smoke-{index:02d}", "question": query, "question_zh": query_zh,
                     "reference_answer": answer, "answer_aliases": aliases,
                     "reference_pages": [page_id(doc, page) for page in pages],
                     "document_ids": [doc], "split": "smoke_dev", "question_type": kind,
                     "gold_origin": "manually_authored_and_checked_against_downloaded_pdf", "page_numbering": "one_based_pdf_page"})
    destination = WORKSPACE / "data/a_baseline/smoke/questions.jsonl"
    write_jsonl(destination, rows)
    write_json(WORKSPACE / "data/a_baseline/smoke/protocol.json", {
        "purpose": "Development smoke test only; do not report as held-out benchmark performance.",
        "questions": 10, "documents": 2, "page_numbering": "PDF physical page, one-based",
        "language": "English queries with Chinese reading translations",
        "gold_review": "Numeric answers and supporting pages checked against these specific PDF files.",
    })
    print(destination)


if __name__ == "__main__":
    main()
