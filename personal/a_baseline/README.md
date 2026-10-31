# 图文文档问答助手：A 基线

个人开发分支：`feat/a-baseline`。上游基准分支：`baseline/upstream-2026-10-31`。
Fork：[AaronYue77/VisRAG](https://github.com/AaronYue77/VisRAG)。上游许可证与引用说明保留，新增代码沿用仓库 Apache-2.0 许可证。

本次实现 **PDF 文本/OCR → BM25 Top-2 页面检索 → 页面图片 → Qwen3.5-2B 回答与引用 → JSONL 评测**。
它是可测量的 A 基线；B 的视觉检索、C 的微调尚未实现。本机 8GB 显存不同时加载两个大模型。

## 先看什么

1. [代码阅读地图](docs/代码阅读地图.md)：对照上游代码追踪一次问答。
2. [评测协议与字段](docs/评测协议与记录格式.md)：弄清楚分区、指标、数据泄漏与版本记录。
3. [问题与解决记录 TXT](docs/问题与解决记录.txt)：真实失败、处理方法、验证与未解决项。
4. 工作区 `reports/a_baseline/smoke10-a-v2/运行报告.md`：逐题答案、检索页与实际输入图片。
5. [操作方法与结果记录 TXT](docs/操作方法与结果记录.txt)：每次操作的方法、结果、产物、验证与问题关联。
6. 工作区 `记录/逐题结果汇总.txt`：所有实验的逐题 TXT；每个运行目录还有独立 `逐题结果.txt`。按批次保留同一题的不同版本，不覆盖旧结果。

## 本机运行

以下在 `E:\大模型项目自学E\doc-vqa-assistant` 的 PowerShell 中运行；不要求激活虚拟环境。

```powershell
# 已备好数据时不必重复下载。无 GPU 的评测逻辑检查：
.\.venv\Scripts\python.exe .\upstream\VisRAG\personal\a_baseline\test_core.py

# 生成固定 10 题，建立 27 页语料：
.\.venv\Scripts\python.exe .\upstream\VisRAG\personal\a_baseline\smoke_questions.py
.\.venv\Scripts\python.exe .\upstream\VisRAG\personal\a_baseline\run.py index --suite smoke

# 新运行用新名字，避免覆盖已有实验：
.\.venv\Scripts\python.exe .\upstream\VisRAG\personal\a_baseline\run.py run --run-name smoke10-my-run

# 公开开发集 50 题，先只检查检索：
.\.venv\Scripts\python.exe .\upstream\VisRAG\personal\a_baseline\run.py run --suite vidoseek --split dev --run-name dev50-my-retrieval --retrieval-only

# 完整问答版本：
.\.venv\Scripts\python.exe .\upstream\VisRAG\personal\a_baseline\run.py run --suite vidoseek --split dev --run-name dev50-my-vlm
```

结果写到工作区 `reports/a_baseline/<run-name>/`。每题落盘；同代码同配置重跑相同名字可继续未完成的题；配置或代码变更必须换名字。
模型位置 `models/Qwen3.5-2B/`。数据位置 `data/a_baseline/`。所有 HF/Torch/临时缓存由入口固定到 E 盘工作区。
若把本仓库克隆到其他结构，先显式设置 `$env:DOC_VQA_WORKSPACE = '你的数据与模型目录'`。

需要重新准备数据时运行 `download_data.py`，再运行 `smoke_questions.py`；公开子集使用 `benchmark_data.py`，然后 `run.py index --suite vidoseek`。
下载器保留证书校验、校验大文件哈希；本机若 Python 3.14 的 HTTPS 重置，可用已有 3.12 执行这两个纯标准库下载入口。推理仍使用项目 `.venv`。

最终测试集现在不用于调参。A/B/C 方案确定后，以新名字显式传入 `--split test --final-test`，再报告最终结果。

## 已验证的范围

- 两份技术论文共 27 页；人工题 10 道，参考答案与物理 PDF 证据页固定。
- 两轮 10 题完整执行成功。v1 的格式提示不稳定；v2 只补充合法 JSON 示例与数组格式要求。
- v2：Hit@2=9/10，MRR@2=0.80，normalized EM=9/10，JSON 解析=10/10；平均生成约 2.01 秒，峰值已分配显存约 4.42 GiB。
- `smoke-05` 未召回参考证据页，v2 返回 NOT_FOUND；没有向模型补 gold 页面，也没有删除失败题。
- 这些是开发冒烟数据，题目多为短数字答案，不能当作真实业务准确率或泛化结论。
- 10 个 CPU 评测保护测试通过，覆盖文档泄漏、错误页码、指标区别、引用边界、PDF 特殊字符读写和 TXT 内容完整性。
- 公开子集 100 题、79 份文档、1560 页已经建库；99 页使用 OCR 回退。按文档分成 dev/test 各 50 题，页码与分区检查通过。3 页未提取到文字仍保留，未自动认定为真正空白页。
- 50 道公开开发题完整运行通过：Hit@2=0.64，MRR@2=0.61，normalized EM=0.36，token F1≈0.478，JSON 解析=1.0；平均生成约 3.00 秒、峰值显存约 4.43 GiB。50 道最终测试题未运行。长答案的 EM 可能低估等价表达，需要结合逐题报告复核。

依赖锁定见 `requirements.local.lock.txt`（本机 Windows 环境），模型配置与数据哈希见每次运行 `config.json`。部署到其他平台时需要独立解析适配 wheel，不直接复用 Windows 二进制环境。

## 自动记录规则

`run.py` 自动记录命令开始、完成/失败；每道题先写 JSONL 并刷新，再追加逐题 TXT；批次结束重建汇总 TXT。下载、抽样、题目生成也会记录方法和结果。`问题与解决记录.txt` 保留异常，修复后应追加同编号的验证结论。Codex 手动排查/改代码/Git 同步仍需按工作区 `AGENTS.md` 补记，不会自动监控项目之外的软件操作。

需要重新导出历史 TXT：`python personal/a_baseline/audit.py`（从仓库目录运行，使用项目 Python）。只新增/重建派生 TXT，不修改原始 JSONL、配置或模型输出。

## 你后面应亲自完成的改进

先解释 smoke-05 为什么错，再用相同评测协议比较 A/B；保持生成模型与输入预算不变。B 再加入页面视觉检索与融合，不先堆多 Agent。C 只在已定位的错误和独立训练数据上做微调。每次改动都保留旧结果、原始输出和配置，证明收益来自哪里。
