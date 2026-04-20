# PharmaCore 推广文案

---

## Reddit r/MachineLearning (英文)

**Title:** [P] PharmaCore — AI drug discovery that runs entirely on a MacBook (Apple Silicon, no cloud)

**Body:**

I built an AI drug discovery platform that runs 100% locally on Apple Silicon. No cloud GPUs, no API keys, no data leaves your machine.

**What it does:**

- **De novo drug discovery** — generates novel drug candidates for any protein target (~7s for 5 molecules on M4)
- **Drug repurposing** — screens FDA-approved drugs for new therapeutic uses (~18s for 12-drug screen)
- **Full audit trail** — every computation step is logged for regulatory transparency

**How it works:**

Uses sparse (50% pruned) protein and molecular language models:
- ESM-2 (Meta) for protein target encoding
- ChemBERTa for molecular/drug encoding
- RDKit for cheminformatics (fingerprints, Lipinski rules, QED)
- PyTorch MPS for Apple Silicon GPU acceleration

The sparse models retain 97%+ quality while being significantly more efficient on consumer hardware.

**Quick validation:** When screening for EGFR inhibitors, the system correctly identifies Erlotinib (a known EGFR inhibitor) as the top repurposing candidate.

**Links:**
- GitHub: https://github.com/reacherwu/PharmaCore
- HuggingFace Models: https://huggingface.co/collections/stephenjun8192/pharmacore-sparse-models
- Live Demo: https://huggingface.co/spaces/stephenjun8192/PharmaCore

MIT licensed. Feedback welcome — especially from anyone in computational chemistry or pharma.

---

## Reddit r/bioinformatics (英文)

**Title:** [Tool] Open-source AI drug discovery platform — runs locally on Apple Silicon, no cloud needed

**Body:**

Sharing a tool I built for the computational biology community: PharmaCore is an AI-powered drug discovery platform that runs entirely on a Mac mini/MacBook with Apple Silicon.

**Why I built this:**

Most AI drug discovery tools require cloud GPU clusters or expensive proprietary platforms. Academic labs and small biotech teams often can't afford these, and sending proprietary compound data to external APIs is a non-starter for many organizations.

**Capabilities:**

1. **De novo molecular generation** — given a protein target (name + optional sequence), generates novel drug candidates scored by drug-likeness (QED), target compatibility, and synthetic accessibility
2. **Drug repurposing** — screens a database of FDA-approved drugs against new targets using protein-drug compatibility scoring
3. **Transparent audit pipeline** — full reproducibility with JSON audit trails

**Technical details:**
- Sparse ESM-2 (50% pruned) for protein embeddings
- Sparse ChemBERTa for molecular embeddings
- Scaffold-based enumeration with RDKit
- All inference on Apple MPS (Metal Performance Shaders)
- Sub-20ms protein inference, sub-5ms molecular inference

**Limitations (being honest):**
- This is a computational screening tool, not a replacement for wet lab validation
- Drug-target scoring is embedding-based, not physics-based docking
- Database of 12 FDA-approved drugs (expandable)

GitHub: https://github.com/reacherwu/PharmaCore
MIT license. PRs welcome.

---

## 知乎文章 (中文)

**标题：** 我在 Mac mini 上搭了一个 AI 新药发现平台——完全本地运行，不需要云 GPU

**正文：**

## 背景

新药研发平均耗资 26 亿美元，周期超过 10 年。AI 制药是近年来最热门的方向之一，但大多数 AI 制药工具要么需要昂贵的云 GPU 集群，要么依赖商业 API，对于学术实验室和小型生物技术公司来说门槛太高。

更关键的是——把化合物数据和靶点信息发送到外部服务器，对很多机构来说是不可接受的。

## PharmaCore 是什么

PharmaCore 是一个完全运行在 Apple Silicon（M1/M2/M3/M4）上的 AI 药物发现平台。两个核心能力：

**1. 全新药物生成（De Novo Discovery）**

给定一个蛋白质靶点，AI 自动生成新的候选药物分子。在 M4 Mac mini 上，生成 5 个候选分子只需约 7 秒。

系统会对每个分子进行多维度评分：
- 类药性（QED）
- 靶点兼容性
- 合成可及性
- Lipinski/Veber 规则

**2. 老药新用（Drug Repurposing）**

筛选已上市的 FDA 批准药物，寻找对新靶点的潜在治疗作用。验证结果：对 EGFR 靶点筛选时，系统正确地将 Erlotinib（已知的 EGFR 抑制剂）排在第一位。

## 技术实现

- **稀疏模型**：对 ESM-2（蛋白质语言模型）和 ChemBERTa（分子语言模型）进行 50% 幅度剪枝，质量保留 97%+
- **Apple MPS 加速**：利用 Metal Performance Shaders 进行 GPU 推理
- **RDKit**：化学信息学计算（分子指纹、描述符、SMILES 处理）
- **审计管线**：每一步计算都有完整的 JSON 审计日志，满足可追溯性要求

## 性能数据（M4 Mac mini, 16GB）

| 任务 | 耗时 |
|------|------|
| 蛋白质嵌入（160aa） | ~8ms |
| 分子嵌入 | ~5ms |
| 新药生成（5个分子） | ~7s |
| 老药新用筛选（12种药） | ~18s |

## 为什么选择本地运行

1. **数据隐私**：化合物结构和靶点信息不出本机
2. **零成本**：不需要 GPU 云服务器，一台 Mac mini 就够
3. **可审计**：所有计算步骤透明可追溯
4. **低延迟**：亚毫秒级推理，适合交互式探索

## 开源地址

- GitHub: https://github.com/reacherwu/PharmaCore
- HuggingFace 模型: https://huggingface.co/collections/stephenjun8192/pharmacore-sparse-models
- 在线 Demo: https://huggingface.co/spaces/stephenjun8192/PharmaCore

MIT 协议，欢迎 Star 和 PR。

如果你在做计算化学、药物设计相关的工作，欢迎交流。

---

## Hacker News (英文，简短)

**Title:** Show HN: PharmaCore – AI drug discovery running locally on Apple Silicon

**Body:**

PharmaCore is an open-source AI drug discovery platform that runs entirely on consumer Apple Silicon hardware (M1-M4 Macs).

Two capabilities:
- De novo drug generation: ~7s for 5 candidates
- Drug repurposing screen: ~18s for 12 FDA-approved drugs

Uses 50%-sparse ESM-2 and ChemBERTa models with 97%+ quality retention. All computation local — no cloud, no API keys, no data exfiltration.

https://github.com/reacherwu/PharmaCore

---

## V2EX (中文，简短)

**标题：** [开源] 在 Mac mini 上跑 AI 新药发现——PharmaCore，完全本地，不需要云 GPU

**正文：**

做了一个 AI 制药平台，跑在 Apple Silicon 上，不需要任何外部 API 或云服务。

核心功能：
- 全新药物分子生成（给靶点，出候选药物，7秒5个分子）
- 老药新用筛选（12种FDA药物 × 任意靶点，18秒出结果）
- 完整审计日志（每步计算可追溯）

技术栈：PyTorch MPS + 稀疏化 ESM-2/ChemBERTa + RDKit

GitHub: https://github.com/reacherwu/PharmaCore
MIT 协议，欢迎 Star。

---

## 掘金 (中文，技术向)

**标题：** 用 Apple Silicon 跑 AI 新药发现：稀疏模型 + 本地推理实战

**正文：**

（可以基于知乎版本，增加更多技术细节：稀疏化方法、MPS 加速原理、模型选型过程等）
