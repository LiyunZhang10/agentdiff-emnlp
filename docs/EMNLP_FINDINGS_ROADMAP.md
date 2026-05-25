## EMNLP 2026 Findings 投稿路线图（路径 2 完整版）

**当前论文版本**：`paper.md` (323 行 / 6,443 词 / 46.9 KB)
**导出产物**：`_export_v3/Paper_EN.docx` (518 KB), `_export_v3/Paper_EN.pdf` (624 KB)
**生成时间**：2026-05-18 18:27 +08:00

---

## 1. 路径 2 完成情况清单

### 1.1 已完成的新实验（直接回应 reviewer 攻击点）

| Track | 内容 | 结果文件 | 关键数字 |
|---|---|---|---|
| C | K=3 family-level wild cluster bootstrap | `track_c/wild_cluster_bootstrap_family.json` | topology p=0.241 (vs K=6 的 0.165) |
| C | K=2 qwen-vs-other 极端检查 | `track_c/wild_cluster_bootstrap_qwen_vs_other.json` | β_topo=+4.02 |
| C | non-qwen 24-cell paired-t | `track_c/headline_non_qwen_only.json` | **+5.54 pp, t=4.41, p=0.0002, 19/24 正** |
| D | SBERT (nomic-embed-text 768-d) embedding 距离 + 三 proxy severity audit | `track_d/severity_with_embeddings.jsonl` (8350 行)<br>`track_d/severity_matched_delta_4proxies.json` | 4 proxy 全部 +13.7~+15.4 pp，p<0.0001 |

### 1.2 论文修改清单

| 段落 | 修改 |
|---|---|
| Abstract | 加入 4-proxy severity 鲁棒性 / non-qwen 5.54pp / K=3 family p=0.241 |
| §1 Contribution 1 | 加入 SBERT cosine 鲁棒性陈述 |
| §1 Contribution 2 | 同时报告 K=6 与 K=3 family-cluster bootstrap |
| §2 Related Work | 加入 Mirzadeh GSM-Symbolic / Lanham faithfulness / Turpin 不诚实 CoT 段落（防 R1-Fatal-4） |
| §4.1 | 加入 Table 1b（4-proxy severity-matched Δ） |
| **§4.8 (新增)** | Family-level wild bootstrap (K=3) + qwen-vs-other (K=2) + non-qwen-only paired-t |
| §6 L2 | 升级为 K=6 与 K=3 双 cluster 都 fail |
| §6 L7 (新增) | Effect-size halving outside qwen |
| §7 Conclusion | 整段重写：4-proxy 鲁棒 / non-qwen 仍显著 / family count 边界 |
| References | 加入 Lanham 2023 / Mirzadeh 2024 / Razeghi 2022 / Turpin 2023 / Wang 2023 |

### 1.3 文本质量清理
- 21 处 `\u20XX` 字面 escape → 真实 Unicode 字符（– — " " ž 等）
- 25 处 `\S4.X` LaTeX 写法 → 真实 § 字符
- 9632 字节级别的 `$\\Delta$ / $\\kappa$ / $\\rho$` 双反斜杠 math → 单反斜杠

---

## 2. 直接回应 review 报告攻击点

### Reviewer 1 攻击 → 我们的反驳证据
| Attack | 反驳 |
|---|---|
| Fatal 1: 14.32pp 不是惊人现象 | 4 proxy 都给出 +13.7~+15.4 pp，p<0.0001（Table 1b）— phenomenon 不依赖度量选择 |
| Fatal 2: topology 反复 retract | 已重写 conclusion 和 contributions，不再 claim topology |
| Fatal 3: K=6 family≈3 fatal | §4.8 直接报 K=3，p=0.241，明确承认 |
| Fatal 4: 关键 2024 文献未引 | Mirzadeh / Lanham / Turpin / Wang 已加入 §2 + Refs |
| Fatal 5: tool = trivial baseline | §5/§6 L5 已明确"prototype, not deployable" |
| Fatal 6: edit distance 不是 severity | §4.1 Table 1b 用 SBERT cosine 等 4 proxy 直接反驳 |

### Reviewer 2 攻击 → 反驳/承认情况
| Attack | 处理 |
|---|---|
| Fatal 1: K=6 cluster 显著性虚高 | §4.8 报 K=3 p=0.241，论文不再 claim |
| Fatal 2: capability claim 张力 | conclusion 改写：accuracy 不再 claim 显著 |
| Fatal 3: pooled trajectory inflated | hierarchical bootstrap 已在 §4.3 |
| Fatal 4: judge calibration | §4.6 已分 stratum 报 κ |
| Fatal 5: multiple comparison 未控 | §4.2 / Table 3 已 BH FDR |

### Reviewer 3 攻击 → 反驳/承认情况
| Attack | 处理 |
|---|---|
| Fatal 1: AgentDiff prototype 增益弱 | §6 L5 明确"sign acc ties trivial mean"，conclusion 不再吹 |
| Fatal 2: classifier 未验证 | §5 已降级为 prototype |
| Fatal 3: 文献 2024 不够新 | 5 条 2022-2024 文献已加 |
| Fatal 4: 无 deployment validation | conclusion 改为"contribution is measurement, not method" |
| Issue 2: qwen-family 占 sample 偏差 | §4.8 直接报 family 分布 + non-qwen 5.54pp |

---

## 3. 仍然存在的弱点（不能在 5-7 天内完全闭环）

| Issue | 影响 | 处置 |
|---|---|---|
| **Family count = 3** | reviewer 仍可说 "3 family 不够" | 论文已诚实报告 + L7 给出 non-qwen 5.54pp，让 reviewer 自己判断 |
| **Cascade depth 在 HotpotQA 反向** | mechanism claim 弱 | conclusion 不 claim mechanism，仅描述现象 + GSM8K-only 说明 |
| **AgentDiff-Probe v2 sign acc 平手 trivial mean** | tool contribution 几乎为零 | §5/§6 L5 明确为 prototype，conclusion 不放大 |
| **Generator-source ranking 在 cross-family 崩** | 单 cell Δ 数值不可移植 | §4.7 + §6 L3 直接说 |

---

## 4. EMNLP Findings 接收概率分析（基于路径 2 完成度）

### 4.1 我估计的接收概率：**45-55%**

依据：
- Findings 接受标准约为 **"valid empirical contribution + honest limitations + 2-3 reviewer 中至少 1 个偏正"**
- 我们已经把 R1-R3 的 13 个 fatal issue 中的 **9 个**直接闭环（Table 1b 的 SBERT 鲁棒、§4.8 的 family-cluster + non-qwen、refs 5 条新增、tool 降级、conclusion 重写）
- 剩 4 个 fatal 是结构性的（family count、cascade reverse、tool 弱、generator 崩），但论文每条都有 limitation 段落显式承认

### 4.2 主会（main track）接收概率：**15-25%**
不推荐投 main，原因仍是 family count 不足。

### 4.3 推荐投稿顺序
1. **第一优先**：EMNLP 2026 Findings（截稿 2026 年 6 月左右）
2. **次选**：ACL 2026 Findings 或 ACL 2026 Workshop on Trustworthy LLMs
3. **保底**：NAACL 2026 Findings

---

## 5. 后续可选加强（如还有 1-2 周时间）

| 加强 | 时间 | 增益 |
|---|---|---|
| 在 GSM-Symbolic 模板上重跑 8 cells，与 Mirzadeh 数字直接并列 | 2 天 | +5-10% 接收概率 |
| 跑 1 个新 family（mistral-7B 或 phi-3-mini），让 family count 从 3 → 4 | 2 天 | +5-10% 接收概率 |
| 用 Lanham-style truncation intervention 替代 cascade depth | 3 天 | +5% 接收概率（机制 claim 增强） |
| 给 AgentDiff-Probe v2 加一个真正超过 trivial mean 的判别器（或干脆删除整章） | 1 天 | +2% 接收概率 |

---

## 6. 投稿前检查清单

- [x] 4 个 reviewer 的 13 个 fatal issue，9 个有直接证据反驳
- [x] 关键 2024 文献全部引用（Mirzadeh / Lanham / Turpin）
- [x] Severity 度量不依赖单一定义（4 proxy 全部稳）
- [x] Family count 边界诚实承认（§4.8 + L7）
- [x] Tool 部分 self-downgrade 为 prototype（不再吹）
- [x] Limitations 1-7 完整列出
- [x] Conclusion 改为 measurement contribution，不 claim mechanism
- [x] docx + pdf 同时输出（518 KB / 624 KB）
- [ ] **TODO（用户决定）**：是否要重画 figures（fig1-5 已存在 `paper_figs/`，确认无 stale）
- [ ] **TODO（用户决定）**：是否要 anonymize 作者信息
- [ ] **TODO（用户决定）**：是否要把 paper_acl/main.tex 也同步更新

---

## 7. 文件位置索引

| 类别 | 路径 |
|---|---|
| 主稿 markdown | `paper.md` |
| Word | `_export_v3/Paper_EN.docx` |
| PDF | `_export_v3/Paper_EN.pdf` |
| 旧版 (含早期 K=6 only) | `_export_v2/` |
| Track A 原始数据 (8350 row severity) | `track_a/severity_per_variant.jsonl` |
| Track B K=6 wild bootstrap | `track_b/wild_cluster_bootstrap.json` |
| Track C K=3 family bootstrap | `track_c/wild_cluster_bootstrap_family.json` |
| Track C non-qwen | `track_c/headline_non_qwen_only.json` |
| Track D SBERT severity | `track_d/severity_with_embeddings.jsonl` (8350 行) |
| Track D 4-proxy matched Δ | `track_d/severity_matched_delta_4proxies.json` |
| 实验脚本 (新) | `track_c_family_cluster.py`, `track_d_embedding_severity.py` |
| BibTeX | `paper_acl/agentdiff.bib` (已加 Lanham/Mirzadeh/Turpin/Wang/Razeghi/HELM/BBH) |

**所有数据可重现**：脚本 deterministic seed 42 / nomic-embed-text 调用本地 ollama。

---

## 8. 一句话总结

**路径 2 已完成**：4 个 reviewer 共识中的 9 个 fatal issue 通过新增 SBERT severity audit (Track D) 和 K=3 family-cluster bootstrap + non-qwen 24-cell paired-t (Track C) 直接闭环；剩余 4 个结构性弱点全部进入 limitations 显式承认。论文以 EMNLP 2026 Findings 为目标，估计接收概率 45-55%。
