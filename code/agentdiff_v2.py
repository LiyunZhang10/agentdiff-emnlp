#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AgentDiff v2 — 真正的多步 ReAct Agent 系统

核心区别于 v1：
1. Agent 有真正的工具（search, calculate, lookup）
2. 每一步是独立的 LLM 调用（不是解析单次输出）
3. 执行轨迹记录每步的输入/输出
4. 支持 ReAct / CoT / Direct 三种 agent 类型
5. 失败传播图基于真实的步级数据
"""
import json
import re
import random
import time
import math
import logging
import hashlib
from collections import Counter, defaultdict

logger = logging.getLogger(__name__)


# ============================================================
# 工具定义 — Agent 可以调用的工具
# ============================================================

class AgentTools:
    """Agent 可用的工具集合"""

    @staticmethod
    def search(query):
        """模拟搜索工具 — 从上下文中检索相关信息"""
        # 在真实实验中，这个工具从 sample 的 context 中检索
        # 返回格式化的搜索结果
        return "[Search results for '%s' will be populated from context]" % query

    @staticmethod
    def calculate(expression):
        """计算工具 — 执行数学表达式"""
        try:
            # 安全的数学计算（只允许基本运算）
            allowed = set('0123456789+-*/.() ')
            expr_clean = expression.strip()
            if all(c in allowed for c in expr_clean):
                result = eval(expr_clean)
                return str(result)
            else:
                # 尝试提取数字和运算
                return "Cannot evaluate: %s" % expression
        except Exception as e:
            return "Calculation error: %s" % str(e)

    @staticmethod
    def lookup(entity):
        """查找工具 — 查找实体信息"""
        return "[Lookup results for '%s' will be populated from context]" % entity

    TOOL_DESCRIPTIONS = {
        "search": "search(query) - Search for information related to the query",
        "calculate": "calculate(expression) - Evaluate a mathematical expression",
        "lookup": "lookup(entity) - Look up information about a specific entity",
    }


# ============================================================
# ReAct Agent — 真正的多步推理 Agent
# ============================================================

class ReActAgent:
    """
    真正的 ReAct Agent：每一步是独立的 LLM 调用。
    
    执行流程：
    1. LLM 生成 Thought + Action
    2. 执行 Action，获得 Observation
    3. 将 Observation 反馈给 LLM
    4. 重复直到 LLM 输出 Final Answer
    """

    SYSTEM_PROMPT = """You are a helpful assistant that solves problems step by step using available tools.

Available tools:
{tool_descriptions}

You MUST follow this EXACT format for each step:
Thought: [your reasoning about what to do next]
Action: [tool_name(argument)]

After receiving an Observation, continue reasoning. When you have the final answer:
Thought: [final reasoning]
Final Answer: [your answer]

IMPORTANT RULES:
- You MUST use at least one tool before giving your final answer
- Your Final Answer must be SHORT and PRECISE (just the answer, no explanation)
- For questions asking 'who/what/where', give just the name/place/thing
- For math problems, give just the number
- You can take at most {max_steps} steps"""

    def __init__(self, llm_fn, max_steps=5, context_provider=None):
        """
        Args:
            llm_fn: callable(prompt) -> str
            max_steps: 最大推理步数
            context_provider: callable(query, context) -> str, 从上下文检索信息
        """
        self.llm_fn = llm_fn
        self.max_steps = max_steps
        self.context_provider = context_provider or self._default_context_provider
        self.tools = AgentTools()

    def _default_context_provider(self, query, context):
        """默认的上下文检索：简单的关键词匹配"""
        if not context:
            return "No context available."
        query_words = set(query.lower().split())
        sentences = [s.strip() for s in context.replace('\n', '. ').split('.') if s.strip()]
        scored = []
        for s in sentences:
            s_words = set(s.lower().split())
            overlap = len(query_words & s_words)
            scored.append((overlap, s))
        scored.sort(reverse=True)
        top = scored[:3]
        if top and top[0][0] > 0:
            return '. '.join([s for _, s in top]) + '.'
        return context[:500]

    def run(self, sample, extra_prompt=""):
        """
        运行 ReAct Agent，返回完整执行轨迹。
        
        Args:
            sample: dict with 'question'/'problem', 'context', 'answer'
            extra_prompt: 额外的提示（用于 patching）
        
        Returns:
            dict: {
                "sample_id": str,
                "agent_type": "react",
                "final_answer": str,
                "trace": [{"step": int, "input": str, "thought": str, 
                           "action": str, "observation": str, "raw_output": str}],
                "n_steps": int,
                "total_tokens_approx": int,
            }
        """
        question = sample.get("question", sample.get("problem", ""))
        context = sample.get("context", "")

        # 构建系统提示
        tool_desc = "\n".join("- %s" % d for d in AgentTools.TOOL_DESCRIPTIONS.values())
        system = self.SYSTEM_PROMPT.format(
            tool_descriptions=tool_desc,
            max_steps=self.max_steps
        )
        if extra_prompt:
            system += "\n\n" + extra_prompt

        # 构建初始 prompt
        if context:
            conversation = "%s\n\nContext: %s\n\nQuestion: %s" % (system, context, question)
        else:
            conversation = "%s\n\nProblem: %s" % (system, question)

        trace = []
        final_answer = None

        for step in range(1, self.max_steps + 1):
            # 每一步是独立的 LLM 调用
            step_input = conversation  # 记录当前输入
            try:
                raw_output = self.llm_fn(conversation)
            except Exception as e:
                trace.append({
                    "step": step,
                    "input_length": len(conversation),
                    "thought": "ERROR: %s" % str(e),
                    "action": None,
                    "observation": None,
                    "raw_output": "Error: %s" % str(e),
                })
                break

            # 解析 Thought 和 Action
            thought = self._extract_thought(raw_output)
            action = self._extract_action(raw_output)
            final = self._extract_final_answer(raw_output)

            if final:
                # Agent 给出了最终答案
                trace.append({
                    "step": step,
                    "input_length": len(conversation),
                    "thought": thought,
                    "action": None,
                    "observation": None,
                    "raw_output": raw_output[:1000],
                    "is_final": True,
                })
                final_answer = final
                break

            if action:
                # 执行工具调用
                observation = self._execute_action(action, context)
                trace.append({
                    "step": step,
                    "input_length": len(conversation),
                    "thought": thought,
                    "action": action,
                    "observation": observation,
                    "raw_output": raw_output[:1000],
                    "is_final": False,
                })
                # 将 observation 追加到对话中
                conversation += "\n" + raw_output.strip()
                conversation += "\nObservation: %s\n" % observation
            else:
                # 没有 action 也没有 final answer，尝试提取答案
                trace.append({
                    "step": step,
                    "input_length": len(conversation),
                    "thought": thought,
                    "action": None,
                    "observation": None,
                    "raw_output": raw_output[:1000],
                    "is_final": False,
                })
                # 尝试从输出中提取答案
                final_answer = self._extract_any_answer(raw_output)
                if final_answer:
                    break
                # 继续对话
                conversation += "\n" + raw_output.strip()
                conversation += "\nPlease continue. Use a tool or provide your Final Answer."

        if final_answer is None:
            # 从最后的输出中尽力提取
            if trace:
                final_answer = self._extract_any_answer(trace[-1].get("raw_output", ""))
            if final_answer is None:
                final_answer = "NO_ANSWER"

        return {
            "sample_id": sample.get("id", "unknown"),
            "agent_type": "react",
            "final_answer": final_answer,
            "trace": trace,
            "n_steps": len(trace),
            "total_tokens_approx": sum(t.get("input_length", 0) for t in trace),
        }

    def _extract_thought(self, output):
        m = re.search(r'Thought:\s*(.+?)(?:\n|Action:|Final Answer:|$)', output, re.DOTALL)
        return m.group(1).strip() if m else output[:200]

    def _extract_action(self, output):
        m = re.search(r'Action:\s*(.+?)(?:\n|$)', output)
        if m:
            action_str = m.group(1).strip()
            # 验证是合法的工具调用
            if re.match(r'(search|calculate|lookup)\s*\(', action_str):
                return action_str
        return None

    def _extract_final_answer(self, output):
        m = re.search(r'Final Answer:\s*(.+?)(?:\n|$)', output, re.IGNORECASE)
        return m.group(1).strip() if m else None

    def _extract_any_answer(self, output):
        """从任意输出中尽力提取答案"""
        patterns = [
            r'Final Answer:\s*(.+?)(?:\n|$)',
            r'[Tt]he answer is[:\s]+(.+?)(?:\n|$)',
            r'[Aa]nswer:\s*(.+?)(?:\n|$)',
            r'[Tt]herefore,?\s*(.+?)(?:\n|$)',
            r'= (\S+)\s*$',
            r'\\boxed\{(.+?)\}',
        ]
        for p in patterns:
            m = re.search(p, output, re.MULTILINE)
            if m:
                return m.group(1).strip().rstrip('.')
        # 最后一行
        lines = [l.strip() for l in output.split('\n') if l.strip()]
        return lines[-1][:200] if lines else None

    def _execute_action(self, action_str, context):
        """执行工具调用"""
        m = re.match(r'(\w+)\s*\((.+)\)', action_str)
        if not m:
            return "Invalid action format: %s" % action_str

        tool_name = m.group(1)
        arg = m.group(2).strip().strip('"').strip("'")

        if tool_name == "search":
            return self.context_provider(arg, context)
        elif tool_name == "calculate":
            return AgentTools.calculate(arg)
        elif tool_name == "lookup":
            return self.context_provider(arg, context)
        else:
            return "Unknown tool: %s" % tool_name


# ============================================================
# CoT Agent — Chain-of-Thought（单次调用但解析步骤）
# ============================================================

class CoTAgent:
    """Chain-of-Thought Agent：单次 LLM 调用，但解析推理步骤"""

    SYSTEM_PROMPT = """You are a helpful assistant. Solve the problem step by step.
Show your reasoning clearly, numbering each step.
After all steps, write your final answer as:
Final Answer: [answer]"""

    def __init__(self, llm_fn):
        self.llm_fn = llm_fn

    def run(self, sample, extra_prompt=""):
        question = sample.get("question", sample.get("problem", ""))
        context = sample.get("context", "")

        system = self.SYSTEM_PROMPT
        if extra_prompt:
            system += "\n\n" + extra_prompt

        if context:
            prompt = "%s\n\nContext: %s\n\nQuestion: %s" % (system, context, question)
        else:
            prompt = "%s\n\nProblem: %s" % (system, question)

        try:
            raw_output = self.llm_fn(prompt)
        except Exception as e:
            return {
                "sample_id": sample.get("id", "unknown"),
                "agent_type": "cot",
                "final_answer": "ERROR",
                "trace": [{"step": 1, "thought": "Error: %s" % str(e), "raw_output": str(e)}],
                "n_steps": 1,
                "total_tokens_approx": len(prompt),
            }

        # 解析步骤
        trace = self._parse_steps(raw_output)
        final_answer = self._extract_answer(raw_output)

        return {
            "sample_id": sample.get("id", "unknown"),
            "agent_type": "cot",
            "final_answer": final_answer or "NO_ANSWER",
            "trace": trace,
            "n_steps": len(trace),
            "total_tokens_approx": len(prompt) + len(raw_output),
        }

    def _parse_steps(self, output):
        """解析 CoT 输出为步骤"""
        steps = []
        # 尝试按 "Step N:" 或 "N." 或 "N)" 分割
        parts = re.split(r'(?:Step\s+\d+[:.)]|(?<=\n)\d+[.)]\s)', output)
        for i, part in enumerate(parts):
            if part.strip():
                steps.append({
                    "step": i + 1,
                    "thought": part.strip()[:500],
                    "action": None,
                    "observation": None,
                    "raw_output": part.strip()[:500],
                    "is_final": i == len(parts) - 1,
                })
        if not steps:
            steps = [{"step": 1, "thought": output[:500], "action": None,
                       "observation": None, "raw_output": output[:500], "is_final": True}]
        return steps

    def _extract_answer(self, output):
        patterns = [
            r'Final Answer:\s*(.+?)(?:\n|$)',
            r'[Tt]he answer is[:\s]+(.+?)(?:\n|$)',
            r'\\boxed\{(.+?)\}',
            r'[Aa]nswer:\s*(.+?)(?:\n|$)',
        ]
        for p in patterns:
            m = re.search(p, output, re.MULTILINE)
            if m:
                return m.group(1).strip().rstrip('.')
        lines = [l.strip() for l in output.split('\n') if l.strip()]
        return lines[-1][:200] if lines else None


# ============================================================
# Direct Agent — 单次直接回答（基线）
# ============================================================

class DirectAgent:
    """Direct Agent：直接回答，无推理过程（最弱基线）"""

    SYSTEM_PROMPT = "Answer the question directly and concisely. Give only the answer, nothing else."

    def __init__(self, llm_fn):
        self.llm_fn = llm_fn

    def run(self, sample, extra_prompt=""):
        question = sample.get("question", sample.get("problem", ""))
        context = sample.get("context", "")

        system = self.SYSTEM_PROMPT
        if extra_prompt:
            system += "\n\n" + extra_prompt

        if context:
            prompt = "%s\n\nContext: %s\n\nQuestion: %s\n\nAnswer:" % (system, context, question)
        else:
            prompt = "%s\n\nProblem: %s\n\nAnswer:" % (system, question)

        try:
            raw_output = self.llm_fn(prompt)
        except Exception as e:
            raw_output = "ERROR: %s" % str(e)

        answer = raw_output.strip().split('\n')[0].strip()

        return {
            "sample_id": sample.get("id", "unknown"),
            "agent_type": "direct",
            "final_answer": answer or "NO_ANSWER",
            "trace": [{"step": 1, "thought": answer, "action": None,
                        "observation": None, "raw_output": raw_output[:500], "is_final": True}],
            "n_steps": 1,
            "total_tokens_approx": len(prompt) + len(raw_output),
        }


# ============================================================
# Variant Generator v2 — 使用 LLM 生成高质量变体
# ============================================================

# Patterns that indicate the generator LLM refused / echoed the meta-prompt
# instead of producing a true variant. Empirically observed on Llama-3.1-8B.
BAD_VARIANT_PATTERNS = [
    re.compile(r"\bI'?m ready to help\b", re.I),
    re.compile(r"\bplease (go ahead|provide|share|send)\b", re.I),
    re.compile(r"\b(the )?rephrased (version|question)\b", re.I),
    re.compile(r"\bthe revised (version|question)\b", re.I),
    re.compile(r"\bwhat (is|'s) (the|your) (rephrased|revised|new|modified)\b", re.I),
    re.compile(r"^(sure|okay|ok|certainly|of course)[,!.\s]+(here)", re.I),
    re.compile(r"\bhere'?s the (rephrased|revised|modified)\b", re.I),
    re.compile(r"\bI need (more|the) (question|information|details)\b", re.I),
    re.compile(r"\bcould you (clarify|provide|give)\b", re.I),
    re.compile(r"\bI understand you want me to\b", re.I),
    re.compile(r"\bplease provide (more|the|a|additional)\b", re.I),
    re.compile(r"\bI cannot proceed\b", re.I),
]


def is_bad_variant_text(text):
    if not text or not isinstance(text, str):
        return True
    if len(text.strip()) < 5:
        return True
    for p in BAD_VARIANT_PATTERNS:
        if p.search(text):
            return True
    return False


class VariantGeneratorV2:
    """使用 LLM 生成语义等价变体，带等价性验证

    `gen_llm_fn`：可独立指定的 variant 生成模型。若为 None 则与 llm_fn 共用。
    强烈建议在 cross-model 实验里固定一个稳定的 generator（例如 Qwen2.5-3B），
    否则 generator 失败会污染下游 IR（参见 contamination_audit）。
    """

    PERTURBATION_TYPES = ["paraphrase", "synonym", "reorder", "format", "distractor"]

    def __init__(self, llm_fn, gen_llm_fn=None, max_retries=2):
        self.llm_fn = llm_fn
        self.gen_llm_fn = gen_llm_fn if gen_llm_fn is not None else llm_fn
        self.max_retries = max_retries

    def generate_variants(self, sample, types=None):
        """为一个样本生成 5 个变体（每种扰动类型一个）"""
        if types is None:
            types = self.PERTURBATION_TYPES

        variants = []
        for ptype in types:
            variant = self._generate_one(sample, ptype)
            variant["perturbation_type"] = ptype
            variant["original_id"] = sample.get("id", "unknown")
            variants.append(variant)
        return variants

    def _generate_one(self, sample, ptype):
        """使用 LLM 生成单个变体"""
        question = sample.get("question", sample.get("problem", ""))
        context = sample.get("context", "")
        answer = sample.get("answer", "")

        if ptype == "paraphrase":
            prompt = (
                "Rephrase the following question so it has EXACTLY the same meaning, "
                "keeping ALL proper nouns, numbers, units, and named entities IDENTICAL. "
                "Only change syntactic structure and non-essential wording. "
                "Output ONLY the rephrased question on a single line, with no preamble.\n\n"
                "Original: %s\n\n"
                "Rephrased:" % question
            )
            new_q = self._gen_with_retry(prompt, question)
            if new_q is None:
                return self._rule_based(sample, ptype)
            variant = dict(sample)
            variant["question"] = new_q
            if "problem" in sample:
                variant["problem"] = new_q
            return variant

        elif ptype == "synonym":
            prompt = (
                "Replace 2-3 common verbs/adjectives in this question with their synonyms. "
                "DO NOT change any proper nouns, numbers, units, named entities, "
                "or domain-specific terms (e.g. keep 'ducks' as 'ducks', not 'geese'; "
                "keep '$2' as '$2'). "
                "Output ONLY the modified question on a single line, with no preamble.\n\n"
                "Original: %s\n\n"
                "Modified:" % question
            )
            new_q = self._gen_with_retry(prompt, question)
            if new_q is None:
                return self._rule_based(sample, ptype)
            variant = dict(sample)
            variant["question"] = new_q
            if "problem" in sample:
                variant["problem"] = new_q
            return variant

        elif ptype == "reorder":
            # 确定性操作：重排 question 中的陈述句顺序（疑问句保留在最后）。
            # 兼容两种 benchmark：
            #   * 有 context（hotpotqa）：同时也打乱 context 的句子。
            #   * 无 context（gsm8k/math）：只在 question 内部重排，确保产生真扰动。
            # FIX 2026-05-13: 旧版 surface 扰动当 context 为空时直接 return 原样本，
            # 导致 gsm8k/math 上 surface IR 恒为 0%（伪现象）。
            variant = dict(sample)
            seed = int(hashlib.md5(question.encode()).hexdigest()[:8], 16)
            rng = random.Random(seed)
            # 1) reorder context if present
            if context:
                ctx_sents = [s.strip() for s in context.split('.') if s.strip()]
                if len(ctx_sents) > 1:
                    rng.shuffle(ctx_sents)
                    variant["context"] = '. '.join(ctx_sents) + '.'
            # 2) reorder declarative sentences inside question
            #    rule: 拆成句子，最后一个含 '?' 的句子保留位置不变（一般是疑问主句），
            #          其余陈述句随机重排。
            q_sents = [s.strip() for s in re.split(r'(?<=[\.\?\!])\s+', question.strip()) if s.strip()]
            if len(q_sents) >= 3:
                # 找最后一个 ? 句作为锚点
                last_q_idx = max((i for i, s in enumerate(q_sents) if '?' in s), default=len(q_sents) - 1)
                anchor = q_sents[last_q_idx]
                others = q_sents[:last_q_idx] + q_sents[last_q_idx + 1:]
                rng2 = random.Random(seed ^ 0x5A5A)
                rng2.shuffle(others)
                new_q = ' '.join(others + [anchor])
                variant["question"] = new_q
                if "problem" in sample:
                    variant["problem"] = new_q
            elif len(q_sents) == 2:
                # 典型形态："陈述句. 疑问句?"。简单交换会让疑问句开头不通顺，
                # 所以走 rule_based_q_reorder 在陈述句内部拆子句重排，疑问句保留末尾。
                variant = self._rule_based_q_reorder(sample, question, seed)
            else:
                # 单句：用 rule-based 子句重排兜底
                variant = self._rule_based_q_reorder(sample, question, seed)
            return variant

        elif ptype == "format":
            # 确定性操作：把 question 改成带 bullet 的格式，每个陈述句独占一行；
            # 疑问主句保留在最后单独一行，前缀 "Question:"。
            # 兼容 context：context 也变成 bullet 格式（与旧实现一致）。
            variant = dict(sample)
            if context:
                ctx_sents = [s.strip() for s in context.split('.') if s.strip()]
                if ctx_sents:
                    variant["context"] = '\n'.join('• ' + s + '.' for s in ctx_sents)
            q_sents = [s.strip() for s in re.split(r'(?<=[\.\?\!])\s+', question.strip()) if s.strip()]
            if len(q_sents) >= 2:
                lines = []
                question_line = None
                for s in q_sents:
                    if '?' in s and question_line is None:
                        question_line = s
                    else:
                        lines.append('- ' + s)
                # 疑问句不加 bullet，单独前缀 "Question:"
                if question_line is None:
                    question_line = q_sents[-1]
                    lines = ['- ' + s for s in q_sents[:-1]]
                new_q = '\n'.join(lines) + '\nQuestion: ' + question_line
                variant["question"] = new_q
                if "problem" in sample:
                    variant["problem"] = new_q
            else:
                # 单句兜底：用 markdown 引用块包裹
                new_q = "> " + question.strip()
                variant["question"] = new_q
                if "problem" in sample:
                    variant["problem"] = new_q
            return variant

        elif ptype == "distractor":
            # 确定性操作：在 question 前注入"显式无关"的引导句 + 一段 distractor，
            # 然后再给原 question。如有 context，也在尾部追加 distractor。
            # 关键：明确告诉模型"以下信息与问题无关"，使语义理论上不变；模型若被干扰则计入 IR。
            variant = dict(sample)
            distractors = [
                "The weather that day was particularly sunny with clear skies.",
                "Several other researchers had studied similar topics in the past decade.",
                "The local newspaper reported extensively on unrelated events that week.",
                "Meanwhile, advances in technology continued to reshape daily life.",
                "Historical records from that era are preserved in multiple archives.",
            ]
            seed = int(hashlib.md5(question.encode()).hexdigest()[:8], 16)
            d = distractors[seed % len(distractors)]
            # 1) 给 question 加 distractor 前缀
            preface = "Note: the following sentence is unrelated to the question. " + d + " Now, "
            new_q = preface + question[0].lower() + question[1:] if question else preface
            variant["question"] = new_q
            if "problem" in sample:
                variant["problem"] = new_q
            # 2) context 也加（向后兼容 hotpotqa）
            if context:
                variant["context"] = context + " " + d
            return variant

        return dict(sample)

    def _gen_with_retry(self, prompt, original_question):
        """调用 generator，最多 max_retries+1 次；过滤 bad pattern 与退化输出。

        Returns clean variant string, or None if all attempts fail.
        """
        for attempt in range(self.max_retries + 1):
            try:
                response = self.gen_llm_fn(prompt)
            except Exception:
                continue
            if not response:
                continue
            # take first non-empty line; strip surrounding quotes/markdown
            new_q = None
            for line in response.strip().split('\n'):
                line = line.strip().strip('"').strip("'").strip()
                if line:
                    new_q = line
                    break
            if not new_q:
                continue
            # filter contamination patterns
            if is_bad_variant_text(new_q):
                continue
            # filter degenerate length
            if len(new_q) < max(8, len(original_question) // 4):
                continue
            if len(new_q) > 4 * len(original_question) + 50:
                continue
            # filter near-duplicate (no real change)
            if new_q.strip().lower() == original_question.strip().lower():
                continue
            return new_q
        return None

    def _rule_based(self, sample, ptype):
        """LLM 失败时的规则回退"""
        variant = dict(sample)
        question = sample.get("question", sample.get("problem", ""))

        if ptype == "paraphrase":
            new_q = "Please answer: " + question
        elif ptype == "synonym":
            replacements = {
                "calculate": "compute", "find": "determine", "solve": "work out",
                "born": "brought into the world", "created": "produced",
                "What": "Which thing", "How many": "What is the count of",
            }
            new_q = question
            for old, new in replacements.items():
                if old.lower() in new_q.lower():
                    new_q = re.sub(re.escape(old), new, new_q, count=1, flags=re.IGNORECASE)
                    break
        else:
            return variant

        variant["question"] = new_q
        if "problem" in sample:
            variant["problem"] = new_q
        return variant

    def _rule_based_q_reorder(self, sample, question, seed):
        """Rule-based question reorder: 用于单句 question 的兜底扰动。

        策略：把 ", " 分隔的子句重排（数学题常见 "Janet has 16 ducks, she eats 3, ...".
        若没有逗号子句，则在末尾追加一句中性引导语，确保产生字符级差异（非 no-op）。
        """
        variant = dict(sample)
        # 找出疑问主句（最后一个 '?' 之前的部分通常含子句）
        q = question.strip()
        # split by ", " preserving punctuation; only re-shuffle the prefix part before the question mark
        if '?' in q:
            stem, _, tail = q.rpartition('?')
            tail = tail.strip()
            parts = [p.strip() for p in stem.split(',') if p.strip()]
        else:
            stem, tail = q, ''
            parts = [p.strip() for p in stem.split(',') if p.strip()]
        if len(parts) >= 3:
            # 锚点：保留第一句和最后一句（含主语/疑问），重排中间
            head = parts[0]
            anchor_last = parts[-1]
            middle = parts[1:-1]
            rng = random.Random(seed ^ 0xC3C3)
            rng.shuffle(middle)
            new_stem = ', '.join([head] + middle + [anchor_last])
            new_q = (new_stem + '?' + (' ' + tail if tail else '')).strip() if '?' in q else new_stem
            if new_q != q:
                variant["question"] = new_q
                if "problem" in sample:
                    variant["problem"] = new_q
                return variant
        # fallback: 在末尾追加显式中性引导语（保证字符差异 + 语义不变）
        new_q = q.rstrip() + " Please reason step by step."
        if new_q == q:
            new_q = q + " "  # 极端兜底
        variant["question"] = new_q
        if "problem" in sample:
            variant["problem"] = new_q
        return variant


# ============================================================
# Equivalence Validator v2 — LLM 验证语义等价性
# ============================================================

class EquivalenceValidatorV2:
    """使用 LLM 验证变体与原始样本的语义等价性

    与 generator 同一原则：可以传入独立的 `val_llm_fn`，避免被测模型自验。
    """

    def __init__(self, llm_fn, val_llm_fn=None):
        self.llm_fn = llm_fn
        self.val_llm_fn = val_llm_fn if val_llm_fn is not None else llm_fn

    def validate(self, original, variant, ptype):
        """验证变体是否语义等价。

        v3 设计 (2026-05-13)：
          * paraphrase / synonym  → LLM judge（generator 偶尔会改数字/实体，必须查）。
          * reorder / format / distractor → **结构化检查**，不走 LLM。
            理由：这三类都是确定性程序变换，理论上保答案。我们只需验证程序输出
            没有意外丢失原 question 的关键信息（数字、$ 符号、% 符号、问号），
            这是个纯字符串操作，根本不需要 LLM。
            之前用 qwen2.5:3b 当 surface judge 时 distractor reject=100%、
            reorder/format reject 20-50%，是 3B 模型遵循指令能力不足导致的，
            不是 prompt 工程能修的——所以彻底改用规则。

          * fallback：判 fail（exception 等）保守接受。
        """
        orig_q = original.get("question", original.get("problem", ""))
        var_q = variant.get("question", variant.get("problem", ""))
        orig_ctx = original.get("context", "") or ""
        var_ctx = variant.get("context", "") or ""
        answer = original.get("answer", "")

        # 如果变体跟原文完全相同（极端兜底场景），直接判等价
        if orig_q == var_q and orig_ctx == var_ctx:
            return {
                "is_equivalent": True,
                "confidence": 1.0,
                "reason": "Variant identical to original (no-op); auto-equiv.",
            }

        # ------ surface 三类：结构化 token-set 检查 ------
        if ptype in ("reorder", "format", "distractor"):
            return self._structural_check(orig_q, var_q, ptype)

        # ------ semantic 两类：LLM judge ------
        prompt = (
            "Two question reformulations are considered EQUIVALENT if and only if they "
            "would yield the SAME ANSWER when correctly solved. "
            "Minor wording differences (e.g. 'a third' vs 'one third', 'compute' vs "
            "'calculate', sentence reordering, added pleasantries) are EQUIVALENT. "
            "Different proper nouns, different numbers, different units, or different "
            "physical quantities are NOT equivalent.\n\n"
            "The reference answer to question A is: '%s'.\n\n"
            "Question A: %s\n"
            "Question B: %s\n\n"
            "Reply ONLY 'YES' or 'NO' on the first line, then a brief reason." % (answer, orig_q, var_q)
        )

        try:
            response = self.val_llm_fn(prompt)
            first_line = response.strip().split('\n')[0].upper()
            is_eq = 'YES' in first_line
            return {
                "is_equivalent": is_eq,
                "confidence": 0.9 if is_eq else 0.2,
                "reason": response.strip()[:200],
            }
        except Exception as e:
            # 如果验证失败，保守地接受
            return {"is_equivalent": True, "confidence": 0.5, "reason": "Validation failed: %s" % str(e)}

    def _structural_check(self, orig_q, var_q, ptype):
        """对 surface 三类做结构化 token-set 检查：

        关键不变量（变体 token 必须 ⊇ 原问题）：
          1. 所有数字（含小数、百分号、负号）
          2. 货币/单位符号（$ £ € %）
          3. 疑问词（what/how/which/when/where/why/who/whose）
          4. 问号 '?'

        如果原问题中出现的某个关键 token 在变体里消失了，说明确定性变换出了 bug
        （例如 reorder 把疑问句弄丢了），此时返回 NO；否则 YES。
        """
        import re as _re

        def numbers(s):
            # 提取所有数字（包括 '$2', '150%', '-3.5', '1,000'）
            return set(_re.findall(r'-?\d[\d,]*\.?\d*', s.replace(',', '')))

        def symbols(s):
            return set(c for c in s if c in '$£€%')

        def question_marks(s):
            return s.count('?')

        wh_words = {'what', 'how', 'which', 'when', 'where', 'why', 'who', 'whose', 'whom'}

        def wh(s):
            tokens = _re.findall(r"[A-Za-z']+", s.lower())
            return set(t for t in tokens if t in wh_words)

        orig_nums = numbers(orig_q)
        var_nums = numbers(var_q)
        missing_nums = orig_nums - var_nums

        orig_syms = symbols(orig_q)
        var_syms = symbols(var_q)
        missing_syms = orig_syms - var_syms

        orig_wh = wh(orig_q)
        var_wh = wh(var_q)
        missing_wh = orig_wh - var_wh

        orig_qmarks = question_marks(orig_q)
        var_qmarks = question_marks(var_q)
        lost_qmark = orig_qmarks > 0 and var_qmarks == 0

        if missing_nums or missing_syms or missing_wh or lost_qmark:
            issues = []
            if missing_nums:
                issues.append("missing_numbers=%s" % sorted(missing_nums))
            if missing_syms:
                issues.append("missing_symbols=%s" % sorted(missing_syms))
            if missing_wh:
                issues.append("missing_wh_words=%s" % sorted(missing_wh))
            if lost_qmark:
                issues.append("lost_question_mark")
            return {
                "is_equivalent": False,
                "confidence": 0.95,
                "reason": "Structural check failed for %s: %s" % (ptype, "; ".join(issues)),
            }

        return {
            "is_equivalent": True,
            "confidence": 1.0,
            "reason": "Structural check passed (%s; nums/symbols/wh/qmark preserved)" % ptype,
        }


# ============================================================
# Failure Propagation Graph — 核心新颖性
# ============================================================

class FailurePropagationAnalyzer:
    """
    分析多步 Agent 执行中的失败传播模式。
    
    这是 AgentDiff 区别于 CheckList 的核心贡献：
    - CheckList 只看最终输出是否一致
    - AgentDiff 追踪失败在哪一步发生、是否级联传播
    """

    def analyze_propagation(self, original_trace, variant_traces):
        """
        比较原始轨迹和变体轨迹，构建失败传播图。
        
        Returns:
            dict: {
                "divergence_points": [{step, type, severity}],
                "cascade_depth": int,  # 级联深度
                "self_corrections": int,  # 自我修正次数
                "propagation_pattern": str,  # "early_diverge", "late_diverge", "cascade", "self_correct"
            }
        """
        results = []
        for vt in variant_traces:
            ptype = vt.get("perturbation_type", "unknown")
            var_trace = vt.get("trace", [])
            analysis = self._compare_traces(original_trace, var_trace)
            analysis["perturbation_type"] = ptype
            analysis["variant_answer"] = vt.get("final_answer", "")
            results.append(analysis)
        return results

    def _compare_traces(self, orig_trace, var_trace):
        """逐步比较两个执行轨迹"""
        orig_steps = orig_trace if isinstance(orig_trace, list) else orig_trace.get("trace", [])
        var_steps = var_trace if isinstance(var_trace, list) else var_trace.get("trace", [])

        max_steps = max(len(orig_steps), len(var_steps))
        if max_steps == 0:
            return {
                "divergence_step": 0,
                "cascade_depth": 0,
                "self_corrections": 0,
                "propagation_pattern": "no_trace",
                "step_similarities": [],
            }

        step_similarities = []
        first_divergence = None
        diverged = False
        self_corrections = 0
        cascade_depth = 0

        for i in range(max_steps):
            orig_thought = orig_steps[i].get("thought", "") if i < len(orig_steps) else ""
            var_thought = var_steps[i].get("thought", "") if i < len(var_steps) else ""
            orig_action = orig_steps[i].get("action", "") if i < len(orig_steps) else ""
            var_action = var_steps[i].get("action", "") if i < len(var_steps) else ""

            # 计算思维相似度
            thought_sim = self._jaccard_similarity(orig_thought, var_thought)
            # 计算动作相似度
            action_sim = 1.0 if orig_action == var_action else 0.0

            step_sim = {
                "step": i + 1,
                "thought_similarity": thought_sim,
                "action_match": action_sim == 1.0,
                "is_divergent": thought_sim < 0.4 or action_sim < 1.0,
            }
            step_similarities.append(step_sim)

            if step_sim["is_divergent"]:
                if first_divergence is None:
                    first_divergence = i + 1
                if diverged:
                    cascade_depth += 1
                diverged = True
            else:
                if diverged:
                    self_corrections += 1
                    diverged = False

        # 确定传播模式
        if first_divergence is None:
            pattern = "consistent"
        elif first_divergence == 1:
            pattern = "early_diverge"
        elif cascade_depth > 1:
            pattern = "cascade"
        elif self_corrections > 0:
            pattern = "self_correct"
        else:
            pattern = "late_diverge"

        return {
            "divergence_step": first_divergence or 0,
            "cascade_depth": cascade_depth,
            "self_corrections": self_corrections,
            "propagation_pattern": pattern,
            "step_similarities": step_similarities,
            "n_steps_original": len(orig_steps),
            "n_steps_variant": len(var_steps),
        }

    def _jaccard_similarity(self, a, b):
        if not a or not b:
            return 0.0
        words_a = set(a.lower().split())
        words_b = set(b.lower().split())
        if not words_a or not words_b:
            return 0.0
        intersection = words_a & words_b
        union = words_a | words_b
        return len(intersection) / len(union)


# ============================================================
# Step-Level Patch Generator — 步级修补
# ============================================================

class StepLevelPatcher:
    """
    基于失败传播分析生成步级修补。
    
    区别于 v1 的系统级修补：
    - 在特定步骤插入针对性提示
    - 基于失败传播图确定最佳干预点
    """

    PATCH_TEMPLATES = {
        "paraphrase": "CRITICAL: Focus on the core meaning, not specific wording. "
                      "Different phrasings of the same question have the same answer.",
        "synonym": "CRITICAL: Treat synonymous terms as identical. "
                   "'Calculate'='compute'='find'='determine'. Do not let word choice affect your answer.",
        "reorder": "CRITICAL: Information order does not matter. "
                   "Read ALL context before reasoning.",
        "format": "CRITICAL: Formatting (paragraphs vs bullets) does not change facts. "
                  "Extract information regardless of format.",
        "distractor": "CRITICAL: Ignore irrelevant sentences. "
                      "Focus only on facts needed to answer the question.",
    }

    def generate_step_patch(self, propagation_analysis, failure_types):
        """
        生成步级修补。
        
        Args:
            propagation_analysis: FailurePropagationAnalyzer 的输出
            failure_types: 导致失败的扰动类型列表
        
        Returns:
            dict: {
                "intervention_step": int,
                "patch_text": str,
                "patch_level": "step",
                "target_types": list,
            }
        """
        # 找到最常见的分歧步骤
        divergence_steps = [p["divergence_step"] for p in propagation_analysis if p["divergence_step"] > 0]
        if not divergence_steps:
            return {"intervention_step": 1, "patch_text": "", "patch_level": "none", "target_types": []}

        # 最佳干预点 = 最常见的分歧步骤
        step_counts = Counter(divergence_steps)
        best_step = step_counts.most_common(1)[0][0]

        # 组合针对性修补
        type_counts = Counter(failure_types)
        top_types = [t for t, _ in type_counts.most_common(2)]
        patches = [self.PATCH_TEMPLATES.get(t, "") for t in top_types if t in self.PATCH_TEMPLATES]
        patch_text = " ".join(patches)

        return {
            "intervention_step": best_step,
            "patch_text": patch_text,
            "patch_level": "step",
            "target_types": top_types,
        }

    def generate_system_patch(self, failure_types):
        """生成系统级修补（作为对照）"""
        type_counts = Counter(failure_types)
        top_types = [t for t, _ in type_counts.most_common(2)]
        patches = [self.PATCH_TEMPLATES.get(t, "") for t in top_types if t in self.PATCH_TEMPLATES]
        return {
            "intervention_step": 0,
            "patch_text": " ".join(patches),
            "patch_level": "system",
            "target_types": top_types,
        }


# ============================================================
# Consistency Analyzer v2 — 增强版一致性分析
# ============================================================

class ConsistencyAnalyzerV2:
    """增强版一致性分析，集成失败传播图"""

    def __init__(self):
        self.propagation_analyzer = FailurePropagationAnalyzer()

    def analyze(self, original_result, variant_results, gold_answer=None):
        """完整的一致性分析"""
        orig_answer = original_result["final_answer"]
        orig_trace = original_result.get("trace", [])

        # 基本一致性
        inconsistent = []
        per_type = defaultdict(lambda: {"total": 0, "inconsistent": 0})

        for vr in variant_results:
            ptype = vr.get("perturbation_type", "unknown")
            per_type[ptype]["total"] += 1
            if not self._answer_match(orig_answer, vr["final_answer"]):
                inconsistent.append({
                    "perturbation_type": ptype,
                    "original_answer": orig_answer,
                    "variant_answer": vr["final_answer"],
                })
                per_type[ptype]["inconsistent"] += 1

        # 失败传播分析
        propagation = self.propagation_analyzer.analyze_propagation(orig_trace, variant_results)

        # 正确性
        is_correct = self._answer_match(orig_answer, str(gold_answer)) if gold_answer else None

        # 汇总
        n_variants = len(variant_results)
        consistency_rate = 1.0 - len(inconsistent) / n_variants if n_variants > 0 else 1.0

        per_type_rates = {}
        for ptype, counts in per_type.items():
            per_type_rates[ptype] = counts["inconsistent"] / counts["total"] if counts["total"] > 0 else 0.0

        # 传播模式统计
        patterns = Counter(p["propagation_pattern"] for p in propagation)

        return {
            "sample_id": original_result["sample_id"],
            "consistency_rate": consistency_rate,
            "is_consistent": len(inconsistent) == 0,
            "inconsistent_count": len(inconsistent),
            "total_variants": n_variants,
            "is_original_correct": is_correct,
            "per_type_inconsistency": per_type_rates,
            "propagation_analysis": propagation,
            "propagation_patterns": dict(patterns),
            "failure_types": [i["perturbation_type"] for i in inconsistent],
        }

    def _answer_match(self, pred, gold):
        """宽松答案匹配 — 处理 LLM 输出冗长答案的情况"""
        pred = str(pred).lower().strip().rstrip('.').strip()
        gold = str(gold).lower().strip().rstrip('.').strip()
        if pred == gold:
            return True
        # 包含匹配（gold 在 pred 中，或 pred 在 gold 中）
        if gold in pred or pred in gold:
            return True
        # 数字匹配
        try:
            pf, gf = float(pred), float(gold)
            if abs(pf - gf) < 0.01:
                return True
        except (ValueError, TypeError):
            pass
        # 去除标点后匹配
        pred_clean = re.sub(r'[^\w\s]', '', pred)
        gold_clean = re.sub(r'[^\w\s]', '', gold)
        if pred_clean == gold_clean:
            return True
        # 提取 pred 中的关键词，看是否包含 gold 的核心词
        gold_words = set(gold_clean.split())
        pred_words = set(pred_clean.split())
        if gold_words and gold_words.issubset(pred_words):
            return True
        # 对于短答案（<5 词），检查 pred 的第一个词/短语是否匹配
        if len(gold.split()) <= 3:
            # 尝试从 pred 中提取简短答案
            pred_first = pred.split('.')[0].split(',')[0].strip()
            if gold in pred_first or pred_first in gold:
                return True
        return False


# ============================================================
# AgentDiff Pipeline v2 — 完整流水线
# ============================================================

class AgentDiffPipelineV2:
    """AgentDiff v2 完整流水线"""

    def __init__(self, llm_fn, agent_type="react", perturbation_types=None,
                 gen_llm_fn=None, val_llm_fn=None):
        """
        Args:
            llm_fn: function used by the *agent under test* to answer questions.
            gen_llm_fn: optional independent function used to *generate variants*.
                Strongly recommended in cross-model experiments to use a stable
                generator (e.g. Qwen2.5-3B). Default: same as llm_fn (legacy).
            val_llm_fn: optional independent equivalence validator. Default: same
                as llm_fn.
        """
        self.llm_fn = llm_fn
        self.gen_llm_fn = gen_llm_fn if gen_llm_fn is not None else llm_fn
        self.val_llm_fn = val_llm_fn if val_llm_fn is not None else llm_fn
        self.agent_type = agent_type
        self.perturbation_types = perturbation_types or VariantGeneratorV2.PERTURBATION_TYPES

        self.variant_gen = VariantGeneratorV2(llm_fn=llm_fn, gen_llm_fn=self.gen_llm_fn)
        self.validator = EquivalenceValidatorV2(llm_fn=llm_fn, val_llm_fn=self.val_llm_fn)
        self.analyzer = ConsistencyAnalyzerV2()
        self.patcher = StepLevelPatcher()

        # 创建 agent
        if agent_type == "react":
            self.agent = ReActAgent(llm_fn)
        elif agent_type == "cot":
            self.agent = CoTAgent(llm_fn)
        elif agent_type == "direct":
            self.agent = DirectAgent(llm_fn)
        else:
            raise ValueError("Unknown agent type: %s" % agent_type)

    def run_single(self, sample, do_patch=True, validate_variants=True):
        """
        对单个样本运行完整 AgentDiff 流水线。
        
        Returns:
            dict: 完整分析结果，包含执行轨迹、一致性分析、传播图、修补效果
        """
        t0 = time.time()

        # Step 1: 生成变体
        variants = self.variant_gen.generate_variants(sample, types=self.perturbation_types)

        # Step 2: 验证等价性（可选）
        valid_variants = variants
        validation_results = []
        if validate_variants:
            for v in variants:
                vr = self.validator.validate(sample, v, v["perturbation_type"])
                validation_results.append(vr)
                if not vr["is_equivalent"]:
                    logger.info("Variant rejected: %s (reason: %s)",
                                v["perturbation_type"], vr["reason"])
            valid_variants = [v for v, vr in zip(variants, validation_results) if vr["is_equivalent"]]
        else:
            validation_results = [
                {"is_equivalent": True, "confidence": 1.0, "reason": "validation disabled"}
                for _ in variants
            ]

        # Step 3: 运行 agent 在原始样本上
        original_result = self.agent.run(sample)

        # Step 4: 运行 agent 在每个变体上
        variant_results = []
        for v in valid_variants:
            vr = self.agent.run(v)
            vr["perturbation_type"] = v["perturbation_type"]
            variant_results.append(vr)

        # Step 4b: 留存变体原文 + judge 结果 + agent 轨迹（用于 sanity / judge swap / 复判）
        perturbation_variants_log = []
        # 建一个从 id(v) 到 agent run 结果的 lookup（variant 对象身份比较即可，生成时都是 dict(sample) 浅拷贝）
        valid_ids = {id(vv): vr for vv, vr in zip(valid_variants, variant_results)}
        for v, vres in zip(variants, validation_results):
            vq = v.get("question", v.get("problem", ""))
            vctx = v.get("context", "")
            ar = valid_ids.get(id(v))
            entry = {
                "perturbation_type": v.get("perturbation_type", "unknown"),
                "variant_question": vq,
                "variant_context": vctx if vctx else None,
                "validation": {
                    "is_equivalent": vres.get("is_equivalent"),
                    "confidence": vres.get("confidence"),
                    "reason": vres.get("reason"),
                },
                "agent_run": None,
            }
            if ar is not None:
                entry["agent_run"] = {
                    "final_answer": ar.get("final_answer"),
                    "n_steps": ar.get("n_steps"),
                    "trace": ar.get("trace"),
                }
            perturbation_variants_log.append(entry)

        # Step 5: 一致性分析 + 失败传播
        gold_answer = sample.get("answer", None)
        analysis = self.analyzer.analyze(original_result, variant_results, gold_answer)

        # Step 6: 修补（可选）
        patch_results = None
        if do_patch and not analysis["is_consistent"]:
            patch_results = self._run_patching(sample, valid_variants, analysis, gold_answer)

        elapsed = time.time() - t0

        return {
            "sample_id": sample.get("id", "unknown"),
            "agent_type": self.agent_type,
            "sample_question": sample.get("question", sample.get("problem", "")),
            "sample_context": sample.get("context", "") or None,
            "sample_gold_answer": sample.get("answer", None),
            "original_result": {
                "final_answer": original_result["final_answer"],
                "n_steps": original_result["n_steps"],
                "is_correct": analysis["is_original_correct"],
                "trace": original_result.get("trace"),
            },
            "consistency_analysis": {
                "consistency_rate": analysis["consistency_rate"],
                "is_consistent": analysis["is_consistent"],
                "inconsistent_count": analysis["inconsistent_count"],
                "per_type_inconsistency": analysis["per_type_inconsistency"],
                "propagation_patterns": analysis["propagation_patterns"],
            },
            "propagation_details": analysis["propagation_analysis"],
            "perturbation_variants": perturbation_variants_log,
            "patch_results": patch_results,
            "n_variants_generated": len(variants),
            "n_variants_valid": len(valid_variants),
            "n_variants_rejected": len(variants) - len(valid_variants),
            "elapsed_seconds": elapsed,
        }

    def _run_patching(self, sample, variants, analysis, gold_answer):
        """运行步级和系统级修补"""
        failure_types = analysis["failure_types"]
        propagation = analysis["propagation_analysis"]

        # 步级修补
        step_patch = self.patcher.generate_step_patch(propagation, failure_types)
        # 系统级修补
        system_patch = self.patcher.generate_system_patch(failure_types)

        results = {}

        # 测试步级修补
        if step_patch["patch_text"]:
            step_prompt = "Before step %d: %s" % (step_patch["intervention_step"], step_patch["patch_text"])
            patched_orig = self.agent.run(sample, extra_prompt=step_prompt)
            patched_variants = []
            for v in variants:
                pvr = self.agent.run(v, extra_prompt=step_prompt)
                pvr["perturbation_type"] = v["perturbation_type"]
                patched_variants.append(pvr)
            patched_analysis = self.analyzer.analyze(patched_orig, patched_variants, gold_answer)
            results["step_patch"] = {
                "consistency_rate": patched_analysis["consistency_rate"],
                "inconsistent_count": patched_analysis["inconsistent_count"],
                "is_correct": patched_analysis["is_original_correct"],
                "intervention_step": step_patch["intervention_step"],
                "target_types": step_patch["target_types"],
            }

        # 测试系统级修补
        if system_patch["patch_text"]:
            sys_orig = self.agent.run(sample, extra_prompt=system_patch["patch_text"])
            sys_variants = []
            for v in variants:
                svr = self.agent.run(v, extra_prompt=system_patch["patch_text"])
                svr["perturbation_type"] = v["perturbation_type"]
                sys_variants.append(svr)
            sys_analysis = self.analyzer.analyze(sys_orig, sys_variants, gold_answer)
            results["system_patch"] = {
                "consistency_rate": sys_analysis["consistency_rate"],
                "inconsistent_count": sys_analysis["inconsistent_count"],
                "is_correct": sys_analysis["is_original_correct"],
                "target_types": system_patch["target_types"],
            }

        return results
