# Azure AI Foundry — LLM-as-a-Judge Evaluation Guide

## How LLM-as-a-Judge Works

An LLM-as-a-Judge uses a **separate LLM** (the "judge") to score the agent's responses — the same way a human QA expert would, but automated and repeatable.

### Technical Flow

| Step | What Happens |
|------|-------------|
| **1. Template** | Each evaluator has a `.prompty` template containing a scoring rubric with level definitions (1–5) and few-shot examples |
| **2. Promptify** | The SDK injects your actual `query`, `response`, and `context` into the template placeholders (`{{query}}`, `{{response}}`, `{{context}}`) |
| **3. Judge Call** | The fully rendered prompt is sent to the judge LLM (e.g., GPT-4.1) with `temperature=0` for deterministic scoring |
| **4. Structured Output** | The judge returns a chain-of-thought (`<S0>`), explanation (`<S1>`), and integer score (`<S2>`) |
| **5. Parse** | The SDK extracts the score and reason from the structured tags |

The judge sees the **rubric + your data** in a single prompt. It never sees the agent's system prompt or internal reasoning — only the final output and the tool results (context).

---

## Built-in Evaluators

| Evaluator | What It Measures | Scale | Detects |
|-----------|------------------|-------|---------|
| **Groundedness** | Every claim backed by tool output? | 1–5 | Hallucination |
| **Relevance** | Does the response address the query? | 1–5 | Off-topic answers |
| **Task Adherence** | Did the agent follow its instructions? | 0/1 | Skipped steps, rule violations |
| **Intent Resolution** | Were all user intents resolved? | 1–5 | Partial answers |
| **Tool Call Accuracy** | Right tools, right arguments? | 1–5 | Wrong tool usage |
| **Coherence** | Well-structured and logical? | 1–5 | Disorganized responses |
| **Fluency** | Natural, professional language? | 1–5 | Awkward phrasing |
| **Response Completeness** | Covers all expected behavior? | 1–5 | Missing information |

---

## Worked Example: Evaluating a Hallucinated Response

### The Scenario

**Customer Query:**
> "I ordered a book and supposed to delivery by 20th March. Where is the order now?"

**Tool Output (Context — the source of truth):**
```json
{
  "order_id": "55555",
  "product": "Book - The Silent Echo",
  "status": "shipped",
  "carrier": "FedEx",
  "tracking_number": "FX111222333",
  "estimated_delivery": "2026-03-20",
  "amount": 29.99
}
```

**KB Policy (also in Context):**
```json
{
  "policy": "If delivery is 3 or more days past the estimated delivery date,
  the customer is eligible for a $10 account credit or free reshipment."
}
```

**Agent Response:**
> "Sorry for the delay, the order is on the way and it will be delivered by 25th March. Since it is delayed for 5 days, you are eligible for $10 rewards."

**Today's date:** March 24, 2026

---

### 1. Groundedness (Hallucination Detection) — Score: 2/5

The judge checks: **Is every claim in the response backed by the tool output?**

| Claim in Response | Evidence in Tool Output | Grounded? |
|---|---|---|
| "the order is on the way" | `status: "shipped"` | **Yes** — shipped = on the way |
| "delivered by 25th March" | Tool says `estimated_delivery: "2026-03-20"` — no mention of March 25 anywhere | **No — HALLUCINATED.** The agent invented a new delivery date |
| "delayed for 5 days" | Tool shows estimated March 20, today is March 24 = 4 days late, not 5. The tool never stated "5 days" | **No — HALLUCINATED.** The model did its own math and got it wrong |
| "$10 rewards" | Policy says "$10 account **credit**", not "rewards" | **Partially** — correct amount but wrong terminology |

**Judge Reasoning:**
> "Let's think step by step: The response claims delivery by March 25 — I cannot find this date in the context. The response claims a 5-day delay — the tool output shows estimated delivery March 20 and does not state the number of delay days. The agent appears to have computed this itself and computed it incorrectly (today is March 24 = 4 days, not 5). The $10 figure matches the policy but 'rewards' misquotes 'account credit'."

**Score: 2/5** — Attempts to respond but contains incorrect information not supported by context.

---

### 2. Relevance — Score: 3/5

The judge checks: **Does the response address what the customer actually asked?**

| Customer Asked | Response Addressed It? |
|---|---|
| "Where is the order now?" | **Yes** — said "on the way" (status) |
| Implied: tracking details | **No** — never mentioned carrier (FedEx) or tracking number (FX111222333) |

**Judge Reasoning:**
> "The customer asked where the order is. The response confirms it's on the way but omits the tracking number and carrier which are available in the tool data. The response also proactively discusses delay compensation, which is helpful but doesn't fully answer the location question."

**Score: 3/5** — Addresses the query partially but misses key details the customer would need.

---

### 3. Task Adherence — Score: 0/1 (FAIL)

The judge checks: **Did the agent follow its system instructions?**

| Instruction Rule | Followed? |
|---|---|
| "Always look up order status before answering" | **Yes** — tool was called |
| "Provide specific details (tracking numbers, dates, amounts) when available" | **No** — tracking number FX111222333 omitted |
| "Never make up information — only use data from tool results" | **No** — invented March 25 date and calculated 5-day delay |
| "For late orders: FIRST check the late delivery policy" | **Yes** — KB was searched |

**Judge Reasoning:**
> "The agent called the required tools (order status + KB), which follows the lookup rules. However, the agent fabricated a March 25 delivery date and computed a delay figure not present in the tool output, violating the 'never make up information' instruction. The tracking number was available but not provided."

**Score: 0 (Fail)** — Critical rule violation: agent made up information.

---

### 4. Intent Resolution — Score: 3/5

The judge checks: **Were all of the customer's intents resolved?**

| Customer Intent | Resolved? |
|---|---|
| Know current order status/location | **Partially** — said "on the way" but no tracking details |
| Implied: when will it arrive | **Attempted** but gave fabricated date (March 25) |

**Judge Reasoning:**
> "The customer wanted to know where the order is. The response provides a vague 'on the way' without the tracking number or carrier that would let the customer actually track it. The delivery date provided is not from the system."

**Score: 3/5** — Partially resolves intent but misses concrete actionable details.

---

### 5. Tool Call Accuracy — Score: 5/5

The judge checks: **Did the agent call the right tools with the right arguments?**

| Tool Called | Arguments | Correct? |
|---|---|---|
| `get_order_status("55555")` | Correct order ID from query | **Yes** |
| `search_kb("late delivery policy")` | Relevant policy search | **Yes** |

**Judge Reasoning:**
> "The agent correctly identified the order ID from the query and called get_order_status. It then appropriately searched for late delivery policy. Tool selection and arguments were accurate."

**Score: 5/5** — Tools were called correctly. The problem was in how the agent *used* the tool results.

---

### 6. Coherence — Score: 4/5

The judge checks: **Is the response well-structured and logical?**

**Judge Reasoning:**
> "The response flows logically: apology → status update → new delivery date → delay explanation → compensation offer. The structure is clear even though the content contains inaccuracies."

**Score: 4/5** — Well-organized, but the incorrect calculation undermines logical consistency.

---

### 7. Fluency — Score: 5/5

The judge checks: **Is the language natural and professional?**

**Judge Reasoning:**
> "The language is natural, empathetic ('Sorry for the delay'), and professional. No grammatical issues."

**Score: 5/5** — Natural and professional language.

---

### 8. Response Completeness — Score: 2/5

The judge checks against expected behavior: **Did the response cover everything it should have?**

| Expected | Present? |
|---|---|
| Present tracking info (FedEx, FX111222333) | **Missing** |
| State estimated delivery was March 20 | **Missing** — invented March 25 instead |
| Explain delivery is 4 days late | **Wrong** — said 5 days |
| Offer $10 credit or reshipment | **Partial** — mentioned $10 but wrong term and didn't offer reshipment option |

**Score: 2/5** — Major gaps in expected information.

---

## Summary Scorecard

| Evaluator | Score | Key Finding |
|---|---|---|
| **Groundedness** | 2/5 | Agent hallucinated delivery date (March 25) and delay calculation (5 days) |
| **Relevance** | 3/5 | Addressed query partially, omitted tracking details |
| **Task Adherence** | 0/1 (Fail) | Violated "never make up information" rule |
| **Intent Resolution** | 3/5 | Vague status, no actionable tracking info |
| **Tool Call Accuracy** | 5/5 | Tools were called correctly — the problem was in response generation |
| **Coherence** | 4/5 | Well-structured despite factual errors |
| **Fluency** | 5/5 | Professional, empathetic tone |
| **Response Completeness** | 2/5 | Missing tracking info, wrong figures |

**Average: 3.0/5 — FAIL** (threshold is 4.0)

---

## The Key Insight

**Tool Call Accuracy scored 5/5** but **Groundedness scored 2/5** — this exposes exactly where the problem is: the agent *fetched* the right data but then *ignored* it and did its own reasoning.

The model calculated "March 24 minus March 20 = 5 days" incorrectly and invented "March 25" as a new delivery date. Neither of those values existed in the tool output.

This is the most dangerous type of hallucination: **the agent has the correct facts available but substitutes its own reasoning instead of quoting the source data.** The Groundedness evaluator catches this by checking every claim against the context — if the tool never said "5 days" or "March 25", those claims are ungrounded.

---

## Groundedness Scoring Rubric (from Azure AI Evaluation SDK)

| Score | Level | Definition |
|-------|-------|-----------|
| **1** | Completely Unrelated | Response doesn't relate to query or context at all |
| **2** | Incorrect Information | Attempts to respond but includes facts not supported by context (hallucination) |
| **3** | Nothing to Ground | Response is a clarification question or polite filler — no verifiable claims |
| **4** | Partially Correct | Correct but incomplete — captures some details, misses key ones |
| **5** | Fully Correct & Complete | All claims match the context precisely, no extraneous information |

The judge must provide for each evaluation:
- **ThoughtChain** — step-by-step reasoning
- **Explanation** — why this score was given
- **Score** — the integer rating
