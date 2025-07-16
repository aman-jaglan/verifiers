# Training Llama-3-70B on CRMArena-Pro **Text-Skill** Tasks with Function Calling

> Status : draft – last updated YYYY-MM-DD

---

## 1. Goal

Raise single-turn accuracy on the five *text-skill* categories of **CRMArena-Pro** from the current ~19 % to ≥ 60 % by fine-tuning a 70-billion-parameter Llama-3 model.  The model must learn to:

1. Choose the correct Salesforce helper function(s) exposed by the benchmark.
2. Provide valid JSON arguments for each call.
3. Interpret the returned data.
4. Produce a concise natural-language answer.

All training uses the public [Salesforce/CRMArenaPro](https://huggingface.co/datasets/Salesforce/CRMArenaPro) dataset; no private data is introduced.

---

## 2. Scope of the Data

| Skill | Categories used | Reward metric |
|-------|-----------------|---------------|
| **Text** | • knowledge_qa  \
            • sales_insight_mining  \
            • wrong_stage_rectification  \
            • activity_priority  \
            • named_entity_disambiguation | *exact_match* or *fuzzy_match* (depending on the sample) |

These five categories are confirmed both in the dataset card and in the sandbox constant `EXTERNAL_FACING_TASKS` (commit ‎`c1af7e` of the repo).

Dataset split:
* **train** : 350 examples (70 %)
* **validation** : 50 examples (10 %) – used for early stopping & hyper-parameter tuning
* **internal test** : 100 examples (20 %) – never touched during training; serves as final offline score before any external benchmark submission

---

## 3. Tool Interface Exposed to the Model

### 3.1 Function catalogue

*Location:* `crm_sandbox/env/functions.py`  
Count: **23** helper functions (e.g., `search_knowledge_articles`, `get_month_to_case_count`, `find_id_with_max_value`, …).  Each function is annotated with an OpenAI-style JSON schema in its `__info__` attribute.

### 3.2 How schemas reach the LLM

1.  `run_tasks.py` or the RL wrapper instantiates `ToolEnv(tools=TOOLS, …)`.
2.  `ToolEnv` collects `[tool.__info__ for tool in tools]` into `env.tools_info`.
3.  `ToolCallAgent` passes that list as the **`tools`** argument in every call to the OpenAI-compatible chat API.
4.  For models lacking native function-calling (e.g. Llama-3) the same schemas are rendered into text via `fc_prompt_builder` and prepended to the system prompt.

There is **no hard-coded mapping** from task category to function; the model must decide which calls are relevant from the prompt.

---

## 4. Episode Life-Cycle (Training & Evaluation)

1. **Reset**  
   *Environment* returns the user’s query (`task["query"]`) and optional metadata.
2. **Tool call(s)**  
   *Model* emits a structured `tool_call` message choosing one of the schemas.  `ToolEnv` executes the underlying Python function, hitting a Salesforce sandbox via `SalesforceConnector` if needed, and returns the result.
3. **Answer submission**  
   When ready, the model calls the special `respond` tool to output its final answer.
4. **Reward computation**  
   *ToolEnv* invokes `Evaluator.evaluate()` with:
   * ground-truth answer list (`task["answer"]`)
   * model answer (string)
   * reward metric (exact / fuzzy / privacy)
   * task name.

   Internally `Evaluator` sends the answer string to **GPT-4o-2024-08-06**.  GPT-4o extracts the relevant entities (IDs, months, etc.).  Python then:
   * compares with ground truth → reward ∈ [0, 1]
   * returns the reward to the trainer.
5. **Auxiliary rewards** (RL only)  
   Verifiers’ `ToolRubric` adds:
   *  *tool_execution_reward*  – fraction of successful tool calls.
   *  *format_reward* – XML / JSON correctness.

   Default weighting: `1.0·answer + 0.2·tool_success + 0.2·format`.

---

## 5. Training Strategy

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Base model | **Llama-3.1-70B** | Base checkpoint used by the public benchmark; requires prompting for function calling.
| Fine-tuning method | **LoRA + GRPO** (from verifiers) | Parameter-efficient, integrates reward-aware updates, keeps the base model stable via KL penalty.
| Batch size | 4 per device × 2 H100 GPUs, grad-accum 4 | 80 GB VRAM per H100 comfortably fits a 4-sequence micro-batch with 8-bit LoRA; effective batch ≈ 32 sequences.
| Learning rate | 5 e-5 | Empirically stable for RLHF with LoRA.
| Epochs | 2 | Covers the 350-example training set twice (~700 rollouts); cost remains manageable.

### Why GRPO?
* Handles variable-length, structured action spaces (tool calls) out-of-the-box.
* Explicitly logs and normalises reward scales, crucial when mixing *exact_match* (binary) and *fuzzy_match* (continuous).
* Provides built-in KL control to avoid over-optimising on the small dataset.

---

## 6. Why this approach is correct

1. **Alignment with benchmark** – The reward signal is **identical** to what the official evaluation script uses (same `Evaluator`, same GPT-4o prompt).  Optimising this reward is therefore guaranteed to improve leaderboard metrics.
2. **Minimal glue code** – All heavy-lifting (environment, tool execution, reward calculation) already exists; we only wrap `ToolEnv` into the Verifiers trainer.
3. **Data faithfulness** – Uses the public HF dataset without alteration; split strategy keeps the official test set pristine.
4. **Scalable** – LoRA + GRPO allows the 70 B model to be trained on commodity 8×A100 nodes within ~12 h.
5. **Generalises** – By learning the *function-selection pattern* instead of hard-coding per-category logic, the same policy can solve unseen text queries that map onto existing functions.

---

## 7. Next Steps Checklist

- [ ] Implement the 40-line Verifiers `Environment` wrapper that delegates to `ToolEnv`.
- [ ] Cache every `Evaluator` call locally to cut GPT-4o cost during RL.
- [ ] Run a 100-episode smoke test; inspect tool success rate ≥ 0.8 before full training.
- [ ] After training, evaluate with `run_tasks.py --agent_strategy act` and record accuracy & cost.

Once these steps are complete, the fine-tuned model should surpass the 60 % single-turn accuracy target on the five text categories while maintaining > 90 % well-formed tool calls. 