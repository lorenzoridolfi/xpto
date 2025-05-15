# Multi-Agent Research Summary System

This document provides an in-depth overview of the AutoGen‑powered multi‑agent program, outlining its architecture, main components, configuration files, runtime workflow, and key design decisions.

---

## 1. Overview

A command‑line Python application orchestrates a group of specialized AI agents (using Microsoft AutoGen) to:

1. **Ingest** domain‑specific files according to a manifest.
2. **Synthesize** a concise research summary from the ingested content.
3. **Verify** both factual completeness (`InformationVerifierAgent`) and writing quality (`TextQualityAgent`) in iterative rounds.
4. **Collect** explicit user feedback via terminal input.
5. **Analyze** the entire process with a `RootCauseAnalyzerAgent`, producing a JSON report of actions, decisions, and deviations from expected behavior.

---

## 2. Configuration

### 2.1 `agent_config.json`

* **Task description**: Defines the high‑level goal.
* **Hierarchy**: Lists agents in order of execution responsibility.
* **Agents**: Per‑agent metadata including `description` and `system_message` prompts.

### 2.2 `agents_configuration.json`

* A copy of `agent_config.json`, provided as input to the RootCauseAnalyzerAgent to cross‑reference actual behavior against intended roles.

*Default configs are embedded as `DEFAULT_CONFIG` in `main.py` and written at startup.*

---

## 3. Agent Roles

| Agent                        | Responsibility                                                                                              |
| ---------------------------- | ----------------------------------------------------------------------------------------------------------- |
| **CoordinatorAgent**         | Oversees workflow. Decides which files to ingest next based on metadata and files already read.             |
| **FileReaderAgent**          | Reads one or more files from disk, aggregates content, and logs each file access for traceability.          |
| **WriterAgent**              | Synthesizes aggregated contents into a coherent, concise research summary, preserving logical flow.         |
| **InformationVerifierAgent** | Validates factual completeness and accuracy. Requests missing data or signals approval via `TERMINATE`.     |
| **TextQualityAgent**         | Evaluates style, clarity, tone, and readability. Suggests improvements or signals approval via `TERMINATE`. |
| **User**                     | Provides final feedback through terminal input.                                                             |
| **RootCauseAnalyzerAgent**   | Analyzes full event logs and configuration JSON to produce a structured root cause report in JSON.          |

---

## 4. Runtime Workflow

1. **Initialization**

   * Write `agent_config.json` and `agents_configuration.json` from `DEFAULT_CONFIG`.
   * Load configuration and instantiate agents dynamically.

2. **Iterative Rounds (up to `MAX_ROUNDS`)**
   a. CoordinatorAgent receives the list of already‑read files and replies with one or more filenames (comma‑separated) or `NO_FILE`.
   b. FileReaderAgent reads the specified files and returns their contents.
   c. WriterAgent composes a new draft summary.
   d. InformationVerifierAgent checks factual completeness and returns `TERMINATE` if OK.
   e. TextQualityAgent checks prose quality and returns `TERMINATE` if OK.
   f. If both verifiers approve (`TERMINATE`), exit the loop early; otherwise, continue until round cap reached.

3. **User Feedback**

   * Display the final summary and prompt the user via `input()` for feedback.

4. **Root Cause Analysis**

   * Invoke RootCauseAnalyzerAgent with:

     * The full configuration JSON (`agents_configuration.json`).
     * The user’s feedback.
     * The recorded `ACTION_LOG`.
   * RootCauseAnalyzerAgent outputs diagnostic insights.
   * Save combined analysis and raw event data (`ROOT_CAUSE_DATA`) to `root_cause.json`.

---

## 5. Logging & Traceability

* **Global logs**:

  * `ACTION_LOG`: sequence of summaries, agent replies, and user feedback.
  * `ROOT_CAUSE_DATA`: structured events captured via `log_event()` hooks (timestamps, inputs, outputs).
  * `FILE_LOG`: records which files were read by FileReaderAgent.

* **Console & File**: All key steps are logged via `logging` to both `stdout` and `agent_system.log`.

---

## 6. Main Design Decisions

### 6.1 Loop vs. Predefined Chat Framework

* **Loop Approach**:

  * Explicit control over rounds and termination conditions.
  * Easy to enforce `MAX_ROUNDS` and dual‑approval logic.
  * Simplifies injection of user input between loops.

* **Alternative**:

  * Use AutoGen’s built‑in `RoundRobinGroupChat` with complex termination rules.
  * **Trade‑off**: less code but harder to express dual‑agent checks and user‑in‑loop prompts cleanly.

### 6.2 Config‑Driven Agent Instantiation

* Embedding `DEFAULT_CONFIG` allows:

  * Centralized tuning of prompts and descriptions.
  * Runtime generation of both config files, ensuring consistency.

### 6.3 Dual‑Verifier Strategy

* Splitting the original Critic into two specialized agents ensures:

  * Clear separation of concerns: factual vs. stylistic checks.
  * More granular feedback and targeted improvements in each iteration.

### 6.4 JSON‑Based Root Cause Analysis

* Capturing raw event data plus the desired configuration schema:

  * Enables automated “expected vs. actual” consistency checks by the analyzer.
  * Results stored in a machine‑readable `root_cause.json` for post‑mortem tools or dashboards.

---

## 7. Alternatives & Extensions

* **Dynamic Hyperparameters**: adjust `MAX_ROUNDS` or dual‑approval thresholds based on file count.
* **Structured Prompts**: replace comma‑separated file lists with JSON arrays for stricter parsing.
* **Parallel Ingestion**: allow FileReaderAgent to read all remaining files in one go, then iterate fewer rounds.
* **GUI Feedback**: swap `input()` for a lightweight web form or TUI for richer user feedback.

---

## 8. Conclusion

This design balances flexibility, traceability, and clear separation of responsibilities across agents.  The loop‑based orchestration gives precise control over iteration count and approval logic, while the JSON‑driven configurations ensure maintainable, tunable prompts and structured root cause reporting.

Feel free to propose further refinements or to adapt components for your specific domain!
