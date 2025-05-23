# Questionnaire and Multi-Answer Program: Architecture & Design

## Overview

This document describes the architecture and design for a modular program that:
- Receives an idea as input (optional upper layer).
- Transforms the idea into a fixed number of questions using an LLM or rules (upper layer).
- Loads pre-generated synthetic users from a JSON file.
- Randomly selects a subset of users from each segment.
- Asks each question to the selected users (sequentially or in parallel).
- Summarizes the answers for each segment/question.
- Supports both sequential and parallel execution via a simple flag.

---

## Modular Architecture

The system is designed in layers, each of which can be developed, tested, and reused independently:

1. **Idea-to-Questions Layer (Optional Upper Layer):**
    - Receives an idea (text).
    - Uses an LLM or rules to generate a fixed number of questions to validate the idea.
    - Output: List of questions.

2. **Questionnaire/Answering Layer (Bottom Layer):**
    - Receives a list of questions for each segment.
    - Loads and selects users, asks questions, collects answers, and summarizes per segment/question.
    - This layer can be used standalone or as a component in a larger pipeline.

This modularity allows you to:
- Swap out the question generation logic without changing the answering/summarization logic.
- Use the answering/summarization layer for any set of questions, not just those derived from an idea.
- Compose more complex workflows by chaining or nesting these modules.

---

## Workflow

1. **(Optional) Generate Questions from Idea:**
    - Input: Idea (string).
    - Output: List of questions (using LLM or rules).

2. **Load Users:**
    - Read all synthetic users from a JSON file.
    - Group users by their segment label for efficient access.

3. **For Each Segment:**
    - Randomly select N users (e.g., 3) from the segment's user pool.
    - For each question assigned to the segment:
        - Ask the question to all selected users (in parallel or sequentially).
        - Collect all answers.
        - Summarize the answers for that segment/question.

4. **Output:**
    - Store or print the answers and summaries for each segment/question.
    - Optionally, log all steps and timings for traceability.

---

## Design Rationale

- **Separation of Concerns:**
    - Each layer (idea-to-questions, answering, summarization) is a distinct module.
- **Reusability:**
    - The answering/summarization layer can be reused for any set of questions.
- **Parallelism:**
    - The question-asking step can be run in parallel for efficiency, especially when using LLMs or other slow operations.
    - A flag allows easy switching between sequential and parallel execution for testing or resource management.
- **Extensibility:**
    - Easy to add more segments, questions, or change the number of users per segment.
    - Summarization logic can be swapped out for more advanced LLM-based aggregation if needed.
    - Additional layers (e.g., global summarization, feedback) can be composed as needed.

---

## Parallelism Options

- **Sequential Mode:**
    - Questions are asked to users one at a time (useful for debugging or when parallelism is not needed).
- **Parallel Mode:**
    - Questions are asked to all selected users simultaneously using Python's `ThreadPoolExecutor` or `asyncio`.
    - Greatly reduces total runtime when user answers require LLM calls or other I/O-bound operations.
- **Configurable:**
    - A simple flag (e.g., `parallel=True`) controls the execution mode.

---

## Example Data Flow

```
Input:
- (Optional) idea (string)
- synthetic_users.json (list of user dicts, each with a segment label)
- segments_questions = {
    "Poupadores": ["How do you save money?", ...],
    "Endividados": ["What are your main financial challenges?", ...],
    ...
  }

Processing:
- If idea is provided:
    - Generate questions from idea (LLM or rules)
    - Use same questions for all segments, or customize per segment
- For each segment:
    - Randomly select 3 users
    - For each question:
        - Ask all 3 users (sequentially or in parallel)
        - Collect answers
        - Summarize answers

Output:
- For each segment/question:
    - List of user answers
    - Summary of answers
```

---

## Example Pseudocode

```python
def generate_questions_from_idea(idea, n_questions=5):
    # Use LLM or rules to generate questions
    prompt = f"Generate {n_questions} questions to validate this idea: {idea}"
    # ... LLM call ...
    return questions_list

def process_segments(segments_questions, users_by_segment, parallel=True):
    results = {}
    for segment, questions in segments_questions.items():
        users = users_by_segment.get(segment, [])
        if len(users) < 3:
            print(f"Not enough users for segment {segment}")
            continue
        selected_users = random.sample(users, 3)
        results[segment] = {}
        for question in questions:
            if parallel:
                with ThreadPoolExecutor(max_workers=3) as executor:
                    answers = list(executor.map(lambda user: ask_user(user, question), selected_users))
            else:
                answers = [ask_user(user, question) for user in selected_users]
            summary = summarize_answers(question, answers)
            results[segment][question] = {
                'answers': answers,
                'summary': summary
            }
    return results

def main_workflow(idea, users_by_segment, segments, n_questions=5, parallel=True):
    questions = generate_questions_from_idea(idea, n_questions)
    segments_questions = {segment: questions for segment in segments}
    return process_segments(segments_questions, users_by_segment, parallel=parallel)
```

---

## Extensibility

- **Change number of users per segment:** Adjust the sample size in the code.
- **Add more segments/questions:** Update the `segments_questions` input or the question generation logic.
- **Swap out answer/summarization logic:** Integrate LLMs or other advanced methods as needed.
- **Traceability:** Add logging or tracing at each step for auditability and debugging.
- **Compose additional layers:** Add global summarization, feedback, or other modules as needed.

---

## Conclusion

This architecture is:
- **Modular:** Each layer is independent and easily testable.
- **Composable:** You can build more complex workflows by chaining or nesting modules.
- **Scalable:** Parallelism can be enabled for speed, or disabled for simplicity.
- **Flexible:** Works with any number of segments, users, or questions, and can be extended with new layers.
- **Ready for production:** Easily integrates with LLMs, logging, and traceability features already present in your system. 