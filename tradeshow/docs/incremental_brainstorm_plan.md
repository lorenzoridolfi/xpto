# Incremental Implementation Plan: Idea Brainstorm with Synthetic Users

## Overview
This document outlines a stepwise, modular approach to building an idea-brainstorming system using synthetic users and agent-based architecture. Each step builds on the previous, allowing for incremental development, testing, and extension.

---

## Step 1: User Questioning and Summarization Class

**Goal:**
Create a class that, given a list of users and a question, asks the question to each user (via an agent) and summarizes the responses.

**Key Features:**
- Accepts:
  - List of user objects (loaded from JSON)
  - A question (string)
  - (Optional) agent configuration
- For each user:
  - Uses an agent to generate the user's answer to the question
  - Can run these calls in parallel (e.g., ThreadPoolExecutor)
- Collects all answers
- Uses a summarization agent to produce a summary of the answers

**Incremental Steps:**
1. Load users from JSON.
2. Implement the class with a method: `ask_and_summarize(users, question) -> summary`
3. Implement parallel question-asking.
4. Integrate summarization agent.

**Class Scaffold:**
```python
from typing import List, Dict

class UserQuestionSummarizer:
    """
    Asks a question to a list of users (via agent) and summarizes the responses.
    """
    def __init__(self, agent, summarizer_agent=None):
        self.agent = agent
        self.summarizer_agent = summarizer_agent

    def ask_and_summarize(self, users: List[Dict], question: str) -> str:
        """
        Ask the question to each user in parallel and summarize the answers.
        """
        # 1. Ask question to each user (parallel)
        # 2. Collect answers
        # 3. Summarize answers using summarizer_agent
        pass
```

**Diagram:**
```
+-------------------+
|  User List (JSON) |
+-------------------+
          |
          v
+---------------------------+
| UserQuestionSummarizer    |
|  - ask_and_summarize()    |
+---------------------------+
          |
          v
+-------------------+
|  Agent Calls      |
+-------------------+
          |
          v
+-------------------+
|  Summarizer Agent |
+-------------------+
          |
          v
+-------------------+
|   Summary Output  |
+-------------------+
```

---

## Step 2: Batch Program for Multiple Questions and Segments

**Goal:**
Build a program that, given a list of questions, user segments, and a number of users per segment, orchestrates the questioning and summarization process.

**Key Features:**
- Accepts:
  - List of questions
  - List of segments (segment = user attribute, e.g., "students")
  - Number of users per segment per question
- For each (question, segment) pair:
  - Randomly select the specified number of users from the segment
  - Use the class from Step 1 to ask the question and summarize
- Collect and output all summaries

**Incremental Steps:**
1. Implement user segmentation logic.
2. Implement random selection of users per segment.
3. Loop over all (question, segment) pairs, invoking the Step 1 class.
4. Output or store the results.

**Class Scaffold:**
```python
from typing import List, Dict
import random

class BatchQuestionOrchestrator:
    """
    Orchestrates asking multiple questions to user segments and summarizes responses.
    """
    def __init__(self, users: List[Dict], summarizer: UserQuestionSummarizer):
        self.users = users
        self.summarizer = summarizer

    def run(self, questions: List[str], segments: List[str], users_per_segment: int) -> Dict:
        """
        For each (question, segment), select users and summarize answers.
        Returns a dict of {(question, segment): summary}
        """
        # 1. Segment users
        # 2. For each (question, segment):
        #    - Randomly select users
        #    - Call summarizer.ask_and_summarize()
        # 3. Collect and return summaries
        pass
```

**Diagram:**
```
+-------------------+
|  Questions List   |
+-------------------+
          |
+-------------------+
|  Segments List    |
+-------------------+
          |
+-------------------+
|  Users (JSON)     |
+-------------------+
          |
          v
+-------------------------------+
| BatchQuestionOrchestrator      |
|   - run()                     |
+-------------------------------+
          |
          v
+-------------------------------+
| UserQuestionSummarizer (Step1) |
+-------------------------------+
          |
          v
+-------------------+
|   Summaries       |
+-------------------+
```

---

## Step 3: Idea-to-Questions and Full Brainstorm Orchestration

**Goal:**
Create a class that, given an idea, number of questions, a segment, and number of users, generates questions from the idea, queries users, and summarizes the overall brainstorm.

**Key Features:**
- Accepts:
  - An idea (string)
  - Number of questions to generate
  - Segment to target
  - Number of users per question
- Uses an agent to generate the specified number of questions from the idea
- For each question:
  - Randomly select users from the segment
  - Use the Step 1 class to ask and summarize
- Uses an agent to produce an overall summary of all responses

**Incremental Steps:**
1. Implement question generation agent.
2. For each generated question, select users and use Step 1 class.
3. Collect all summaries.
4. Implement overall summarization agent.
5. Return or store the final summary.

**Class Scaffold:**
```python
from typing import List, Dict

class IdeaBrainstormer:
    """
    Orchestrates full brainstorm: idea -> questions -> user answers -> overall summary.
    """
    def __init__(self, question_agent, summarizer: UserQuestionSummarizer, overall_summarizer_agent=None):
        self.question_agent = question_agent
        self.summarizer = summarizer
        self.overall_summarizer_agent = overall_summarizer_agent

    def brainstorm(self, idea: str, num_questions: int, segment: str, users_per_question: int) -> str:
        """
        Generate questions from idea, ask users, and summarize.
        """
        # 1. Generate questions from idea
        # 2. For each question:
        #    - Select users from segment
        #    - Use summarizer.ask_and_summarize()
        # 3. Summarize all answers with overall_summarizer_agent
        pass
```

**Diagram:**
```
+-------------------+
|      Idea         |
+-------------------+
          |
          v
+---------------------------+
| Question Generation Agent |
+---------------------------+
          |
          v
+-------------------+
|  Questions List   |
+-------------------+
          |
          v
+-------------------------------+
| For each Question:             |
|   - Select Users from Segment  |
|   - UserQuestionSummarizer     |
+-------------------------------+
          |
          v
+-------------------+
|  All Summaries    |
+-------------------+
          |
          v
+---------------------------+
| Overall Summarizer Agent  |
+---------------------------+
          |
          v
+-------------------+
|  Final Summary    |
+-------------------+
```

---

## Validation & Suggestions
- The plan is modular and each step can be tested independently.
- Use ThreadPoolExecutor for parallel agent calls.
- The design is extensible for new segments, question types, or summarization strategies.
- Add robust error handling for agent failures, empty user lists, etc.
- Document each class and method for clarity and future maintenance.

---

## Example Class/Method Names

- `UserQuestionSummarizer` (Step 1)
  - `ask_and_summarize(users, question) -> summary`
- `BatchQuestionOrchestrator` (Step 2)
  - `run(questions, segments, users_per_segment)`
- `IdeaBrainstormer` (Step 3)
  - `brainstorm(idea, num_questions, segment, users_per_question) -> overall_summary`

---

## Next Steps
- Start with Step 1 and test with a small set of users and a single question.
- Gradually add complexity and parallelism.
- Integrate agent calls and summarization as you go. 