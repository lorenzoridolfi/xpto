# Model Selection Scenarios for Autogen Agents

## Introduction
This document outlines two recommended OpenAI model configurations—**High-Quality** and **Good Cost/Benefit**—for a three-agent (Generator → Critic → Reviewer) pipeline in Microsoft Autogen, balancing performance (generation diversity, reasoning precision, refinement fidelity) against cost and latency.

---

## 1. High-Quality Scenario
- **Generator Agent:** GPT-4o, Temperature 0.7 [2]  
- **Critic Agent:** o3, Temperature 0.0 [1]  
- **Reviewer Agent:** GPT-4o, Temperature 0.2 [2]  

## 2. Good Cost/Benefit Scenario
- **Generator Agent:** o4-mini, Temperature 0.7 [1]  
- **Critic Agent:** o3-mini, Temperature 0.0 [3]  
- **Reviewer Agent:** o4-mini, Temperature 0.2 [1]  

---

## References

1. [Introducing OpenAI o3 and o4-mini](https://openai.com/index/introducing-o3-and-o4-mini/)  
2. [Hello GPT-4o](https://openai.com/index/hello-gpt-4o/)  
3. [Introducing OpenAI o3-mini](https://help.openai.com/en/articles/9624314-model-release-notes)  
