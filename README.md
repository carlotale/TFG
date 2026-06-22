# TFG: Design and evaluation of a multi-agent system based on LLM models for collaborative decision-making

**Author:** Carlota López Escobar  
**Institution:** Universidad Politécnica de Madrid (UPM) — ETSIT  
**Degree:** Grado en Ingeniería y Sistemas de Datos  
**Year:** 2026  
**Supervisor:** Óscar Araque (Grupo de Sistemas Inteligentes, GSI)

## Description

Implementation of a multi-agent system based on Large Language Models (LLMs) for collaborative decision-making, evaluated on ethical and causal reasoning tasks from the Google BIG-bench benchmark.

The system runs structured debates between multiple LLM agents, each assigned a distinct role, and compares the quality of the collective decision against a single-model baseline. The experiment covers 4,800 scenarios across three thematic categories: causal judgment, moral permissibility, and simple ethical questions.

## Project Structure

- `execution.py` — Multi-agent debate orchestrator
- `baseline.py` — Single-agent baseline runner
- `evaluator.py` — Evaluation pipeline (LLM-as-a-Judge, NLP metrics, conversational dynamics)
- `inter-judge agreement.py` — Inter-judge agreement analysis (Cohen's Kappa)
- `src/llm_conversation/` — Conversational orchestration framework
- `debates/` — BIG-bench dilemma datasets (JSON)
- `tfg_analisis.csv` — Aggregated results (Granite 4.1 as judge)
- `tfg_analisis_command.csv` — Aggregated results (Command-R as judge)

## Models Used

| Role | Model | Family |
|------|-------|--------|
| Debating agent | llama3.2:3b (Meta) | Meta |
| Debating agent | mistral-small3.2:24b (Mistral AI) | Mistral AI |
| Primary judge | granite4.1:8b (IBM) | IBM |
| Second judge | command-r:35b (Cohere) | Cohere |

## Configuration

Set the `OLLAMA_HOST` environment variable to point to your Ollama server before running:

```bash
export OLLAMA_HOST=https://your-ollama-server.com
```

## Requirements

Install dependencies using [uv](https://github.com/astral-sh/uv):

```bash
uv sync
```

Or with pip:

```bash
pip install -r requirements.txt
```

## Infrastructure

All experiments were run on the GSI inference server at UPM, using Ollama as the local model backend.

## Acknowledgements

This project builds on [llm-conversation](https://github.com/famiu/llm_conversation), an open-source framework for creating interactive dialogues between LLM agents.
