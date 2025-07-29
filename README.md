# LLM Evaluation Pipeline

## Overview
This project implements a pipeline to evaluate LLMs using BERTScore, containerized with Docker and automated via GitHub Actions. It evaluates prompts against reference answers, capturing success/failure signals and edge cases.

## Requirements
- Docker
- Python 3.9
- GitHub account (for CI)
- Optional: Kubernetes (e.g., Minikube)

## Local Execution
```bash
docker build -t llm-evaluation .
mkdir output
docker run -v $(pwd)/output:/app/output llm-evaluation
