# ialacol-code  (localai for code)

🦄 Self hosted code completion inference server on Kubernetes cluster

A sister project of [ialacol](https://github.com/chenhunghan/ialacol) but focuses on code completion/inference

## Introduction

A helm chart for deploying an endpoint for [huggingface-vscode](https://github.com/huggingface/huggingface-vscode)

Mostly inspired by [huggingface-vscode-endpoint-server](https://github.com/LucienShui/huggingface-vscode-endpoint-server)

## Quick Start

To quickly get started with ialacol-code, follow the steps below:

```sh
helm repo add ialacol-code https://chenhunghan.github.io/ialacol-code
helm repo update
helm install replitcode3b ialacol-code/ialacol-code
```

By defaults, it will deploy [Replit's Code-v1-3B](https://huggingface.co/replit/replit-code-v1-3b) model.

Port-forward

```sh
kubectl port-forward svc/replitcode3b 80:8000
```

Chat with the default model `replit-code-v1-3b` using `curl`

```sh
curl -X POST \
     -H 'Content-Type: application/json' \
     -d '{"inputs": "def hello_world(): "}' \
     http://localhost:8000/v1/code/completions
```

Install [HF Code Autocomplete](https://marketplace.visualstudio.com/items?itemName=HuggingFace.huggingface-vscode)
Fill <http://localhost:8000/v1/code/completions> into Hugging Face Code > Model ID or Endpoint in VSCode.

## Development

```sh
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
pip freeze > requirements.txt
```
