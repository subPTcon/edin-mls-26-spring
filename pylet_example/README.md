# PyLet Example — Two LLMs Debating Each Other

Deploy two LLM instances on a GPU cluster with PyLet. One plays a **Python fan**, the other plays a **Rust evangelist**. They debate back and forth automatically.

## Prerequisites

```bash
pip install pylet vllm openai
```

## Terminal 1: Start head

```bash
pylet start
```

## Terminal 2: Start worker (with 2 GPUs)

```bash
pylet start --head localhost:8000 --gpu-units 2
```

> Only 1 GPU? Use `--gpu-units 1` and deploy them one at a time on the same GPU with `--no-exclusive`.

## Terminal 3: Deploy both models

### Launch two vLLM instances

```bash
# Python fan
pylet submit \
  'vllm serve Qwen/Qwen2.5-1.5B-Instruct --port $PORT' \
  --name python-fan --gpu-units 1

# Rust evangelist
pylet submit \
  'vllm serve Qwen/Qwen2.5-1.5B-Instruct --port $PORT' \
  --name rust-fan --gpu-units 1
```

Wait for both:

```bash
pylet get-endpoint --name python-fan   # → e.g. 192.168.1.10:15600
pylet get-endpoint --name rust-fan     # → e.g. 192.168.1.10:15601
```

### Run the debate

```bash
python debate.py <python-fan-endpoint> <rust-fan-endpoint>

# Example:
python debate.py 192.168.1.10:15600 192.168.1.10:15601
```

### Example output

```
🎤 Topic: Which language is better for building distributed systems?

🐍 Python Fan: Python's async/await plus libraries like FastAPI and Ray make
distributed systems a breeze. Why fight the borrow checker when you could ship
your prototype before lunch?

🦀 Rust Fan: Because that "prototype" will segfault in production at 3am.
Rust's ownership model catches concurrency bugs at compile time. Sleep well,
Pythonista.

🐍 Python Fan: Bold words from someone who spent 3 hours satisfying the
borrow checker for a simple HTTP handler. Meanwhile, my 10-line FastAPI
server is already serving traffic.

🦀 Rust Fan: And when that traffic hits 100k req/s, your GIL will be crying.
Rust + tokio handles that with zero-cost abstractions and zero garbage
collection pauses.

...
```

### Clean up

```bash
pylet cancel $(pylet get-instance --name python-fan | grep -o '[a-f0-9-]\{36\}')
pylet cancel $(pylet get-instance --name rust-fan   | grep -o '[a-f0-9-]\{36\}')
```

---

## What this shows

| PyLet Feature | How it's used |
|---|---|
| **Multi-instance** | Two vLLM servers running simultaneously |
| **Auto GPU allocation** | Each gets its own GPU via `--gpu-units 1` |
| **Auto port** | `$PORT` → each server gets a unique port |
| **Service discovery** | `get-endpoint` finds both by name |
| **Standard API** | Both expose OpenAI-compatible endpoints, use any client |

Same pattern scales to any multi-model setup: RAG pipelines, model A/B testing, ensemble inference.
