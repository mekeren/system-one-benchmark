# System One Decision Engine & 3-Way Benchmark Arena

An ultra-low latency, CPU-friendly micro-decision routing engine and benchmarking arena inspired by **TypeSafe AI Jev** primitives. 

This repository provides an end-to-end framework to evaluate and route categorical decisions (`choice`, `score`, `noul`) across three distinct decision engines:
1. **Rule Engine (CPU Deterministic):** Ultra-fast regex & keyword scoring (< 5 ms).
2. **Local SLM Engine (CPU Neural):** Quantized Small Language Model (`Qwen2.5-1.5B-Instruct-GGUF`) running in-process via `llama-cpp-python` (< 400 ms on standard CPU).
3. **Cloud LLM Engine (External Cloud):** Any OpenAI-compatible provider (Groq, vLLM, Ollama, OpenAI) for complex reasoning and consensus comparison.

---

## 🏛️ Architecture Overview

The system is architected as an isolated, containerized microservices suite with an API gateway and interactive web arena:

```
                  ┌────────────────────────────────────────┐
                  │    Decision UI Gateway (Port: 8150)    │
                  │   Interactive Arena & Dynamic Config   │
                  └──────┬──────────────┬────────────┬─────┘
                         │              │            │
            ┌────────────┴───┐   ┌──────┴─────┐   ┌──┴─────────────┐
            │  Rule Engine   │   │ SLM Engine │   │  Cloud Engine  │
            │  (Port: 8001)  │   │(Port: 8002)│   │  (Port: 8003)  │
            │  Regex / CPU   │   │ Qwen-1.5B  │   │ Any OpenAI API │
            └────────────────┘   └────────────┘   └────────────────┘
```

- **TypeSafe Jev Protocol Compliance:** 100% compliant with TypeSafe AI System One decision endpoints (`/v1/systemone`, `/v1/models`).
- **Dynamic Decision Choices:** Add, remove, and configure target choices and criteria on the fly directly from the Web UI or JSON request body.
- **Zero Cloud Dependence:** Works 100% offline and on-premise without GPU requirements.

---

## 🚀 Quick Start with Docker Compose

To build and run all 4 microservices simultaneously:

```bash
docker compose up -d
```

Service mapping:
| Service | Container Name | Port | Description |
|---|---|---|---|
| **UI Gateway** | `decision-ui-gateway` | `8150` | Interactive Web Arena & Aggregator API |
| **Rule Engine** | `decision-rule-engine` | `8001` | Fast deterministic rule-based router |
| **SLM Engine** | `decision-slm-engine` | `8002` | Local GGUF Small Language Model engine |
| **Cloud Engine** | `decision-cloud-engine` | `8003` | Proxy & converter for external LLMs |

Open your browser at: **`http://localhost:8150`**

---

## 💻 Manual / Standalone Setup

### Prerequisites
- Python 3.10+
- GGUF model placed in `models/` (e.g. `qwen2.5-1.5b-instruct-q4_k_m.gguf`)

### Installation
```bash
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### Running Standalone
```bash
python server.py
# or on Windows:
start_server.bat
```

---

## 📡 API Specification

### 1. Unified 3-Way Benchmark (`/v1/compare`)
Executes all three decision engines concurrently and computes latency, agreement, and confidence.

```http
POST http://localhost:8150/v1/compare
Content-Type: application/json
```

**Request Payload:**
```json
{
  "message": "Production deployment failed during release pipeline",
  "api_key": "",
  "base_url": "https://api.groq.com/openai/v1",
  "model": "llama-3.3-70b-versatile",
  "choices": {
    "agent-a": "Backup, snapshot, restore operations",
    "agent-b": "CI/CD pipelines, release management, production deployment",
    "agent-c": "Agile board, task management, sprint tracking",
    "unclear_fallback": "General inquiries or ambiguous tasks"
  }
}
```

**Response:**
```json
{
  "message": "Production deployment failed during release pipeline",
  "consensus": {
    "all_agree": true,
    "winner_speed": "Engine 1 (Kural Motoru)"
  },
  "engine_1_rule": {
    "engine": "System One Engine 1 (Rule Engine)",
    "choice": "agent-b",
    "confidence": 0.95,
    "latency_ms": 1.2,
    "reason": "Regex kuralı eşleşti: 'deploy' -> agent-b"
  },
  "engine_2_slm": {
    "engine": "System One Engine 2 (Lokal SLM)",
    "choice": "agent-b",
    "confidence": 0.9912,
    "latency_ms": 320.5,
    "reason": "SLM logit & kalibre olasılık dağılımı"
  },
  "engine_3_cloud": {
    "engine": "System One Engine 3 (Bulut LLM)",
    "choice": "agent-b",
    "confidence": 0.999,
    "latency_ms": 580.1,
    "reason": "Cloud LLM intent extraction"
  }
}
```

### 2. TypeSafe Jev Standard Endpoint (`/v1/systemone`)

```http
POST http://localhost:8150/v1/systemone
Content-Type: application/json
```

**Request:**
```json
{
  "state": {
    "user_message": "Can you check why last night's database backup failed?"
  },
  "model": "system-one-cpu",
  "questions": {
    "target_agent": {
      "type": "choice",
      "instructions": "Select the best matching service agent",
      "criteria": {
        "agent-a": "Backup, disaster recovery and snapshots",
        "agent-b": "Deployment and infrastructure pipelines",
        "agent-c": "Project management and tasks"
      }
    }
  }
}
```

**Response:**
```json
{
  "model": "system-one-cpu-v1.0",
  "answers": {
    "target_agent": {
      "type": "choice",
      "choice": "agent-a",
      "confidence": 0.985,
      "probabilities": {
        "agent-a": 0.985,
        "agent-b": 0.010,
        "agent-c": 0.005
      }
    }
  },
  "latency_ms": 2.1
}
```

---

## ⚙️ Environment Configuration

You can customize service endpoints via environment variables:

| Variable | Default | Description |
|---|---|---|
| `RULE_SERVICE_URL` | `http://localhost:8001` | Rule Engine microservice URL |
| `SLM_SERVICE_URL` | `http://localhost:8002` | Local SLM microservice URL |
| `CLOUD_SERVICE_URL` | `http://localhost:8003` | Cloud LLM microservice URL |
| `SLM_MODEL_PATH` | `models/qwen2.5-1.5b-instruct-q4_k_m.gguf` | Path to local GGUF model |
| `PORT` | `8150` | Port for UI Gateway |

---

## 🔒 Security & Privacy

- **No Hardcoded Keys:** All cloud credentials are submitted via request payload or kept in local browser memory (`localStorage`).
- **GGUF Model Ignored:** Weights (`*.gguf`) are excluded in `.gitignore` to prevent large binary leaks.
- **Local Isolation:** Rule and SLM engines process sensitive data purely in memory on localhost/private Docker network.

---

## 📄 License
MIT License. Free for research, prototyping, and production micro-decision routing.
