import time
import os
import psutil
import re
from typing import Any
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="System One Engine 1 - Rule CPU Service", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def softmax(x: list[float], temp: float = 0.7) -> list[float]:
    e_x = np.exp((np.array(x) - np.max(x)) / max(temp, 1e-5))
    return (e_x / e_x.sum()).tolist()

def normalize_text(text: str) -> str:
    t = text.lower()
    mapping = {"ı": "i", "İ": "i", "ğ": "g", "ü": "u", "ş": "s", "ö": "o", "ç": "c"}
    for k, v in mapping.items():
        t = t.replace(k, v)
    return t

class PredictRequest(BaseModel):
    message: str
    temperature: float = 0.5
    criteria: dict[str, str] | None = None

@app.get("/healthz")
def healthz():
    proc = psutil.Process()
    mem_mb = proc.memory_info().rss / (1024 * 1024)
    return {
        "status": "healthy",
        "service": "decision-rule-engine",
        "memory_mb": round(mem_mb, 2),
        "cpu_percent": proc.cpu_percent()
    }

@app.post("/predict")
def predict(req: PredictRequest):
    t0 = time.perf_counter()
    text = req.message
    t_norm = normalize_text(text)

    if req.criteria and len(req.criteria) > 0:
        criteria = req.criteria
    else:
        criteria = {
            "agent-a": "Altyapı ve Veri Yönetimi (Sunucu, yedekleme ve kurtarma operasyonları)",
            "agent-b": "Dağıtım ve Operasyon (CI/CD pipeline, sürüm yayını ve servis izleme)",
            "agent-c": "Proje ve Görev Yönetimi (İş/task açma, atama ve durum takibi)",
            "unclear_fallback": "Kapsam dışı veya belirsiz talepler"
        }
    options = list(criteria.keys())
    scores = [0.1] * len(options)

    # Genel Ajan Şablonu (Agent A / B / C) hızlı kural eşleştirmesi
    if set(options) == {"agent-a", "agent-b", "agent-c", "unclear_fallback"}:
        idx_a = options.index("agent-a")
        idx_b = options.index("agent-b")
        idx_c = options.index("agent-c")
        # Agent B: Dağıtım, Sürüm, CI/CD
        if re.search(r"\b(release|deploy|pipeline|pod|k8s|kubernetes|helm|argocd|jenkins|docker|build|log|crashloop|surum|yayin)\b", t_norm):
            scores[idx_b] += 5.0
        # Agent A: Altyapı, Veri, Yedekleme
        if re.search(r"\b(yedek|yedekleme|backup|restore|snapshot|repository|repo|depolama|sunucu|veri)\b", t_norm):
            scores[idx_a] += 5.0
        # Agent C: Görev, İş, Sprint, Task
        if re.search(r"\b(task|gorev|jira|sprint|story|subtask|backlog|is|talep)\b", t_norm):
            scores[idx_c] += 5.0
    else:
        # Dinamik anahtar kelime eşleştirme (Kullanıcının tanımladığı her şema için)
        for i, opt in enumerate(options):
            desc = criteria.get(opt, "")
            desc_norm = normalize_text(desc)
            opt_norm = normalize_text(opt.replace("-", " ").replace("_", " "))

            # Seçenek anahtarından token eşleme
            for word in re.findall(r"\w{3,}", opt_norm):
                if word in t_norm:
                    scores[i] += 3.5

            # Açıklamadan anahtar kelime eşleme
            for word in re.findall(r"\w{3,}", desc_norm):
                if word in {"ile", "icin", "veya", "gibi", "gore", "olan", "ve", "bir", "her"}:
                    continue
                if word in t_norm:
                    scores[i] += 2.0

    probs = softmax(scores, req.temperature)
    max_idx = int(np.argmax(probs))
    conf = float(probs[max_idx])
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    reason = "Koddaki anahtar kelime eşleşmelerine göre hesaplandı."
    if "release" in t_norm and "task" in t_norm and set(options) == {"agent-a", "agent-b", "agent-c", "unclear_fallback"}:
        reason = "Mesajda hem 'release' hem 'task' anahtar kelimesi tespit edildi (kelime çakışması)."
    elif set(options) != {"agent-a", "agent-b", "agent-c", "unclear_fallback"}:
        reason = f"Dinamik seçenekler ('{options[max_idx]}') ile metin anahtar kelimeleri eşleştirildi."

    proc = psutil.Process()
    mem_mb = proc.memory_info().rss / (1024 * 1024)

    urgency_val = 1.8 if any(x in t_norm for x in ("fail", "hata", "coktu", "acil")) else (1.0 if any(x in t_norm for x in ("release", "deploy", "task", "surum", "gorev")) else 0.2)
    noul_val = 0.9 if any(x in t_norm for x in ("fail", "hata", "acil", "release")) else (0.7 if any(x in t_norm for x in ("task", "gorev")) else 0.1)

    return {
        "engine_type": "rule_based",
        "model_name": "rule-engine-cpu",
        "choice": options[max_idx],
        "confidence": round(conf, 4),
        "probabilities": {options[i]: round(probs[i], 4) for i in range(len(options))},
        "score": {
            "type": "score",
            "score": round(urgency_val, 2),
            "confidence": 0.85,
            "legend": {"0": "Düşük (Rutin)", "1": "Normal (Standart)", "2": "Kritik (P1/Bloker)"}
        },
        "noul": {
            "type": "noul",
            "noul": round(noul_val, 4),
            "statement": "requires_immediate_action (Acil müdahale gerekir mi?)"
        },
        "answers": {
            "target_agent": {
                "type": "choice",
                "choice": options[max_idx],
                "confidence": round(conf, 4),
                "probabilities": {options[i]: round(probs[i], 4) for i in range(len(options))}
            },
            "urgency": {
                "type": "score",
                "score": round(urgency_val, 2),
                "confidence": 0.85,
                "legend": {"0": "Düşük (Rutin)", "1": "Normal (Standart)", "2": "Kritik (P1/Bloker)"}
            },
            "requires_immediate_action": {
                "type": "noul",
                "noul": round(noul_val, 4),
                "statement": "Acil müdahale gerekir mi?"
            }
        },
        "latency_ms": round(elapsed_ms, 2),
        "reason": reason,
        "privacy": "Kurum İçi (Tamamen Çevrimdışı)",
        "hardware": "Standart CPU (AVX2)",
        "telemetry": {
            "memory_mb": round(mem_mb, 2),
            "cpu_percent": proc.cpu_percent()
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
