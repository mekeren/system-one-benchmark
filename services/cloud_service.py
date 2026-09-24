import time
import os
import psutil
import json
import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="System One Engine 3 - Cloud LLM Gateway", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CloudPredictRequest(BaseModel):
    message: str
    api_key: str = ""
    base_url: str = "https://api.groq.com/openai/v1"
    model: str = "openai/gpt-oss-120b"
    temperature: float = 0.1
    criteria: dict[str, str] | None = None

@app.get("/healthz")
def healthz():
    proc = psutil.Process()
    mem_mb = proc.memory_info().rss / (1024 * 1024)
    return {
        "status": "healthy",
        "service": "decision-cloud-engine",
        "memory_mb": round(mem_mb, 2),
        "cpu_percent": proc.cpu_percent()
    }

@app.post("/predict")
async def predict(req: CloudPredictRequest):
    proc = psutil.Process()
    api_key = req.api_key.strip()
    if not api_key:
        mem_mb = proc.memory_info().rss / (1024 * 1024)
        return {
            "engine_type": "cloud_llm",
            "model_name": req.model,
            "choice": "API Key Yok",
            "confidence": 0.0,
            "latency_ms": 0.0,
            "reason": "Harici API anahtari girilmedi. Ayarlar kartindan API Key tanimlayabilirsiniz.",
            "error": "API Key eksik",
            "telemetry": {
                "memory_mb": round(mem_mb, 2),
                "cpu_percent": proc.cpu_percent()
            }
        }

    t0 = time.perf_counter()
    if req.criteria and len(req.criteria) > 0:
        opts_desc = "\n".join([f"- '{k}': {v}" for k, v in req.criteria.items()])
        available_keys = list(req.criteria.keys())
    else:
        opts_desc = """- 'agent-a': Altyapi ve Veri Yonetimi (Sunucu, yedekleme ve kurtarma operasyonlari)
- 'agent-b': Dagitim ve Operasyon (CI/CD pipeline, surum yayini, servis ve konteyner izleme)
- 'agent-c': Proje ve Gorev Yonetimi (Is/task acma, atama, durum sorgulama ve planlama)
- 'unclear_fallback': Kapsam disi veya genel talepler"""
        available_keys = ["agent-a", "agent-b", "agent-c", "unclear_fallback"]

    system_prompt = f"""Sen gelen kurumsal mesajlari veya talepleri hedef uzmana/kategoriye yonlendiren ve System One (Choice, Score, Noul) kararlari ureten bir yapay zekasin.
Kullanilabilir Secenekler (Choice Categories):
{opts_desc}

Kullanicinin mesajini analiz et. SADECE asagidaki JSON formatinda sonuc dondur:
{{
  "agent": "secilen_secenek_id",
  "confidence": 0.95,
  "urgency_score": 1.1,
  "requires_immediate_action": 0.80,
  "reason": "secim nedeni kisa ozet"
}}"""

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": req.model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": req.message}
        ],
        "temperature": req.temperature,
        "response_format": {"type": "json_object"}
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(f"{req.base_url}/chat/completions", headers=headers, json=payload)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            if resp.status_code != 200:
                mem_mb = proc.memory_info().rss / (1024 * 1024)
                return {
                    "engine_type": "cloud_llm",
                    "model_name": req.model,
                    "choice": "API Hatası",
                    "confidence": 0.0,
                    "latency_ms": round(elapsed_ms, 2),
                    "reason": f"API {resp.status_code}: {resp.text[:200]}",
                    "error": True,
                    "telemetry": {"memory_mb": round(mem_mb, 2), "cpu_percent": proc.cpu_percent()}
                }

            data = resp.json()
            raw_content = data["choices"][0]["message"]["content"]
            parsed = json.loads(raw_content)

            mem_mb = proc.memory_info().rss / (1024 * 1024)
            chosen_agent = parsed.get("agent", available_keys[0])
            conf_val = round(float(parsed.get("confidence", 0.95)), 4)
            urgency_val = round(float(parsed.get("urgency_score", 1.0)), 2)
            noul_val = round(float(parsed.get("requires_immediate_action", 0.75)), 4)

            probs = {k: 0.01 for k in available_keys}
            if chosen_agent in probs:
                probs[chosen_agent] = conf_val
                rem = max(0.0, 1.0 - conf_val)
                others = [k for k in available_keys if k != chosen_agent]
                if others:
                    for o in others:
                        probs[o] = round(rem / len(others), 4)
            else:
                probs[chosen_agent] = conf_val

            return {
                "engine_type": "cloud_llm",
                "model_name": req.model,
                "choice": chosen_agent,
                "confidence": conf_val,
                "probabilities": probs,
                "score": {
                    "type": "score",
                    "score": urgency_val,
                    "confidence": 0.90,
                    "legend": {"0": "Düşük (Rutin)", "1": "Normal (Standart)", "2": "Kritik (P1/Bloker)"}
                },
                "noul": {
                    "type": "noul",
                    "noul": noul_val,
                    "statement": "requires_immediate_action (Acil canlı aksiyonu veya yönetici onayı gerektirir)"
                },
                "answers": {
                    "target_agent": {
                        "type": "choice",
                        "choice": chosen_agent,
                        "confidence": conf_val,
                        "probabilities": probs
                    },
                    "urgency": {
                        "type": "score",
                        "score": urgency_val,
                        "confidence": 0.90,
                        "legend": {"0": "Düşük (Rutin)", "1": "Normal (Standart)", "2": "Kritik (P1/Bloker)"}
                    },
                    "requires_immediate_action": {
                        "type": "noul",
                        "noul": noul_val,
                        "statement": "Acil canlı aksiyonu gerektirir mi?"
                    }
                },
                "latency_ms": round(elapsed_ms, 2),
                "reason": parsed.get("reason", "Bulut modeli cikarimi."),
                "privacy": "Dis Bulut (Internet / Saglayici API)",
                "hardware": "Bulut GPU (Groq LPU / OpenAI)",
                "telemetry": {
                    "memory_mb": round(mem_mb, 2),
                    "cpu_percent": proc.cpu_percent()
                }
            }
    except Exception as e:
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        mem_mb = proc.memory_info().rss / (1024 * 1024)
        return {
            "engine_type": "cloud_llm",
            "model_name": req.model,
            "choice": "Hata",
            "confidence": 0.0,
            "latency_ms": round(elapsed_ms, 2),
            "reason": f"Baglanti Hatasi: {str(e)}",
            "error": True,
            "telemetry": {"memory_mb": round(mem_mb, 2), "cpu_percent": proc.cpu_percent()}
        }

class ModelsRequest(BaseModel):
    api_key: str
    base_url: str = "https://api.groq.com/openai/v1"

@app.post("/models")
async def get_models(req: ModelsRequest):
    api_key = req.api_key.strip()
    if not api_key:
        return {"models": []}
    try:
        headers = {"Authorization": f"Bearer {api_key}"}
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{req.base_url}/models", headers=headers)
            if resp.status_code == 200:
                data = resp.json()
                model_ids = [m["id"] for m in data.get("data", [])]
                return {"models": sorted(model_ids)}
            else:
                return {"error": f"API {resp.status_code}: {resp.text}", "models": []}
    except Exception as e:
        return {"error": str(e), "models": []}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)

