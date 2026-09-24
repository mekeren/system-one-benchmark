import asyncio
import os
import psutil
import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

app = FastAPI(title="System One Benchmark Gateway & UI", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Service endpoints (Docker Compose service names or localhost fallback)
RULE_URL = os.environ.get("RULE_SERVICE_URL", "http://127.0.0.1:8001")
SLM_URL = os.environ.get("SLM_SERVICE_URL", "http://127.0.0.1:8002")
CLOUD_URL = os.environ.get("CLOUD_SERVICE_URL", "http://127.0.0.1:8003")

@app.get("/healthz")
def healthz():
    proc = psutil.Process()
    return {
        "status": "healthy",
        "service": "decision-ui-gateway",
        "memory_mb": round(proc.memory_info().rss / (1024*1024), 2),
        "cpu_percent": proc.cpu_percent()
    }

@app.get("/v1/cluster/status")
async def cluster_status():
    """Tüm 4 servisin anlık RAM ve CPU kullanımını toplar."""
    gateway_proc = psutil.Process()
    metrics = {
        "gateway_ui": {
            "name": "decision-ui-gateway",
            "status": "online",
            "memory_mb": round(gateway_proc.memory_info().rss / (1024*1024), 2),
            "cpu_percent": gateway_proc.cpu_percent()
        }
    }
    
    async with httpx.AsyncClient(timeout=2.0) as client:
        # Rule Service
        try:
            r = await client.get(f"{RULE_URL}/healthz")
            metrics["rule_engine"] = r.json()
        except Exception:
            metrics["rule_engine"] = {"status": "offline", "memory_mb": 0, "cpu_percent": 0}

        # SLM Service
        try:
            r = await client.get(f"{SLM_URL}/healthz")
            metrics["slm_engine"] = r.json()
        except Exception:
            metrics["slm_engine"] = {"status": "offline", "memory_mb": 0, "cpu_percent": 0}

        # Cloud Service
        try:
            r = await client.get(f"{CLOUD_URL}/healthz")
            metrics["cloud_engine"] = r.json()
        except Exception:
            metrics["cloud_engine"] = {"status": "offline", "memory_mb": 0, "cpu_percent": 0}

    return metrics

@app.post("/v1/compare")
async def compare_all_three(request: Request):
    body = await request.json()
    message = body.get("message", "").strip()
    api_key = body.get("api_key", "").strip()
    base_url = body.get("base_url", "https://api.groq.com/openai/v1").rstrip("/")
    model = body.get("model", "openai/gpt-oss-120b")
    criteria = body.get("criteria", None)

    async with httpx.AsyncClient(timeout=15.0) as client:
        # 3 microservice concurrently
        async def call_rule():
            try:
                res = await client.post(f"{RULE_URL}/predict", json={"message": message, "criteria": criteria})
                return res.json()
            except Exception as e:
                return {"engine_type": "rule_based", "choice": "Bağlantı Hatası", "latency_ms": 0, "reason": str(e), "error": True}

        async def call_slm():
            try:
                res = await client.post(f"{SLM_URL}/predict", json={"message": message, "criteria": criteria})
                return res.json()
            except Exception as e:
                return {"engine_type": "local_slm_neural", "choice": "Bağlantı Hatası", "latency_ms": 0, "reason": str(e), "error": True}

        async def call_cloud():
            try:
                res = await client.post(f"{CLOUD_URL}/predict", json={"message": message, "api_key": api_key, "base_url": base_url, "model": model, "criteria": criteria})
                return res.json()
            except Exception as e:
                return {"engine_type": "cloud_llm", "choice": "Bağlantı Hatası", "latency_ms": 0, "reason": str(e), "error": True}

        res_rule, res_slm, res_cloud = await asyncio.gather(call_rule(), call_slm(), call_cloud())

    # Consensus calculation
    choices = [res_rule.get("choice"), res_slm.get("choice"), res_cloud.get("choice")]
    valid_choices = [c for c in choices if c and c not in ("API Key Yok", "Hata", "Bağlantı Hatası", "API Hatası")]
    all_agree = len(set(valid_choices)) == 1 if len(valid_choices) >= 2 else False

    latencies = [l for l in [res_rule.get("latency_ms", 0), res_slm.get("latency_ms", 0), res_cloud.get("latency_ms", 0)] if l > 0]
    fastest = min(latencies) if latencies else 0

    speed_winner = "Bilinmiyor"
    if fastest == res_rule.get("latency_ms"):
        speed_winner = "Kural Tabanlı CPU"
    elif fastest == res_slm.get("latency_ms"):
        speed_winner = "Lokal SLM (Qwen)"
    elif fastest == res_cloud.get("latency_ms"):
        speed_winner = "Bulut LLM"

    return {
        "message": message,
        "consensus": {
            "all_agree": all_agree,
            "status": "3/3 Tam Fikir Birliği" if all_agree else "Ayrışma Var (Farklı Kararlar)",
            "winner_speed": speed_winner
        },
        "engine_1_rule": res_rule,
        "engine_2_slm": res_slm,
        "engine_3_cloud": res_cloud
    }

@app.post("/v1/systemone")
async def evaluate_systemone(request: Request):
    """TypeSafe Jev standardına tam uyumlu mikro-karar uç noktası."""
    body = await request.json()
    state = body.get("state", {})
    message = state.get("user_message", "")
    questions = body.get("questions", {})
    
    # Extract criteria from target_agent question if provided
    target_q = questions.get("target_agent", {})
    criteria = target_q.get("criteria", None)

    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            r = await client.post(f"{SLM_URL}/predict", json={"message": message, "criteria": criteria})
            data = r.json()
        except Exception as e:
            # Fallback to rule engine if SLM fails
            try:
                r = await client.post(f"{RULE_URL}/predict", json={"message": message, "criteria": criteria})
                data = r.json()
            except Exception as e2:
                data = {
                    "choice": "unclear_fallback",
                    "confidence": 0.0,
                    "probabilities": {},
                    "latency_ms": 0.0,
                    "reason": f"Hata: {str(e2)}"
                }

    choice = data.get("choice", "unclear_fallback")
    conf = data.get("confidence", 0.0)
    probs = data.get("probabilities", {})
    reason = data.get("reason", "")
    score_obj = data.get("score", {"score": 1.0, "confidence": 0.85})
    noul_obj = data.get("noul", {"noul": 0.9 if choice == "agent-a" else 0.05})

    return {
        "model": "system-one-cpu-v1.0",
        "answers": {
            "target_agent": {
                "type": "choice",
                "choice": choice,
                "confidence": conf,
                "probabilities": probs,
                "reason": reason
            },
            "urgency_score": {
                "type": "score",
                "score": score_obj.get("score", 1.0),
                "confidence": score_obj.get("confidence", 0.85)
            },
            "is_backup_issue": {
                "type": "noul",
                "noul": noul_obj.get("noul", 0.9 if choice == "agent-a" else 0.05)
            },
            "requires_immediate_action": {
                "type": "noul",
                "noul": noul_obj.get("noul", 0.5)
            }
        },
        "latency_ms": data.get("latency_ms", 0.0)
    }

@app.post("/v1/models")
async def proxy_models(request: Request):
    body = await request.json()
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            r = await client.post(f"{CLOUD_URL}/models", json=body)
            return r.json()
        except Exception as e:
            return {"error": str(e), "models": []}

@app.get("/", response_class=HTMLResponse)
def index_ui():
    with open(os.path.join(os.path.dirname(__file__), "ui_template.html"), "r", encoding="utf-8") as f:
        return f.read()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8150)

