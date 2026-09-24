"""
System One Multi-Model Gateway
------------------------------
FastAPI HTTP Servisi:
  - Motor 1: Kural / Kelime Tabanlı CPU Motoru
  - Motor 2: Lokal Nöral SLM (Qwen2.5-1.5B) Motoru
  - Motor 3: Harici Bulut LLM (Groq / OpenAI Uyumlu) Motoru
  - POST /v1/compare       : 3 Motoru Eşzamanlı Çalıştırıp Karşılaştırma Uç Noktası
  - POST /v1/systemone     : Standart System One Uç Noktası
  - POST /v1/llm/systemone : Harici Model ile System One
  - GET  /                 : Canlı Yönetim, Test ve 3'lü Karşılaştırma Arenası
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
from typing import Any, Dict, Optional

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx
import uvicorn

from engine import SystemOneEngine

app = FastAPI(
    title="System One 3-Way Benchmark Gateway",
    description="3 Motoru (Kural, Lokal SLM, Bulut LLM) Eşzamanlı Karşılaştıran Gateway",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = SystemOneEngine(default_temperature=0.85)


@app.get("/healthz")
def healthz():
    return {
        "status": "healthy",
        "engine": "SystemOne-3Way-Gateway",
        "engines_available": [
            "rule_based_cpu",
            "qwen2.5_1.5b_neural_slm",
            "cloud_llm_groq_openai"
        ]
    }


@app.post("/v1/systemone")
async def evaluate_local(request: Request):
    """Standart lokal System One çağrısı."""
    try:
        payload = await request.json()
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"Geçersiz JSON: {str(e)}"})
    return engine.evaluate(payload)


@app.post("/v1/llm/systemone")
async def evaluate_external_llm_direct(request: Request):
    """Harici model ile tekli System One çağrısı."""
    body = await request.json()
    return await _call_cloud_llm(
        message=body.get("message", ""),
        api_key=body.get("api_key", "").strip(),
        base_url=body.get("base_url", "https://api.groq.com/openai/v1").rstrip("/"),
        model=body.get("model", "openai/gpt-oss-120b"),
        criteria=body.get("criteria")
    )


async def _call_cloud_llm(message: str, api_key: str, base_url: str, model: str, criteria: Optional[dict] = None) -> dict:
    """Harici LLM (Groq / OpenAI) çağrısı yardımcı fonksiyonu."""
    if not criteria:
        criteria = {
            "agent-a": "Altyapı ve Veri Yönetimi (Sunucu, yedekleme ve kurtarma operasyonları).",
            "agent-b": "Dağıtım ve Operasyon (CI/CD pipeline, servis ve konteyner izleme).",
            "agent-c": "Proje ve Görev Yönetimi (İş/task açma, atama, durum sorgulama ve planlama).",
            "unclear_fallback": "Kapsam dışı veya genel talepler."
        }

    if not api_key:
        return {
            "engine_type": "cloud_llm",
            "model_name": model,
            "choice": "API Key Yok",
            "confidence": 0.0,
            "latency_ms": 0.0,
            "reason": "API anahtarı girilmediği için harici model çağrılamadı.",
            "error": "API Key eksik"
        }

    criteria_str = "\n".join([f"- {k}: {v}" for k, v in criteria.items()])
    system_prompt = f"""Sen bir System One yönlendirme ve karar modelisin.
Gelen kullanıcı mesajını aşağıdaki ajanlardan en uygununa yönlendir.
Ajanlar:
{criteria_str}

KURALLAR:
1. SADECE geçerli bir JSON döndür. Başka metin veya selamlama yazma.
2. Format:
{{
  "choice": "<secilen_ajan_id>",
  "confidence": <0.0-1.0 arasi eminlik skoru>,
  "probabilities": {{"agent-a": 0.0, "agent-b": 0.0, "agent-c": 0.0, "unclear_fallback": 0.0}},
  "reason": "<kisa gerekce>"
}}"""

    start_time = time.perf_counter()
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message}
        ],
        "temperature": 0.1,
        "max_tokens": 250
    }

    try:
        async with httpx.AsyncClient(timeout=25.0) as client:
            resp = await client.post(f"{base_url}/chat/completions", headers=headers, json=payload)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

            if resp.status_code != 200:
                return {
                    "engine_type": "cloud_llm",
                    "model_name": model,
                    "choice": "Hata",
                    "confidence": 0.0,
                    "latency_ms": round(elapsed_ms, 2),
                    "reason": f"API Sağlayıcı Hatası ({resp.status_code}): {resp.text[:120]}",
                    "error": True
                }

            data = resp.json()
            raw_content = data["choices"][0]["message"]["content"]
            match = re.search(r"\{.*\}", raw_content, re.DOTALL)
            parsed = json.loads(match.group(0)) if match else json.loads(raw_content)

            return {
                "engine_type": "cloud_llm",
                "model_name": model,
                "choice": parsed.get("choice", "unclear_fallback"),
                "confidence": round(float(parsed.get("confidence", 0.9)), 4),
                "probabilities": parsed.get("probabilities", {}),
                "latency_ms": round(elapsed_ms, 2),
                "reason": parsed.get("reason", "Model doğrudan karar verdi."),
                "privacy": "Dış Bulut (İnternet / Sağlayıcı API)",
                "hardware": "Bulut GPU (Groq LPU / OpenAI)"
            }

    except Exception as e:
        return {
            "engine_type": "cloud_llm",
            "model_name": model,
            "choice": "Hata",
            "confidence": 0.0,
            "latency_ms": 0.0,
            "reason": f"Bağlantı Hatası: {str(e)}",
            "error": True
        }


# =========================================================================
# 🥊 3 MOTORU EŞZAMANLI KARŞILAŞTIRAN UÇ NOKTA
# =========================================================================
@app.post("/v1/compare")
async def compare_all_three(request: Request):
    """
    Tek bir istekle 3 motoru eşzamanlı çalıştırır:
      1. Motor 1: Kural / Kelime Tabanlı CPU
      2. Motor 2: Lokal Nöral SLM (Qwen2.5-1.5B)
      3. Motor 3: Harici Bulut LLM (Groq / gpt-oss-120b)
    """
    body = await request.json()
    message = body.get("message", "").strip()
    api_key = body.get("api_key", "").strip()
    base_url = body.get("base_url", "https://api.groq.com/openai/v1").rstrip("/")
    model = body.get("model", "openai/gpt-oss-120b")

    payload = {
        "state": {"user_message": message},
        "questions": {
            "target_agent": {
                "type": "choice",
                "criteria": {
                    "agent-a": "Altyapı ve Veri Yönetimi (Sunucu, yedekleme ve kurtarma operasyonları).",
                    "agent-b": "Dağıtım ve Operasyon (CI/CD pipeline, servis ve konteyner izleme).",
                    "agent-c": "Proje ve Görev Yönetimi (İş/task açma, atama, durum sorgulama ve planlama).",
                    "unclear_fallback": "Kapsam dışı veya genel talepler."
                }
            }
        }
    }

    # 3 Motoru asenkron olarak paralel koştur
    task_rule = asyncio.to_thread(engine.evaluate_rule_based, payload)
    task_slm = asyncio.to_thread(engine.evaluate_slm_neural, payload)
    task_cloud = _call_cloud_llm(message, api_key, base_url, model)

    res_rule, res_slm, res_cloud = await asyncio.gather(task_rule, task_slm, task_cloud)

    # Fikir Birliği (Consensus) Analizi
    choices = [res_rule.get("choice"), res_slm.get("choice"), res_cloud.get("choice")]
    valid_choices = [c for c in choices if c and c not in ("API Key Yok", "Hata")]
    
    all_agree = len(set(valid_choices)) == 1 if valid_choices else False
    fastest = min([res_rule["latency_ms"], res_slm["latency_ms"], res_cloud["latency_ms"] if res_cloud["latency_ms"] > 0 else 9999])

    return {
        "message": message,
        "consensus": {
            "all_agree": all_agree,
            "status": "3/3 Tam Fikir Birliği" if all_agree else "Ayrışma Var (Farklı Kararlar)",
            "winner_speed": "Kural Tabanlı CPU" if fastest == res_rule["latency_ms"] else "Lokal SLM"
        },
        "engine_1_rule": res_rule,
        "engine_2_slm": res_slm,
        "engine_3_cloud": res_cloud
    }


@app.get("/", response_class=HTMLResponse)
def index_ui():
    """Gelişmiş Web Dashboard & 3'lü Benchmark Arenası."""
    html_content = """
    <!DOCTYPE html>
    <html lang="tr" class="dark">
    <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>System One & 3-Way Benchmark Arenası</title>
      <script src="https://cdn.tailwindcss.com"></script>
      <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">
      <style>
        body { background-color: #0c0e17; color: #e2e8f0; font-family: system-ui, -apple-system, sans-serif; }
        .glass { background: rgba(24, 27, 38, 0.85); backdrop-filter: blur(12px); border: 1px solid rgba(255, 255, 255, 0.08); }
        .gradient-text { background: linear-gradient(135deg, #ec4899 0%, #8b5cf6 50%, #3b82f6 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
      </style>
    </head>
    <body class="min-h-screen p-4 sm:p-8 flex flex-col items-center">
      <div class="max-w-6xl w-full space-y-6">

        <!-- Üst Başlık -->
        <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-gray-800 pb-5">
          <div class="flex items-center space-x-3">
            <div class="w-12 h-12 rounded-2xl bg-gradient-to-tr from-pink-600 via-purple-600 to-blue-500 flex items-center justify-center font-bold text-white text-xl shadow-lg shadow-pink-500/20">
              <i class="fa-solid fa-code-compare"></i>
            </div>
            <div>
              <div class="flex items-center gap-2">
                <h1 class="text-2xl font-black text-white tracking-wide">SYSTEM ONE BENCHMARK ARENASI</h1>
                <span class="text-[11px] px-2 py-0.5 rounded bg-pink-500/20 text-pink-400 font-bold border border-pink-500/30">3 MOTOR YAN YANA</span>
              </div>
              <p class="text-xs text-gray-400">Kural Motoru &bull; Lokal Nöral SLM (Qwen-1.5B) &bull; Harici Bulut LLM</p>
            </div>
          </div>
          <span class="px-3 py-1.5 rounded-xl bg-gray-900 border border-gray-800 text-xs text-emerald-400 flex items-center gap-2">
            <span class="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span> Port: 8150 Aktif
          </span>
        </div>

        <!-- 🔑 API AYARLARI KARTI -->
        <div class="glass p-5 rounded-3xl border border-gray-800 space-y-4 shadow-xl">
          <div class="flex items-center justify-between border-b border-gray-800/80 pb-2.5">
            <h2 class="text-xs font-bold text-white flex items-center gap-2">
              <i class="fa-solid fa-key text-pink-400"></i> Bulut Modeli (Motor 3) Ayarları
            </h2>
            <span class="text-[11px] text-gray-400">Key sadece 3. motor çağrıldığında kullanılır</span>
          </div>

          <div class="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs">
            <div class="space-y-1">
              <label class="font-semibold text-gray-300">API Key:</label>
              <input id="apiKeyInput" type="password" placeholder="gsk_... veya key yapıştırın" 
                     class="w-full px-3 py-2 rounded-xl bg-gray-950 border border-gray-700 text-white font-mono focus:outline-none focus:border-pink-500">
            </div>
            <div class="space-y-1">
              <label class="font-semibold text-gray-300">Sağlayıcı (Base URL):</label>
              <input id="baseUrlInput" type="text" value="https://api.groq.com/openai/v1" 
                     class="w-full px-3 py-2 rounded-xl bg-gray-950 border border-gray-700 text-white font-mono focus:outline-none focus:border-pink-500">
            </div>
            <div class="space-y-1">
              <label class="font-semibold text-gray-300">Bulut Modeli:</label>
              <select id="modelSelect" class="w-full px-3 py-2 rounded-xl bg-gray-950 border border-gray-700 text-white font-mono focus:outline-none focus:border-pink-500">
                <option value="openai/gpt-oss-120b" selected>openai/gpt-oss-120b</option>
                <option value="openai/gpt-oss-20b">openai/gpt-oss-20b</option>
                <option value="qwen/qwen3.8-27b">qwen/qwen3.8-27b</option>
                <option value="llama-3.3-70b-versatile">llama-3.3-70b-versatile</option>
              </select>
            </div>
          </div>
        </div>

        <!-- 🥊 3'LÜ TEST ALANI (INPUT & BUTTON) -->
        <div class="space-y-3">
          
          <!-- Hızlı Test Şablonları -->
          <div class="flex flex-wrap gap-2 text-xs">
            <span class="text-gray-400 py-1 font-semibold">Tuzak & Kritik Test Mesajları:</span>
            <button onclick="setQuery('Veri tabanı yedekleme işlemi neden hata verdi, backup durumunu açıkla')" class="px-3 py-1 rounded-lg bg-gray-900 hover:bg-gray-800 text-emerald-400 border border-gray-800">
              💾 Agent A: "Yedekleme & snapshot durumu"
            </button>
            <button onclick="setQuery('Yeni sürümü production ortamına deploy et ve servisi yayına al')" class="px-3 py-1 rounded-lg bg-gray-900 hover:bg-gray-800 text-blue-400 border border-gray-800">
              🚀 Agent B: "Production deployment & release"
            </button>
            <button onclick="setQuery('Sürüm yayını ile ilgili sprint panosuna yeni task açar mısın')" class="px-3 py-1 rounded-lg bg-gray-900 hover:bg-gray-800 text-purple-400 border border-gray-800">
              🔥 Tuzak: "Sprint panosuna task aç" (Agent C)
            </button>
            <button onclick="setQuery('Şirket yemekhanesinde bugün tatlı ne var?')" class="px-3 py-1 rounded-lg bg-gray-900 hover:bg-gray-800 text-yellow-400 border border-gray-800">
              ⚠️ Kapsam Dışı: Yemekhane
            </button>
          </div>

          <!-- Mesaj Girişi & Karşılaştır Butonu -->
          <div class="flex gap-2">
            <input id="testMsgInput" type="text" value="Sürüm yayını ile ilgili sprint panosuna yeni task açar mısın" 
                   class="flex-1 px-4 py-3.5 rounded-2xl bg-gray-950 border border-gray-700 text-white text-sm focus:outline-none focus:border-pink-500 font-mono shadow-inner">
            <button onclick="runCompare()" id="runCompareBtn" class="px-6 py-3.5 rounded-2xl bg-gradient-to-r from-pink-600 via-purple-600 to-blue-600 hover:opacity-90 text-white font-extrabold text-sm shadow-xl shadow-pink-600/30 flex items-center gap-2 transition">
              <i class="fa-solid fa-bolt text-xs"></i> 3 Motoru Karşılaştır
            </button>
          </div>
        </div>

        <!-- 🏆 FİKİR BİRLİĞİ (CONSENSUS) VE HIZ ÖZETİ -->
        <div id="consensusBanner" class="hidden p-4 rounded-2xl bg-gray-900/90 border border-gray-800 flex flex-col sm:flex-row items-center justify-between gap-3 text-xs">
          <div class="flex items-center gap-2.5">
            <span id="consensusIcon" class="w-7 h-7 rounded-full flex items-center justify-center font-bold text-sm bg-emerald-500/20 text-emerald-400">✓</span>
            <div>
              <span class="text-gray-400 block text-[11px]">Model Fikir Birliği (Consensus)</span>
              <span id="consensusText" class="font-bold text-white text-sm">3/3 Tam Fikir Birliği</span>
            </div>
          </div>
          <div class="flex items-center gap-4 font-mono">
            <div>
              <span class="text-gray-400 text-[11px] block">En Hızlı Motor</span>
              <span id="fastestText" class="text-emerald-400 font-bold">Kural Tabanlı CPU (~3ms)</span>
            </div>
          </div>
        </div>

        <!-- 🥊 3 YAN YANA KARŞILAŞTIRMA KARTI (GRID) -->
        <div class="grid grid-cols-1 md:grid-cols-3 gap-5">

          <!-- 1. KART: KURAL / KELİME TABANLI MOTOR -->
          <div class="glass p-5 rounded-3xl border border-gray-800 space-y-4 relative flex flex-col justify-between">
            <div class="space-y-3">
              <div class="flex items-center justify-between border-b border-gray-800 pb-2.5">
                <div class="flex items-center gap-2">
                  <span class="w-6 h-6 rounded-lg bg-gray-800 text-gray-300 flex items-center justify-center text-xs font-bold">1</span>
                  <h3 class="font-bold text-white text-sm">Kural / Kelime Motoru</h3>
                </div>
                <span class="text-[10px] px-2 py-0.5 rounded bg-gray-900 text-gray-400 font-mono">Regex / String</span>
              </div>

              <div class="space-y-2">
                <div class="flex items-baseline justify-between">
                  <span class="text-xs text-gray-400">Seçilen Ajan:</span>
                  <span id="card1Choice" class="text-sm font-black text-white px-2.5 py-1 rounded-lg bg-gray-900 border border-gray-800">--</span>
                </div>
                <div class="flex items-baseline justify-between text-xs">
                  <span class="text-gray-400">Model Güveni:</span>
                  <span id="card1Conf" class="font-bold text-emerald-400 font-mono">--</span>
                </div>
                <div class="flex items-baseline justify-between text-xs">
                  <span class="text-gray-400">Gecikme (Hız):</span>
                  <span id="card1Latency" class="font-bold text-blue-400 font-mono">-- ms</span>
                </div>
              </div>

              <div class="p-3 rounded-xl bg-gray-950 border border-gray-800/80 text-[11px] space-y-1">
                <span class="text-gray-400 block font-semibold">Karar Mantığı / Analiz:</span>
                <p id="card1Reason" class="text-gray-300 italic">Test bekleniyor...</p>
              </div>
            </div>

            <div class="pt-3 border-t border-gray-800/60 text-[11px] flex justify-between text-gray-500">
              <span>Gizlilik: %100 Kurum İçi</span>
              <span>Donanım: CPU</span>
            </div>
          </div>

          <!-- 2. KART: LOKAL NÖRAL SLM (QWEN-1.5B) -->
          <div class="glass p-5 rounded-3xl border border-purple-900/40 space-y-4 relative flex flex-col justify-between bg-purple-950/5">
            <div class="space-y-3">
              <div class="flex items-center justify-between border-b border-gray-800 pb-2.5">
                <div class="flex items-center gap-2">
                  <span class="w-6 h-6 rounded-lg bg-purple-500/20 text-purple-400 flex items-center justify-center text-xs font-bold">2</span>
                  <h3 class="font-bold text-purple-200 text-sm">Lokal Nöral SLM</h3>
                </div>
                <span class="text-[10px] px-2 py-0.5 rounded bg-purple-950 text-purple-300 font-mono">Qwen2.5-1.5B</span>
              </div>

              <div class="space-y-2">
                <div class="flex items-baseline justify-between">
                  <span class="text-xs text-gray-400">Seçilen Ajan:</span>
                  <span id="card2Choice" class="text-sm font-black text-white px-2.5 py-1 rounded-lg bg-purple-900/40 border border-purple-800 text-purple-200">--</span>
                </div>
                <div class="flex items-baseline justify-between text-xs">
                  <span class="text-gray-400">Model Güveni:</span>
                  <span id="card2Conf" class="font-bold text-emerald-400 font-mono">--</span>
                </div>
                <div class="flex items-baseline justify-between text-xs">
                  <span class="text-gray-400">Gecikme (Hız):</span>
                  <span id="card2Latency" class="font-bold text-purple-400 font-mono">-- ms</span>
                </div>
              </div>

              <div class="p-3 rounded-xl bg-gray-950 border border-gray-800/80 text-[11px] space-y-1">
                <span class="text-purple-300 block font-semibold">Anlamsal Niyet Çözümlemesi:</span>
                <p id="card2Reason" class="text-gray-300 italic">Test bekleniyor...</p>
              </div>
            </div>

            <div class="pt-3 border-t border-gray-800/60 text-[11px] flex justify-between text-purple-300">
              <span>Gizlilik: %100 Kurum İçi</span>
              <span>Donanım: CPU (Kuantize)</span>
            </div>
          </div>

          <!-- 3. KART: HARİCİ BULUT LLM (GROQ / OPENAI) -->
          <div class="glass p-5 rounded-3xl border border-pink-900/40 space-y-4 relative flex flex-col justify-between bg-pink-950/5">
            <div class="space-y-3">
              <div class="flex items-center justify-between border-b border-gray-800 pb-2.5">
                <div class="flex items-center gap-2">
                  <span class="w-6 h-6 rounded-lg bg-pink-500/20 text-pink-400 flex items-center justify-center text-xs font-bold">3</span>
                  <h3 class="font-bold text-pink-200 text-sm">Harici Bulut LLM</h3>
                </div>
                <span id="card3ModelBadge" class="text-[10px] px-2 py-0.5 rounded bg-pink-950 text-pink-300 font-mono">gpt-oss-120b</span>
              </div>

              <div class="space-y-2">
                <div class="flex items-baseline justify-between">
                  <span class="text-xs text-gray-400">Seçilen Ajan:</span>
                  <span id="card3Choice" class="text-sm font-black text-white px-2.5 py-1 rounded-lg bg-pink-900/40 border border-pink-800 text-pink-200">--</span>
                </div>
                <div class="flex items-baseline justify-between text-xs">
                  <span class="text-gray-400">Model Güveni:</span>
                  <span id="card3Conf" class="font-bold text-emerald-400 font-mono">--</span>
                </div>
                <div class="flex items-baseline justify-between text-xs">
                  <span class="text-gray-400">Gecikme (Hız):</span>
                  <span id="card3Latency" class="font-bold text-pink-400 font-mono">-- ms</span>
                </div>
              </div>

              <div class="p-3 rounded-xl bg-gray-950 border border-gray-800/80 text-[11px] space-y-1">
                <span class="text-pink-300 block font-semibold">Büyük Model Muhakemesi:</span>
                <p id="card3Reason" class="text-gray-300 italic">Test bekleniyor...</p>
              </div>
            </div>

            <div class="pt-3 border-t border-gray-800/60 text-[11px] flex justify-between text-pink-400">
              <span>Gizlilik: Dış Bulut</span>
              <span>Donanım: Bulut GPU/LPU</span>
            </div>
          </div>

        </div>

      </div>

      <script>
        // Sayfa yüklendiğinde ayarları oku
        window.addEventListener('DOMContentLoaded', () => {
          const savedKey = localStorage.getItem('decision_gateway_key') || '';
          const savedBaseUrl = localStorage.getItem('decision_gateway_base_url') || 'https://api.groq.com/openai/v1';
          const savedModel = localStorage.getItem('decision_gateway_model') || 'openai/gpt-oss-120b';

          document.getElementById('apiKeyInput').value = savedKey;
          document.getElementById('baseUrlInput').value = savedBaseUrl;
          document.getElementById('modelSelect').value = savedModel;
          document.getElementById('card3ModelBadge').innerText = savedModel.split('/').pop();
        });

        document.getElementById('apiKeyInput').addEventListener('change', (e) => {
          localStorage.setItem('decision_gateway_key', e.target.value.trim());
        });

        document.getElementById('modelSelect').addEventListener('change', (e) => {
          localStorage.setItem('decision_gateway_model', e.target.value);
          document.getElementById('card3ModelBadge').innerText = e.target.value.split('/').pop();
        });

        function setQuery(text) {
          document.getElementById('testMsgInput').value = text;
          runCompare();
        }

        async function runCompare() {
          const message = document.getElementById('testMsgInput').value.trim();
          const key = document.getElementById('apiKeyInput').value.trim();
          const baseUrl = document.getElementById('baseUrlInput').value.trim();
          const model = document.getElementById('modelSelect').value;
          const btn = document.getElementById('runCompareBtn');

          btn.disabled = true;
          btn.innerHTML = `<i class="fa-solid fa-spinner fa-spin text-xs"></i> 3 Motor Koşturuluyor...`;

          try {
            const resp = await fetch("/v1/compare", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({
                message: message,
                api_key: key,
                base_url: baseUrl,
                model: model
              })
            });

            const data = await resp.json();
            btn.disabled = false;
            btn.innerHTML = `<i class="fa-solid fa-bolt text-xs"></i> 3 Motoru Karşılaştır`;

            // 1. Motor (Kural)
            const r1 = data.engine_1_rule;
            document.getElementById('card1Choice').innerText = (r1.choice || "--").toUpperCase();
            document.getElementById('card1Conf').innerText = "%" + (r1.confidence * 100).toFixed(1);
            document.getElementById('card1Latency').innerText = r1.latency_ms + " ms";
            document.getElementById('card1Reason').innerText = r1.reason;

            // 2. Motor (Lokal SLM)
            const r2 = data.engine_2_slm;
            document.getElementById('card2Choice').innerText = (r2.choice || "--").toUpperCase();
            document.getElementById('card2Conf').innerText = "%" + (r2.confidence * 100).toFixed(1);
            document.getElementById('card2Latency').innerText = r2.latency_ms + " ms";
            document.getElementById('card2Reason').innerText = r2.reason;

            // 3. Motor (Bulut LLM)
            const r3 = data.engine_3_cloud;
            document.getElementById('card3Choice').innerText = (r3.choice || "--").toUpperCase();
            document.getElementById('card3Conf').innerText = r3.confidence ? "%" + (r3.confidence * 100).toFixed(1) : "--";
            document.getElementById('card3Latency').innerText = r3.latency_ms + " ms";
            document.getElementById('card3Reason').innerText = r3.reason;

            // Fikir Birliği Banner
            const banner = document.getElementById('consensusBanner');
            banner.classList.remove('hidden');
            const cText = document.getElementById('consensusText');
            const cIcon = document.getElementById('consensusIcon');

            if (data.consensus.all_agree) {
              cText.innerText = "✓ 3/3 Tam Fikir Birliği (" + (r2.choice || "").toUpperCase() + ")";
              cText.className = "font-bold text-emerald-400 text-sm";
              cIcon.className = "w-7 h-7 rounded-full flex items-center justify-center font-bold text-sm bg-emerald-500/20 text-emerald-400";
              cIcon.innerText = "✓";
            } else {
              cText.innerText = "⚠️ Ayrışma Var! (Motorlar farklı ajanlar seçti)";
              cText.className = "font-bold text-yellow-400 text-sm";
              cIcon.className = "w-7 h-7 rounded-full flex items-center justify-center font-bold text-sm bg-yellow-500/20 text-yellow-400";
              cIcon.innerText = "!";
            }

            document.getElementById('fastestText').innerText = data.consensus.winner_speed + " (" + Math.min(r1.latency_ms, r2.latency_ms) + " ms)";

          } catch (err) {
            btn.disabled = false;
            btn.innerHTML = `<i class="fa-solid fa-bolt text-xs"></i> 3 Motoru Karşılaştır`;
            alert("Hata: " + err);
          }
        }
      </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8150))
    print(f"🚀 System One 3-Way Benchmark Gateway Başlatılıyor: http://127.0.0.1:{port}")
    uvicorn.run("server:app", host="127.0.0.1", port=port, reload=False, log_level="info")
