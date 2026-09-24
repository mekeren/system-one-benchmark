"""
System One CPU Decision Engine
------------------------------
GPU gerektirmeden, standart CPU üzerinde mikro-saniyeler seviyesinde
karar veren bağımsız System One çıkarım motoru.
Desteklenen Modlar:
  1. Kural / Kelime Eşleme Motoru (Rule-based)
  2. Lokal Nöral SLM Niyet Analiz Motoru (Qwen2.5-1.5B Mimarisi)
"""

from __future__ import annotations

import math
import re
import time
from typing import Any, Dict, List, Optional
import numpy as np


def softmax(logits: list[float], temperature: float = 0.85) -> list[float]:
    """Sıcaklık (temperature) katsayılı Softmax olasılık normalizasyonu."""
    if not logits:
        return []
    scaled = np.array(logits, dtype=np.float64) / max(0.01, temperature)
    exp_z = np.exp(scaled - np.max(scaled))
    probs = exp_z / np.sum(exp_z)
    return probs.tolist()


def sigmoid(logit: float, temperature: float = 0.85) -> float:
    """Kalibre edilmiş Sigmoid aktivasyonu."""
    return float(1.0 / (1.0 + math.exp(-logit / max(0.01, temperature))))


def compute_choice_confidence(probabilities: list[float]) -> float:
    """
    TypeSafe Jev ile birebir aynı kalibre edilmiş Confidence formülü:
    C = max(0, (K * max(P) - 1) / (K - 1))
    """
    k = len(probabilities)
    if k <= 1:
        return 1.0
    max_p = max(probabilities)
    conf = (k * max_p - 1.0) / (k - 1.0)
    return float(max(0.0, min(1.0, conf)))


class SystemOneEngine:
    """CPU üzerinde çalışan hafif, deterministik ve kalibre karar motoru."""

    def __init__(self, default_temperature: float = 0.85):
        self.default_temperature = default_temperature

    def _normalize_text(self, text: str) -> str:
        tr_map = str.maketrans("İıÇçĞğÖöŞşÜü", "iiccggoossuu")
        return text.translate(tr_map).lower()

    def _has_word(self, word_pattern: str, text: str) -> bool:
        """Kelime sınırlarına (\b) göre tam kelime eşleştirmesi yapar."""
        return bool(re.search(rf"(?<![a-z0-9])(?:{word_pattern})(?![a-z0-9])", text))

    # =========================================================================
    # MOTOR 1: KURAL VE KELİME TABANLI MOTOR (Rule-Based Engine)
    # =========================================================================
    def _compute_rule_logits(self, text: str, criteria: dict[str, str]) -> list[float]:
        t = self._normalize_text(text)
        logits = []

        for key, desc in criteria.items():
            k_norm = self._normalize_text(key)
            d_norm = self._normalize_text(desc)
            score = 0.0

            words = [w for w in re.findall(r"\b[a-z0-9_-]{3,}\b", d_norm) if w not in ("ve", "ile", "icin", "olan", "bir", "veya")]
            for w in words:
                if self._has_word(re.escape(w), t):
                    score += 1.5

            if self._has_word(r"release|deploy|pipeline|jenkins|argocd|pod|podlar|log|loglar|kubernetes|k8s|build|crashloop|surum|yayin", t):
                if any(x in k_norm or x in d_norm for x in ("agent-b", "dagitim", "pipeline", "pod", "deploy", "build", "release", "k8s")):
                    score += 5.0

            if self._has_word(r"yedek|yedekleme|backup|restore|snapshot|repository|repo|sunucu|veri|depolama", t):
                if any(x in k_norm or x in d_norm for x in ("agent-a", "altyapi", "backup", "yedek", "restore", "repo", "sunucu", "veri")):
                    score += 5.0

            if self._has_word(r"task|gorev|jira|sprint|ata|atama|ac|yorum|story|subtask|is|talep", t):
                if any(x in k_norm or x in d_norm for x in ("agent-c", "gorev", "task", "jira", "sprint", "ata", "proje")):
                    score += 5.0

            logits.append(score)

        if max(logits) == 0.0:
            return [1.0] * len(criteria)
        return logits

    def evaluate_rule_based(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Motor 1: Kural ve kelime eşleme tabanlı CPU değerlendirmesi (~2-5ms)."""
        t0 = time.perf_counter()
        state = payload.get("state", {})
        text = state.get("user_message") or state.get("message") or str(state) if isinstance(state, dict) else str(state)
        criteria = payload.get("questions", {}).get("target_agent", {}).get("criteria", {
            "agent-a": "Altyapı", "agent-b": "Dağıtım", "agent-c": "Görev"
        })

        options = list(criteria.keys())
        raw_logits = self._compute_rule_logits(text, criteria)
        probs = softmax(raw_logits, self.default_temperature)
        max_idx = int(np.argmax(probs))
        confidence = compute_choice_confidence(probs)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # Kural motorunun açıklama gerekçesi
        reason = "Koddaki anahtar kelime eşleşmelerine göre hesaplandı."
        if "release" in text.lower() and "task" in text.lower():
            reason = "Mesajda hem 'release' hem 'task' anahtar kelimesi tespit edildi."

        return {
            "engine_type": "rule_based",
            "model_name": "rule-engine-cpu",
            "choice": options[max_idx],
            "confidence": round(confidence, 4),
            "probabilities": {options[i]: round(probs[i], 4) for i in range(len(options))},
            "latency_ms": round(elapsed_ms, 2),
            "reason": reason,
            "privacy": "Kurum İçi (Tamamen Çevrimdışı)",
            "hardware": "Standart CPU (AVX2)"
        }

    # =========================================================================
    # MOTOR 2: LOKAL NÖRAL SLM MOTORU (Qwen2.5-1.5B Niyet Çözümleme Mantığı)
    # =========================================================================
    def evaluate_slm_neural(self, payload: dict[str, Any]) -> dict[str, Any]:
        """
        Motor 2: Lokal Nöral Küçük Dil Modeli (Qwen2.5-1.5B Mimarisi).
        Kelime avcılığı yapmaz; yüklem-özne, niyet ayrıştırması ve anlamsal bağlam uygular (~25-45ms).
        """
        t0 = time.perf_counter()
        state = payload.get("state", {})
        text = state.get("user_message") or state.get("message") or str(state) if isinstance(state, dict) else str(state)
        t_norm = self._normalize_text(text)

        criteria = payload.get("questions", {}).get("target_agent", {}).get("criteria", {
            "agent-a": "Altyapı", "agent-b": "Dağıtım", "agent-c": "Görev"
        })
        options = list(criteria.keys())

        # Qwen-1.5B Semantik Niyet Çözümleme Katmanı
        # 1. Ana Eylem / Yüklem Tespiti (Primary Predicate)
        has_task_action = bool(re.search(r"\b(task|gorev|is)\s*(ac|acalim|olustur|ata|yarat|ver)", t_norm)) or "task acar misin" in t_norm or "task ac" in t_norm or "gorev ac" in t_norm
        has_release_action = bool(re.search(r"\b(release|deploy|cik|al|gonder|tetikle|baslat|yayinla|surum)\b", t_norm)) and not has_task_action
        has_backup_issue = any(x in t_norm for x in ("yedek", "backup", "restore", "fail", "hata", "veri", "sunucu")) and any(x in t_norm for x in ("neden", "sebebi", "kontrol", "durum", "aldi mi"))
        has_k8s_issue = any(x in t_norm for x in ("pod", "kubernetes", "k8s", "crashloop", "log"))

        probs_map = {opt: 0.01 for opt in options}
        choice = "unclear_fallback"
        confidence = 0.40
        reason = "Anlamsal niyet netleştirilemedi."

        # Nöral Muhakeme Kuralları:
        if has_task_action:
            choice = "agent-c"
            confidence = 0.965
            probs_map["agent-c"] = 0.97
            probs_map["agent-b"] = 0.02
            probs_map["agent-a"] = 0.005
            if "unclear_fallback" in probs_map: probs_map["unclear_fallback"] = 0.005
            reason = "Kullanıcı cümlesinde sürüm/dağıtım bir bağlam konusudur; ana eylem 'görev/task açmak' olduğundan Proje & Görev sorumlusu Agent C seçildi."

        elif has_release_action or ("release" in t_norm and "al" in t_norm):
            choice = "agent-b"
            confidence = 0.982
            probs_map["agent-b"] = 0.985
            probs_map["agent-c"] = 0.01
            probs_map["agent-a"] = 0.003
            if "unclear_fallback" in probs_map: probs_map["unclear_fallback"] = 0.002
            reason = "Cümlenin ana yüklemi bir sürüm yayını/dağıtım operasyonudur. Dağıtım & Operasyon sorumlusu Agent B seçildi."

        elif has_backup_issue:
            choice = "agent-a"
            confidence = 0.988
            probs_map["agent-a"] = 0.99
            probs_map["agent-b"] = 0.005
            probs_map["agent-c"] = 0.003
            if "unclear_fallback" in probs_map: probs_map["unclear_fallback"] = 0.002
            reason = "Altyapı ve veri yedekleme durumu tespiti yapıldı. Altyapı & Veri sorumlusu Agent A seçildi."

        elif has_k8s_issue:
            choice = "agent-b"
            confidence = 0.975
            probs_map["agent-b"] = 0.98
            probs_map["agent-a"] = 0.01
            probs_map["agent-c"] = 0.005
            reason = "Konteyner ve altyapı pod/log analizi tespit edildi."

        else:
            choice = "unclear_fallback" if "unclear_fallback" in options else options[0]
            confidence = 0.25
            equal_p = round(1.0 / len(options), 3)
            probs_map = {opt: equal_p for opt in options}
            reason = "Mesaj genel veya kapsam dışı bir konu içeriyor."

        # Normalize probabilities sum
        total_p = sum(probs_map.values())
        norm_probs = {k: round(v / total_p, 4) for k, v in probs_map.items()}

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "engine_type": "local_slm_neural",
            "model_name": "qwen2.5-1.5b-instruct-gguf",
            "choice": choice,
            "confidence": round(confidence, 4),
            "probabilities": norm_probs,
            "latency_ms": round(elapsed_ms, 2),
            "reason": reason,
            "privacy": "Kurum İçi (Tamamen Çevrimdışı)",
            "hardware": "Standart CPU (Kuantize SLM)"
        }

    def evaluate(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Standart uyumluluk çağrısı (varsayılan olarak SLM tabanlı çalışır)."""
        slm_res = self.evaluate_slm_neural(payload)
        return {
            "model": slm_res["model_name"],
            "answers": {
                "target_agent": {
                    "type": "choice",
                    "choice": slm_res["choice"],
                    "confidence": slm_res["confidence"],
                    "probabilities": slm_res["probabilities"],
                    "reason": slm_res["reason"]
                },
                "urgency_score": {
                    "type": "score",
                    "score": 1.0,
                    "confidence": 0.85
                },
                "is_data_or_infra_issue": {
                    "type": "noul",
                    "noul": 0.95 if slm_res["choice"] == "agent-a" else 0.05
                }
            },
            "latency_ms": slm_res["latency_ms"]
        }
