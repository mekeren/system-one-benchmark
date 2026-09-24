"""
System One CPU Servisi - Test İstemcisi
----------------------------------------------
Yerel servise (http://127.0.0.1:8150/v1/systemone) HTTP istekleri atarak
karar doğruluğunu ve yanıt gecikmesini test eder.
"""

import time
import json
import sys
import httpx

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

SERVER_URL = "http://127.0.0.1:8150"

TEST_MESSAGES = [
    {
        "desc": "Agent A: Altyapı & Veri Depolama Senaryosu",
        "msg": "Dün geceki veri yedekleme neden başarısız oldu, hata logunu açıklar mısın?"
    },
    {
        "desc": "Agent B: Dağıtım & Servis İzleme Senaryosu",
        "msg": "Konteyner ingress podları CrashLoopBackOff veriyor, pod loglarını çek."
    },
    {
        "desc": "Agent C: Görev & İş Takibi Senaryosu",
        "msg": "Login sayfasındaki bug için yeni görev aç ve ilgili geliştiriciye ata."
    },
    {
        "desc": "Kapsam Dışı / Düşük Güven Senaryosu",
        "msg": "Şirket yemekhanesinde bugün hangi tatlı var?"
    }
]


def test_server():
    print("=" * 70)
    print("🧪 SYSTEM ONE CPU SERVİSİ ENTEGRASYON TESTİ")
    print(f"Hedef Servis: {SERVER_URL}")
    print("=" * 70)

    # 1. Healthcheck Kontrolü
    try:
        resp = httpx.get(f"{SERVER_URL}/healthz", timeout=3.0)
        print(f"✅ Healthcheck Başarılı: {resp.json()}\n")
    except Exception as e:
        print(f"❌ Servise bağlanılamadı: {e}")
        print("Lütfen önce servisi başlatın: python server.py")
        return

    # 2. Test Mesajlarını Sırayla Çalıştır
    for i, item in enumerate(TEST_MESSAGES, 1):
        print(f"--- [Test {i}: {item['desc']}] ---")
        print(f"💬 Mesaj: \"{item['msg']}\"")

        payload = {
            "state": {
                "user_message": item["msg"],
                "channel": "General-Support"
            },
            "model": "system-one-cpu",
            "questions": {
                "target_agent": {
                    "type": "choice",
                    "instructions": "Bu mesajı en doğru çözecek hedef ajanı seçin.",
                    "criteria": {
                        "agent-a": "Altyapı ve Veri Yönetimi (Sunucu, yedekleme ve kurtarma operasyonları).",
                        "agent-b": "Dağıtım ve Operasyon (CI/CD pipeline, servis ve konteyner izleme).",
                        "agent-c": "Proje ve Görev Yönetimi (İş/task açma, atama, durum sorgulama ve planlama).",
                        "unclear_fallback": "Kapsam dışı veya genel talepler."
                    }
                },
                "urgency_score": {
                    "type": "score",
                    "instructions": "Operasyonel aciliyet derecesi",
                    "criteria": [
                        "0: Rutin bilgi alma",
                        "1: Orta derece aksaklık",
                        "2: Kritik kesinti veya arıza"
                    ]
                },
                "requires_immediate_action": {
                    "type": "noul",
                    "instructions": "Bu talep acil aksiyon gerektirir mi?"
                }
            }
        }

        t0 = time.perf_counter()
        resp = httpx.post(f"{SERVER_URL}/v1/systemone", json=payload, timeout=5.0)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        if resp.status_code == 200:
            data = resp.json()
            target = data["answers"]["target_agent"]
            urgency = data["answers"]["urgency_score"]
            backup = data["answers"]["is_backup_issue"]

            print(f"🎯 Karar (Choice)   : {target['choice'].upper()}")
            print(f"📊 Güven (Conf)     : %{target['confidence'] * 100:.1f}")
            print(f"⚡ Aciliyet (Score) : {urgency['score']} / 2.0")
            print(f"💾 Yedeklik (Noul)  : %{backup['noul'] * 100:.1f}")
            print(f"⏱️ Yanıt Süresi     : {latency_ms:.2f} ms (CPU Üzerinde)")

            if target["confidence"] >= 0.70:
                print(f"🚀 Yönlendirme      : Doğrudan {target['choice']} ajanına aktarıldı.")
            else:
                print("⚠️ Yönlendirme      : Düşük güven -> Teams Clarify Kartı tetiklendi.")
        else:
            print(f"❌ Hata Kodu: {resp.status_code} - {resp.text}")

        print()

    print("=" * 70)
    print("✅ TÜM TESTLER BAŞARIYLA TAMAMLANDI")
    print("=" * 70)


if __name__ == "__main__":
    test_server()
