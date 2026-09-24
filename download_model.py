import os
import sys
import time
import urllib.request

sys.stdout.reconfigure(encoding="utf-8")

MODEL_URL = "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf"
OUTPUT_DIR = os.path.join(os.getcwd(), "models")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "qwen2.5-1.5b-instruct-q4_k_m.gguf")

os.makedirs(OUTPUT_DIR, exist_ok=True)

def download_file(url, target_path):
    print("Model indirme baslatiliyor: " + url, flush=True)
    print("Hedef dosya: " + target_path, flush=True)
    
    t0 = time.time()
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    
    with urllib.request.urlopen(req) as response, open(target_path, "wb") as out_file:
        total_size = int(response.headers.get("content-length", 0))
        total_mb = total_size / (1024 * 1024)
        print(f"Toplam Boyut: {total_mb:.2f} MB", flush=True)
        
        downloaded = 0
        chunk_size = 1024 * 1024 * 4
        last_log = 0
        
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            out_file.write(chunk)
            downloaded += len(chunk)
            percent = (downloaded / total_size) * 100 if total_size > 0 else 0
            
            if percent - last_log >= 10.0 or downloaded == total_size:
                elapsed = time.time() - t0
                speed = (downloaded / (1024 * 1024)) / elapsed if elapsed > 0 else 0
                print(f"Ilerleme: %{percent:.1f} ({downloaded/(1024*1024):.1f}/{total_mb:.1f} MB) - Hiz: {speed:.2f} MB/s", flush=True)
                last_log = percent

    elapsed = time.time() - t0
    print(f"Indirme Tamamlandi! Sure: {elapsed:.1f} saniye. Boyut: {os.path.getsize(target_path)/(1024*1024):.2f} MB", flush=True)

if __name__ == "__main__":
    download_file(MODEL_URL, OUTPUT_FILE)
