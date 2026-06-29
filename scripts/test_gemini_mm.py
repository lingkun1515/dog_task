import os, sys, json, base64, traceback
import urllib.request

API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyBlvvoCPQ4iw8c91S7RID8iye3N0QV_ogg")
MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
PROXY = os.environ.get("PROXY_URL", "http://127.0.0.1:7897")
IMG = sys.argv[1] if len(sys.argv) > 1 else "/tmp/vframes/gemini_test/frame_02.jpg"

print(f"MODEL={MODEL} PROXY={PROXY}", flush=True)
print(f"Image: {IMG}", flush=True)

with open(IMG, "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode()
print(f"Image base64 length: {len(img_b64)}", flush=True)

url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={API_KEY}"
payload = json.dumps({
    "contents": [{"parts": [
        {"text": "Describe this image in 2-3 sentences. Focus on any robot, its pose, and objects."},
        {"inline_data": {"mime_type": "image/jpeg", "data": img_b64}}
    ]}],
    "generationConfig": {"maxOutputTokens": 200}
}).encode()

print(f"Payload: {len(payload)} bytes. Sending...", flush=True)

proxy_handler = urllib.request.ProxyHandler({"https": PROXY, "http": PROXY})
opener = urllib.request.build_opener(proxy_handler)
req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})

try:
    resp = opener.open(req, timeout=45)
    data = json.loads(resp.read())
    text = data["candidates"][0]["content"]["parts"][0]["text"]
    print("=== SUCCESS (multimodal) ===", flush=True)
    print(text, flush=True)
    print(f"Token usage: {data.get('usageMetadata', {})}", flush=True)
except urllib.error.HTTPError as e:
    body = e.read().decode()
    print(f"HTTP {e.code}: {body[:500]}", flush=True)
    sys.exit(1)
except Exception:
    traceback.print_exc()
    sys.exit(1)
