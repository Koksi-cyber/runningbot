import requests

try:
    r = requests.get("https://fapi.binance.com/fapi/v1/ping", timeout=10)
    print("✅ STATUS:", r.status_code)
    print("✅ RESPONSE:", r.text)
except Exception as e:
    print("❌ ERROR:", e)
