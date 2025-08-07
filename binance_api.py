import time
import requests
import hmac
import hashlib
import os
from urllib.parse import urlencode
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("BINANCE_API_KEY")
API_SECRET = os.getenv("BINANCE_API_SECRET")
BASE_URL = os.getenv("BASE_URL", "https://fapi.binance.com")

HEADERS = {
    "X-MBX-APIKEY": API_KEY
}


# === Utility ===

def _get_timestamp():
    return int(time.time() * 1000)


def _sign(params):
    query = urlencode(params)
    return hmac.new(API_SECRET.encode('utf-8'), query.encode('utf-8'), hashlib.sha256).hexdigest()


def _request(method, endpoint, params=None, retries=3):
    if params is None:
        params = {}
    params['timestamp'] = _get_timestamp()
    query = urlencode(params)
    signature = _sign(params)
    full_url = f"{BASE_URL}{endpoint}?{query}&signature={signature}"

    for attempt in range(retries):
        try:
            if method == "GET":
                res = requests.get(full_url, headers=HEADERS, timeout=10)
            elif method == "POST":
                res = requests.post(full_url, headers=HEADERS, timeout=10)
            elif method == "DELETE":
                res = requests.delete(full_url, headers=HEADERS, timeout=10)
            else:
                raise ValueError("Unsupported HTTP method")

            res.raise_for_status()
            return res.json()

        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Binance API request failed (attempt {attempt+1}):", e)
            time.sleep(2)

    print("[FAIL] Max retries reached for Binance API.")
    return None


# === Market Data ===

def get_klines(symbol="BTCUSDT", interval="1m", limit=1):
    try:
        url = f"{BASE_URL}/fapi/v1/klines"
        params = {
            "symbol": symbol.upper(),
            "interval": interval,
            "limit": limit
        }
        res = requests.get(url, params=params, headers=HEADERS)
        res.raise_for_status()
        return res.json()
    except Exception as e:
        print(f"[ERROR] Failed to fetch klines for {symbol}:", e)
        return []


def get_price(symbol="BTCUSDT"):
    try:
        url = f"{BASE_URL}/fapi/v1/ticker/price"
        res = requests.get(url, params={"symbol": symbol.upper()}, headers=HEADERS)
        res.raise_for_status()
        return float(res.json()["price"])
    except Exception as e:
        print("[ERROR] Failed to get price:", e)
        return 0.0


# === Account Info ===

def get_account_balance():
    result = _request("GET", "/fapi/v2/account")
    if result:
        usdt = next((a for a in result['assets'] if a['asset'] == 'USDT'), None)
        return float(usdt['availableBalance']) if usdt else 0.0
    return 0.0


# === Place Market Order ===

def place_market_order(symbol, side, usdt_amount):
    mark_price = get_price(symbol)
    if mark_price <= 0:
        print("[ERROR] Invalid market price. Aborting.")
        return None

    quantity = round(usdt_amount / mark_price, 3)
    if quantity <= 0:
        print(f"[ERROR] Quantity is 0. USDT: {usdt_amount}, Price: {mark_price}")
        return None

    data = {
        "symbol": symbol.upper(),
        "side": side.upper(),  # BUY or SELL
        "type": "MARKET",
        "quantity": quantity,
        "positionSide": "LONG"
    }

    print(f"[ORDER] Sending MARKET order: {side} {quantity} {symbol} @ ${mark_price:.2f}")
    return _request("POST", "/fapi/v1/order", data)


# === SL/TP Simulation (Handled by trade loop) ===

def place_sl_tp_orders(symbol, position_side, entry_price, stop_loss, take_profit):
    print(f"[INFO] SL/TP simulated: SL={stop_loss:.2f}, TP={take_profit:.2f}")
    # Let the trade_loop handle monitoring
    pass


# === Cancel / Position ===

def cancel_all_orders(symbol):
    return _request("DELETE", "/fapi/v1/allOpenOrders", {"symbol": symbol.upper()})


def get_position(symbol):
    result = _request("GET", "/fapi/v2/positionRisk")
    if result:
        for pos in result:
            if pos['symbol'] == symbol.upper() and float(pos['positionAmt']) != 0:
                return pos
    return None
