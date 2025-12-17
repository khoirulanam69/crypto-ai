import ccxt
import os
import time
from dotenv import load_dotenv
from utils.dns_resolver import DNSResolver
from utils.proxy_manager import ProxyManager
from executor.state_manager import StateManager

load_dotenv()


class OrderManager:
    def __init__(self, testnet: bool = False):
        self.testnet = testnet
        self.mode = os.getenv("MODE", "paper")
        self.state = StateManager(self.exchange, symbol=os.getenv("SYMBOL", "BTC/USDT"))

        # ======================
        # API KEYS
        # ======================
        api_key = os.getenv("BINANCE_API_KEY")
        secret = os.getenv("BINANCE_API_SECRET")

        if not api_key or not secret:
            raise RuntimeError("BINANCE_API_KEY / BINANCE_API_SECRET not set")

        # ======================
        # DNS + PROXY
        # ======================
        self.resolver = DNSResolver(
            fallback_ips=[
                "18.162.165.240",
                "18.181.3.53",
            ]
        )

        proxies = os.getenv("PROXIES", "")
        proxy_list = [p.strip() for p in proxies.split(",") if p.strip()]
        self.proxy_manager = ProxyManager(proxy_list)

        # ======================
        # EXCHANGE INIT
        # ======================
        self.exchange = ccxt.binance({
            "apiKey": api_key,
            "secret": secret,
            "enableRateLimit": True,
            "timeout": 20000,
            "options": {
                "adjustForTimeDifference": True,
                "recvWindow": 10000,
            },
        })

        # ======================
        # TIME SYNC (CRITICAL)
        # ======================
        try:
            self.exchange.load_time_difference()
        except Exception as e:
            print("[OrderManager] time sync failed:", e)

        # ======================
        # APPLY PROXY (IF ANY)
        # ======================
        try:
            proxy = self.proxy_manager.get_working_proxy()
            if proxy and hasattr(self.exchange, "session"):
                self.exchange.session.proxies.update({
                    "http": proxy,
                    "https": proxy,
                })
                print(f"[proxy] {proxy}")
            else:
                print("[proxy] none (direct)")
        except Exception as e:
            print("[proxy] error:", e)

    # ==========================================================
    # SAFE REQUEST (RETRY + PROXY ROTATION)
    # ==========================================================
    def _safe_request(self, func, *args, retries: int = 3, backoff: int = 2, **kwargs):
        last_exc = None

        for attempt in range(1, retries + 1):
            try:
                return func(*args, **kwargs)

            except Exception as e:
                last_exc = e
                print(f"[retry {attempt}/{retries}] {e}")

                # rotate proxy
                try:
                    proxy = self.proxy_manager.get_working_proxy()
                    if proxy and hasattr(self.exchange, "session"):
                        self.exchange.session.proxies.update({
                            "http": proxy,
                            "https": proxy,
                        })
                        print(f"[proxy switch] {proxy}")
                except:
                    pass

                time.sleep(backoff * attempt)

        raise last_exc

    # ==========================================================
    # MARKET DATA
    # ==========================================================
    def fetch_ohlcv(self, symbol, timeframe="1m", limit=100):
        return self._safe_request(
            self.exchange.fetch_ohlcv,
            symbol,
            timeframe=timeframe,
            limit=limit,
        )
    
    # ==========================================================
    # GET BALANCE
    # ==========================================================

    def get_balance(self, asset):
        balance = self.exchange.fetch_balance()
        return float(balance.get(asset, {}).get("free", 0))

    # ==========================================================
    # ORDERS
    # ==========================================================
    
    def safe_market_buy(self, symbol, quote_amount):
        quote = symbol.split("/")[1]
        free_balance = self.get_balance(quote)

        if not self.state.can_buy():
            print("[GUARD] Insufficient quote balance → HOLD")
            return None

        if free_balance < quote_amount:
            print(f"[SKIP] Insufficient {quote} balance")
            return None

        if quote_amount < 5:
            print("[SKIP] Buy below minimum notional")
            return None

        return self._safe_request(
            self.exchange.create_market_buy_order,
            symbol,
            quote_amount
        )
    
    def safe_market_sell(self, symbol, amount):
        base = symbol.split("/")[0]
        free_balance = self.get_balance(base)
        
        if not self.state.can_sell():
            print("[GUARD] No position, SELL blocked → HOLD")
            return None

        if free_balance <= 0:
            print(f"[SKIP] No {base} balance to sell")
            return None

        sell_amount = min(amount, free_balance * 0.999)

        if sell_amount * self.exchange.fetch_ticker(symbol)['last'] < 5:
            print("[SKIP] Sell amount below minimum notional")
            return None

        return self._safe_request(
            self.exchange.create_market_sell_order,
            symbol,
            sell_amount
        )

    # ========================================================
    # FETCH EQUITY
    # ========================================================

    def get_equity(self, symbol="BTC/USDT"):
        balance = self.exchange.fetch_balance()

        base, quote = symbol.split("/")

        free_quote = balance["free"].get(quote, 0)
        free_base = balance["free"].get(base, 0)

        ticker = self.exchange.fetch_ticker(symbol)
        price = float(ticker["last"])

        equity = free_quote + free_base * price
        return float(equity)
