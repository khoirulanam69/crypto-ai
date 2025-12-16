import ccxt
import os
import time
from dotenv import load_dotenv
from utils.dns_resolver import DNSResolver
from utils.proxy_manager import ProxyManager

load_dotenv()


class OrderManager:
    """
    Central execution layer for exchange interaction.
    Responsible for:
    - Exchange initialization
    - Time sync
    - Proxy handling
    - DNS fallback
    - Safe request retry
    """

    def __init__(self, testnet: bool = False):
        self.testnet = testnet
        self.mode = os.getenv("MODE", "paper")

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
    # ORDERS
    # ==========================================================
    def create_market_buy(self, symbol, quote_amount):
        if self.mode == "paper":
            print(f"[PAPER BUY] {symbol} quote={quote_amount}")
            return None

        return self._safe_request(
            self.exchange.create_order,
            symbol,
            "market",
            "buy",
            None,
            quote_amount,
            {"quoteOrderQty": quote_amount},
        )


    def create_market_sell(self, symbol, base_amount):
        if self.mode == "paper":
            print(f"[PAPER SELL] {symbol} base={base_amount}")
            return None

        return self._safe_request(
            self.exchange.create_market_sell_order,
            symbol,
            base_amount,
        )

    # ========================================================
    # FETCH EQUITY
    # ========================================================

    def get_equity(self):
        """
        Return total equity in quote currency (USDT)
        """
        balance = self.exchange.fetch_balance()

        total = 0.0
        for asset, data in balance['total'].items():
            if data is None or data <= 0:
                continue

            if asset == "USDT":
                total += float(data)
            else:
                symbol = f"{asset}/USDT"
                try:
                    price = self.exchange.fetch_ticker(symbol)["last"]
                    total += float(data) * price
                except Exception:
                    pass

        return total
    
    def market_buy(self, symbol, quote_amount):
        price = self.exchange.fetch_ticker(symbol)['last']
        amount = quote_amount / price
        return self.exchange.create_market_buy_order(symbol, amount)

    def market_sell(self, symbol, amount):
        return self.exchange.create_market_sell_order(symbol, amount)

