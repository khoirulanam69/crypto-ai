import ccxt
import os
import time
import logging
import requests
import urllib3
from dotenv import load_dotenv

from utils.dns_resolver import DNSResolver
from utils.proxy_manager import ProxyManager

load_dotenv()

# ===================================================================
#   LOGGING
# ===================================================================
logger = logging.getLogger("OrderManager")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s"))
logger.addHandler(handler)

# ===================================================================
#   CUSTOM SSL ADAPTER FOR FORCED SNI + HOST HEADER
# ===================================================================

class HostHeaderSSLAdapter(requests.adapters.HTTPAdapter):
    """
    Force requests to connect to IP but keep SNI & Host header to the original domain.
    This is REQUIRED for Binance to accept the request while bypassing DNS.
    """
    def __init__(self, override_ip, host, **kwargs):
        self.override_ip = override_ip
        self.host = host
        super().__init__(**kwargs)

    def get_connection(self, url, proxies=None):
        # Replace host with IP, but keep https://
        new_url = url.replace(self.host, self.override_ip)
        return super().get_connection(new_url, proxies)

    def add_headers(self, request, **kwargs):
        # Binance requires correct Host header
        request.headers['Host'] = self.host
        return super().add_headers(request, **kwargs)

    def init_poolmanager(self, *args, **kwargs):
        # Enforce SNI
        kwargs['server_hostname'] = self.host
        return super().init_poolmanager(*args, **kwargs)

# ===================================================================
#   ORDER MANAGER FINAL
# ===================================================================

class OrderManager:
    def __init__(self, api_key=None, secret=None, testnet=False):
        self.api_key = api_key or os.getenv("BINANCE_API_KEY")
        self.secret = secret or os.getenv("BINANCE_API_SECRET")
        self.testnet = testnet
        
        # Load DNS resolver
        self.resolver = DNSResolver(
            fallback_ips=["18.162.165.240", "18.181.3.53"]
        )

        # Load proxies from .env
        proxies_env = os.getenv("PROXIES", "")
        proxy_list = [p.strip() for p in proxies_env.split(",") if p.strip()]
        self.proxy_manager = ProxyManager(proxy_list)
        logger.info(f"Loaded {len(proxy_list)} proxies")

        # Init exchange
        self.exchange = ccxt.binance({
            "apiKey": self.api_key,
            "secret": self.secret,
            "enableRateLimit": True,
            "timeout": 20000,
            "options": {
                "adjustForTimeDifference": True,
                "recvWindow": 10000,
            },
        })

        # Apply working proxy to ccxt session
        self._apply_proxy()

        # Force SNI + Host HEADER using session adapter
        self._apply_dns_override()

    # ===================================================================
    #   APPLY PROXY
    # ===================================================================

    def _apply_proxy(self):
        try:
            proxy = self.proxy_manager.get_working_proxy()
            if proxy and hasattr(self.exchange, "session"):
                logger.info(f"[proxy] Using {proxy}")
                self.exchange.session.proxies.update(
                    {"http": proxy, "https": proxy}
                )
            else:
                logger.info("[proxy] none (direct)")
        except:
            logger.warning("Failed applying proxy")

    # ===================================================================
    #   APPLY DNS OVERRIDE (WITHOUT BREAKING SNI)
    # ===================================================================

    def _apply_dns_override(self):
        host = "api.binance.com"
        resolved_ip = self.resolver.resolve(host)

        if not resolved_ip:
            logger.warning("DNS override failed. Using normal domain.")
            return
        
        logger.info(f"Resolved {host} -> {resolved_ip} (applying SNI adapter)")

        if hasattr(self.exchange, "session"):
            adapter = HostHeaderSSLAdapter(resolved_ip, host)
            self.exchange.session.mount("https://api.binance.com", adapter)

    # ===================================================================
    #   SAFE REQUEST (RETRY + PROXY SWITCH + BACKOFF)
    # ===================================================================

    def _safe_request(self, func, *args, retries=4, **kwargs):
        delay = 1
        last_err = None

        for attempt in range(1, retries + 1):
            try:
                return func(*args, **kwargs)

            except Exception as e:
                last_err = e
                logger.warning(
                    f"Safe request attempt {attempt}/{retries} failed: {e}"
                )

                # Switch proxy
                try:
                    self._apply_proxy()
                except:
                    pass

                logger.info(f"Sleeping {delay:.1f}s before retry")
                time.sleep(delay)
                delay *= 2

        raise last_err

    # ===================================================================
    #   MARKET DATA
    # ===================================================================

    def fetch_ohlcv(self, symbol, timeframe='1m', limit=100):
        return self._safe_request(
            self.exchange.fetch_ohlcv,
            symbol,
            timeframe=timeframe,
            limit=limit
        )

    def fetch_ticker(self, symbol):
        return self._safe_request(
            self.exchange.fetch_ticker,
            symbol
        )

    # ===================================================================
    #   TRADING FUNCTIONS
    # ===================================================================

    def create_limit_buy(self, symbol, price, amount):
        logger.info(f"[TRADE] BUY {symbol} @ {price} amount={amount}")
        return self._safe_request(
            self.exchange.create_limit_buy_order,
            symbol, amount, price
        )

    def create_limit_sell(self, symbol, price, amount):
        logger.info(f"[TRADE] SELL {symbol} @ {price} amount={amount}")
        return self._safe_request(
            self.exchange.create_limit_sell_order,
            symbol, amount, price
        )

    def create_market_buy(self, symbol, amount):
        logger.info(f"[TRADE] MARKET BUY {symbol} amount={amount}")
        return self._safe_request(
            self.exchange.create_market_buy_order,
            symbol, amount
        )

    def create_market_sell(self, symbol, amount):
        logger.info(f"[TRADE] MARKET SELL {symbol} amount={amount}")
        return self._safe_request(
            self.exchange.create_market_sell_order,
            symbol, amount
        )
    
    # ===================================================================
    #   SIMPLE ALIAS METHODS (for compatibility with old scripts)
    # ===================================================================

    def market_buy(self, symbol, amount):
        """
        Alias for create_market_buy, because run_realtime.py calls market_buy()
        """
        return self.create_market_buy(symbol, amount)

    def market_sell(self, symbol, amount):
        """
        Alias for create_market_sell
        """
        return self.create_market_sell(symbol, amount)
