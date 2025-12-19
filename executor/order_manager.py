import ccxt
import os
import time
import logging
from dotenv import load_dotenv
from utils.dns_resolver import DNSResolver
from utils.proxy_manager import ProxyManager
from executor.state_manager import StateManager

load_dotenv()

# Setup logging
logger = logging.getLogger(__name__)

class OrderManager:
    def __init__(self, testnet: bool = False):
        self.testnet = testnet
        self.mode = os.getenv("MODE", "paper")
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests
        
        # Cache initialization
        self._balance_cache = None
        self._balance_cache_time = 0
        self._ticker_cache = {}
        self._ticker_cache_time = {}

        # ======================
        # API KEYS
        # ======================
        api_key = os.getenv("BINANCE_API_KEY")
        secret = os.getenv("BINANCE_API_SECRET")

        if not api_key or not secret:
            raise RuntimeError("API credentials not configured")

        # ======================
        # DNS + PROXY
        # ======================
        # DNS Fallback with proper handling
        fallback_ips_str = os.getenv("DNS_FALLBACK_IPS", "")
        fallback_ips = [ip.strip() for ip in fallback_ips_str.split(",") if ip.strip()]
        if not fallback_ips:
            fallback_ips = ["18.162.165.240", "18.181.3.53"]  # defaults
        
        self.resolver = DNSResolver(fallback_ips=fallback_ips)

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
            "timeout": 30000,  # 30 seconds (default 10000)
            "rateLimit": 1000,  # requests per minute
            "options": {
                "adjustForTimeDifference": True,
                "recvWindow": 60000,  # 60 seconds
                "defaultType": "spot",
            },
            "headers": {
                "User-Agent": "crypto-ai-bot/1.0"
            }
        })
        self.exchange.aiohttp_trust_env = True  # Allow proxy from env
        self.exchange.verbose = DEBUG  # Debug mode

        # ======================
        # TIME SYNC (CRITICAL)
        # ======================
        try:
            self.exchange.load_time_difference()
            logger.info("Time sync successful")
        except Exception as e:
            logger.warning(f"Time sync failed: {e}")

        # ===============================
        # INIT STATE MANAGER
        # ===============================
        initial_cash = float(os.getenv("INITIAL_CASH", "0"))
        self.state = StateManager(initial_cash)

        # ======================
        # APPLY PROXY (IF ANY)
        # ======================
        try:
            proxy = self.proxy_manager.get_working_proxy()
            self._apply_proxy(proxy)
            if proxy:
                logger.info(f"Proxy applied: {proxy}")
            else:
                logger.info("No proxy (direct connection)")
        except Exception as e:
            logger.error(f"Proxy error: {e}")

    # ==========================================================
    # RATE LIMIT ENFORCEMENT
    # ==========================================================
    def _enforce_rate_limit(self):
        """Additional manual rate limiting"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()

    # ==========================================================
    # SAFE REQUEST (RETRY + PROXY ROTATION)
    # ==========================================================
    def _safe_request(self, func, *args, retries: int = 3, backoff: int = 2, **kwargs):
        last_exc = None

        for attempt in range(1, retries + 1):
            try:
                self._enforce_rate_limit()
                return func(*args, **kwargs)

            except ccxt.RequestTimeout as e:  # ✅ TAMBAH SPECIFIC EXCEPTION
                last_exc = e
                logger.warning(f"[RequestTimeout retry {attempt}/{retries}] {str(e)[:100]}")
                
                # Increase timeout untuk attempt berikutnya
                if attempt < retries:
                    self.exchange.timeout = min(60000, self.exchange.timeout + 10000)  # max 60s
                    logger.info(f"Increased timeout to {self.exchange.timeout}ms")
                
                # Rotate proxy
                try:
                    proxy = self.proxy_manager.get_working_proxy()
                    self._apply_proxy(proxy)
                    logger.info(f"Proxy rotated due to timeout: {proxy}")
                except Exception as proxy_error:
                    logger.error(f"Proxy rotation failed: {proxy_error}")
                
                time.sleep(backoff * attempt * 2)  # Longer sleep for timeouts

            except ccxt.NetworkError as e:
                last_exc = e
                logger.warning(f"[NetworkError retry {attempt}/{retries}] {type(e).__name__}")
                
                try:
                    proxy = self.proxy_manager.get_working_proxy()
                    self._apply_proxy(proxy)
                    logger.info(f"Proxy rotated: {proxy}")
                except Exception as proxy_error:
                    logger.error(f"Proxy rotation failed: {proxy_error}")
                
                time.sleep(backoff * attempt)

            except ccxt.ExchangeError as e:
                last_exc = e
                error_msg = str(e).lower()
                
                if "insufficient balance" in error_msg:
                    logger.error("Insufficient balance")
                    raise  # Don't retry for balance errors
                elif "rate limit" in error_msg:
                    logger.warning("Rate limit exceeded, sleeping 10s")
                    time.sleep(10)
                    continue
                elif "maintenance" in error_msg:
                    logger.error("Exchange under maintenance")
                    raise
                    
                logger.warning(f"[ExchangeError retry {attempt}/{retries}] {type(e).__name__}")
                time.sleep(backoff * attempt)

            except Exception as e:
                last_exc = e
                logger.warning(f"[GeneralError retry {attempt}/{retries}] {type(e).__name__}: {str(e)[:100]}")
                time.sleep(backoff * attempt)

        if last_exc:
            logger.error(f"All retries failed: {type(last_exc).__name__}")
            raise last_exc
        else:
            raise Exception("Max retries exceeded")

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
    
    def get_ticker(self, symbol):
        """Get ticker with caching"""
        cache_timeout = 2  # seconds
        
        if (symbol in self._ticker_cache and 
            time.time() - self._ticker_cache_time.get(symbol, 0) < cache_timeout):
            return self._ticker_cache[symbol]
        
        ticker = self._safe_request(self.exchange.fetch_ticker, symbol)
        self._ticker_cache[symbol] = ticker
        self._ticker_cache_time[symbol] = time.time()
        return ticker
    
    # ==========================================================
    # GET BALANCE
    # ==========================================================
    def get_balance(self, asset):
        """Get balance with caching"""
        cache_timeout = 5  # seconds
        
        if (self._balance_cache is None or 
            time.time() - self._balance_cache_time > cache_timeout):
            self._balance_cache = self._safe_request(self.exchange.fetch_balance)
            self._balance_cache_time = time.time()

        asset_upper = asset.upper()
        return float(self._balance_cache.get(asset_upper, {}).get("free", 0))

    # ==========================================================
    # ORDERS
    # ==========================================================
    def safe_market_buy(self, symbol, quote_amount):
        """
        Market buy using quote amount (e.g., buy $100 worth of BTC)
        Returns order info or None if failed
        """
        # Get market info for validation
        market = self.exchange.market(symbol)
        min_notional = market.get('limits', {}).get('cost', {}).get('min', 10.0)
        
        # Validate minimum notional
        if quote_amount < min_notional:
            logger.warning(f"[SKIP] Below min notional ${min_notional:.2f} for {symbol}")
            return None
        
        # Get current price
        ticker = self.get_ticker(symbol)
        price = float(ticker['last'])
        
        # Calculate base amount
        base_amount = quote_amount / price
        
        # Validate minimum amount
        min_amount = market['limits']['amount']['min']
        if base_amount < min_amount:
            logger.warning(f"[SKIP] Base amount {base_amount:.8f} below minimum {min_amount:.8f}")
            return None
        
        # Validate sufficient balance
        quote_asset = symbol.split('/')[1]
        free_balance = self.get_balance(quote_asset)
        if free_balance < quote_amount:
            logger.warning(f"[SKIP] Insufficient {quote_asset} balance: {free_balance:.2f} < {quote_amount:.2f}")
            return None
        
        # Create order
        logger.info(f"Buying {base_amount:.6f} {symbol} for ${quote_amount:.2f} @ ${price:.2f}")
        try:
            order = self._safe_request(
                self.exchange.create_market_buy_order,
                symbol,
                base_amount
            )
            logger.info(f"Buy order executed: {order.get('id', 'N/A')}")
            return order
        except Exception as e:
            logger.error(f"Buy order failed: {e}")
            return None
    
    def safe_market_sell(self, symbol, amount):
        """
        Market sell using base amount (e.g., sell 0.01 BTC)
        Returns order info or None if failed
        """
        # Validate state
        if not self.state.can_sell():
            logger.warning("[GUARD] No position, SELL blocked → HOLD")
            return None
        
        # Get base asset and balance
        base = symbol.split("/")[0]
        free_balance = self.get_balance(base)
        
        if free_balance <= 0:
            logger.warning(f"[SKIP] No {base} balance to sell")
            return None
        
        # Adjust sell amount (leave small buffer)
        sell_amount = min(amount, free_balance * 0.999)
        
        # Get market info for validation
        market = self.exchange.market(symbol)
        
        # Validate minimum amount
        min_amount = market['limits']['amount']['min']
        if sell_amount < min_amount:
            logger.warning(f"[SKIP] Sell amount {sell_amount:.8f} below minimum {min_amount:.8f}")
            return None
        
        # Validate minimum notional
        ticker = self.get_ticker(symbol)
        price = float(ticker['last'])
        notional_value = sell_amount * price
        min_notional = market.get('limits', {}).get('cost', {}).get('min', 5.0)
        
        if notional_value < min_notional:
            logger.warning(f"[SKIP] Notional value ${notional_value:.2f} below minimum ${min_notional:.2f}")
            return None
        
        # Create order
        logger.info(f"Selling {sell_amount:.6f} {symbol} @ ${price:.2f} (value: ${notional_value:.2f})")
        try:
            order = self._safe_request(
                self.exchange.create_market_sell_order,
                symbol,
                sell_amount
            )
            logger.info(f"Sell order executed: {order.get('id', 'N/A')}")
            return order
        except Exception as e:
            logger.error(f"Sell order failed: {e}")
            return None

    # ========================================================
    # FETCH EQUITY
    # ========================================================
    def get_equity(self, symbol="BTC/USDT"):
        """Calculate total portfolio value in quote currency"""
        try:
            balance = self._safe_request(self.exchange.fetch_balance)
            base, quote = symbol.split("/")

            free_quote = balance["free"].get(quote, 0)
            free_base = balance["free"].get(base, 0)
            
            ticker = self.get_ticker(symbol)
            price = float(ticker["last"])

            equity = free_quote + free_base * price
            return float(equity)
        except Exception as e:
            logger.error(f"Failed to calculate equity: {e}")
            return 0.0
    
    # ========================================================
    # APPLY PROXY
    # ========================================================
    def _apply_proxy(self, proxy):
        """Apply proxy to exchange session"""
        if proxy and hasattr(self.exchange, "session"):
            self.exchange.session.proxies = {"http": proxy, "https": proxy}

    # ========================================================
    # SIMPLER MARKET ORDER METHODS (for backward compatibility)
    # ========================================================
    def market_buy(self, symbol, size):
        """Simple market buy - accepts base amount (legacy compatibility)"""
        return self._safe_request(
            self.exchange.create_market_buy_order,
            symbol,
            size
        )
    
    def market_sell(self, symbol, size):
        """Simple market sell - accepts base amount (legacy compatibility)"""
        return self._safe_request(
            self.exchange.create_market_sell_order,
            symbol,
            size
        )

    def test_connection(self, max_attempts=3):
        """Test connection to exchange with retry"""
        for attempt in range(1, max_attempts + 1):
            try:
                logger.info(f"Connection test attempt {attempt}/{max_attempts}")
                
                # Test 1: Fetch time (lightweight)
                server_time = self.exchange.fetch_time()
                logger.info(f"✓ Server time: {server_time}")
                
                # Test 2: Fetch ticker
                test_symbol = "BTC/USDT"
                ticker = self.exchange.fetch_ticker(test_symbol)
                logger.info(f"✓ Ticker {test_symbol}: ${ticker['last']}")
                
                # Test 3: Check if authenticated
                if self.exchange.check_required_credentials():
                    logger.info("✓ API credentials valid")
                
                logger.info("✅ All connection tests passed")
                return True
                
            except ccxt.RequestTimeout as e:
                logger.warning(f"Connection timeout (attempt {attempt}): {e}")
                if attempt < max_attempts:
                    time.sleep(5 * attempt)
                    # Try with different timeout
                    self.exchange.timeout = min(60000, 20000 + (attempt * 10000))
            except Exception as e:
                logger.error(f"Connection test failed: {type(e).__name__}: {e}")
                return False
        
        logger.error("❌ All connection tests failed")
        return False