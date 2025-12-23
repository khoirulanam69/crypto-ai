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
        self.debug = os.getenv("DEBUG", "false").lower() == "true"
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests
        
        # Cache initialization
        self._balance_cache = None
        self._balance_cache_time = 0
        self._ticker_cache = {}
        self._ticker_cache_time = {}
        self._market_cache = {}  # Cache for market info

        # ======================
        # API KEYS - SECURITY WARNING
        # ======================
        api_key = os.getenv("BINANCE_API_KEY")
        secret = os.getenv("BINANCE_API_SECRET")

        if not api_key or not secret:
            raise RuntimeError("API credentials not configured. Set BINANCE_API_KEY and BINANCE_API_SECRET in .env")
        
        # Mask keys in logs
        def mask_key(key):
            if not key or len(key) < 8:
                return "***"
            return key[:4] + "***" + key[-4:]
        
        logger.info(f"API Key: {mask_key(api_key)}")
        logger.info(f"Testnet mode: {testnet}")

        # ======================
        # DNS + PROXY
        # ======================
        # DNS Fallback
        fallback_ips_str = os.getenv("DNS_FALLBACK_IPS", "")
        fallback_ips = [ip.strip() for ip in fallback_ips_str.split(",") if ip.strip()]
        if not fallback_ips:
            fallback_ips = ["18.162.165.240", "18.181.3.53"]  # Binance Hong Kong IPs
        
        self.resolver = DNSResolver(fallback_ips=fallback_ips)

        # Proxy configuration
        proxies = os.getenv("PROXIES", "")
        proxy_list = [p.strip() for p in proxies.split(",") if p.strip()]
        self.proxy_manager = ProxyManager(proxy_list)
        
        if proxy_list:
            logger.info(f"Proxy manager initialized with {len(proxy_list)} proxies")
        else:
            logger.info("No proxies configured, using direct connection")

        # ======================
        # EXCHANGE INIT
        # ======================
        exchange_config = {
            "apiKey": api_key,
            "secret": secret,
            "enableRateLimit": True,  # Use CCXT's built-in rate limiter
            "timeout": 30000,  # 30 seconds
            "rateLimit": 1000,  # requests per minute
            "options": {
                "adjustForTimeDifference": True,
                "recvWindow": 60000,  # 60 seconds
                "defaultType": "spot",
                "warnOnFetchOHLCVLimitArgument": False,
            },
            "headers": {
                "User-Agent": "crypto-ai-bot/1.0"
            }
        }
        
        # Testnet configuration
        if testnet:
            exchange_config.update({
                "urls": {
                    "api": {
                        "public": "https://testnet.binance.vision/api",
                        "private": "https://testnet.binance.vision/api",
                    }
                }
            })
            logger.info("Using Binance TESTNET")
        
        # Create exchange instance
        self.exchange = ccxt.binance(exchange_config)
        
        # Configure proxy support
        self.exchange.aiohttp_trust_env = True  # Allow proxy from env
        
        # WARNING: verbose mode can log sensitive data
        if self.debug:
            logger.warning("⚠️ DEBUG MODE ENABLED - Be careful with sensitive data in logs")
            # Consider if you really need verbose mode
            # self.exchange.verbose = True  # DANGEROUS - logs API keys!

        # ======================
        # TIME SYNC (CRITICAL)
        # ======================
        try:
            self.exchange.load_time_difference()
            time_diff = self.exchange.time_difference
            logger.info(f"Time sync successful. Server-client difference: {time_diff}ms")
        except Exception as e:
            logger.warning(f"Time sync failed: {e}. Trading may fail due to timestamp issues.")

        # ===============================
        # INIT STATE MANAGER
        # ===============================
        initial_cash = float(os.getenv("INITIAL_CASH", "0"))
        self.state = StateManager(initial_cash)
        logger.info(f"StateManager initialized with ${initial_cash:.2f}")

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
            logger.error(f"Proxy initialization error: {e}")
            # Continue without proxy

    # ==========================================================
    # RATE LIMIT ENFORCEMENT (ADDITIONAL)
    # ==========================================================
    def _enforce_rate_limit(self):
        """Additional manual rate limiting on top of CCXT's."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            sleep_time = self.min_request_interval - elapsed
            time.sleep(sleep_time)
        self.last_request_time = time.time()

    # ==========================================================
    # SAFE REQUEST (RETRY + PROXY ROTATION) - IMPROVED
    # ==========================================================
    def _safe_request(self, func, *args, retries: int = 3, backoff: int = 2, **kwargs):
        last_exc = None
        original_timeout = self.exchange.timeout

        for attempt in range(1, retries + 1):
            try:
                # Apply additional rate limiting
                self._enforce_rate_limit()
                
                # Execute the function
                result = func(*args, **kwargs)
                
                # Reset timeout to original if it was increased
                if self.exchange.timeout != original_timeout:
                    self.exchange.timeout = original_timeout
                
                return result

            except ccxt.RequestTimeout as e:
                last_exc = e
                logger.warning(f"[RequestTimeout retry {attempt}/{retries}] {type(e).__name__}")
                
                # Increase timeout for next attempt (max 60s)
                if attempt < retries:
                    new_timeout = min(60000, self.exchange.timeout + 10000)
                    self.exchange.timeout = new_timeout
                    logger.debug(f"Increased timeout to {new_timeout}ms")
                
                # Rotate proxy on timeout
                self._rotate_proxy(f"timeout (attempt {attempt})")
                
                # Longer sleep for timeouts
                sleep_time = backoff * attempt * 2
                logger.debug(f"Sleeping {sleep_time}s before retry")
                time.sleep(sleep_time)

            except ccxt.NetworkError as e:
                last_exc = e
                logger.warning(f"[NetworkError retry {attempt}/{retries}] {type(e).__name__}")
                
                # Rotate proxy on network error
                self._rotate_proxy(f"network error (attempt {attempt})")
                
                sleep_time = backoff * attempt
                logger.debug(f"Sleeping {sleep_time}s before retry")
                time.sleep(sleep_time)

            except ccxt.ExchangeError as e:
                last_exc = e
                error_msg = str(e).lower()
                
                # Don't retry for these errors
                if "insufficient balance" in error_msg:
                    logger.error("Insufficient balance - aborting")
                    raise ccxt.InsufficientFunds(str(e)) from e
                elif "rate limit" in error_msg or "too many requests" in error_msg:
                    logger.warning("Rate limit exceeded, sleeping 10s")
                    time.sleep(10)
                    continue  # Retry after sleep
                elif "maintenance" in error_msg or "system busy" in error_msg:
                    logger.error("Exchange under maintenance - aborting")
                    raise
                elif "invalid order" in error_msg or "precision" in error_msg:
                    logger.error(f"Invalid order parameters: {e}")
                    raise  # Don't retry parameter errors
                    
                logger.warning(f"[ExchangeError retry {attempt}/{retries}] {type(e).__name__}: {error_msg[:100]}")
                time.sleep(backoff * attempt)

            except ccxt.AuthenticationError as e:
                logger.error(f"Authentication error: {e}")
                raise  # Don't retry auth errors

            except Exception as e:
                last_exc = e
                logger.warning(f"[GeneralError retry {attempt}/{retries}] {type(e).__name__}: {str(e)[:100]}")
                time.sleep(backoff * attempt)

        # All retries failed
        if last_exc:
            logger.error(f"All {retries} retries failed. Last error: {type(last_exc).__name__}: {str(last_exc)[:200]}")
            raise last_exc
        else:
            raise Exception(f"Max retries ({retries}) exceeded for {func.__name__}")

    def _rotate_proxy(self, reason: str):
        """Rotate to a different proxy."""
        try:
            proxy = self.proxy_manager.get_working_proxy(rotate=True)
            if proxy:
                self._apply_proxy(proxy)
                logger.info(f"Proxy rotated ({reason}): {proxy}")
            else:
                logger.debug(f"No proxy available to rotate ({reason})")
        except Exception as e:
            logger.error(f"Proxy rotation failed: {e}")

    # ==========================================================
    # MARKET DATA - IMPROVED
    # ==========================================================
    def fetch_ohlcv(self, symbol, timeframe="1m", limit=100):
        """Fetch OHLCV data with validation."""
        try:
            # Validate symbol format
            if '/' not in symbol:
                raise ValueError(f"Invalid symbol format: {symbol}. Expected 'BTC/USDT'")
            
            ohlcv = self._safe_request(
                self.exchange.fetch_ohlcv,
                symbol,
                timeframe=timeframe,
                limit=limit
            )
            
            # Validate OHLCV data
            if not ohlcv or len(ohlcv) == 0:
                logger.warning(f"No OHLCV data returned for {symbol}")
                return []
            
            # Check for valid price data
            for i, candle in enumerate(ohlcv[-5:]):  # Check last 5 candles
                if len(candle) < 6:
                    logger.warning(f"Invalid candle format at index {i}: {candle}")
                    continue
                
                open_price, high, low, close = candle[1], candle[2], candle[3], candle[4]
                if not all(isinstance(p, (int, float)) and p > 0 for p in [open_price, high, low, close]):
                    logger.warning(f"Invalid price in candle {i}: {candle}")
            
            return ohlcv
            
        except Exception as e:
            logger.error(f"Failed to fetch OHLCV for {symbol}: {e}")
            raise
    
    def get_ticker(self, symbol, cache_timeout: int = 2):
        """Get ticker with caching and validation."""
        # Clean symbol for cache key
        cache_key = symbol.replace('/', '').upper()
        
        # Check cache
        current_time = time.time()
        if (cache_key in self._ticker_cache and 
            current_time - self._ticker_cache_time.get(cache_key, 0) < cache_timeout):
            return self._ticker_cache[cache_key]
        
        try:
            ticker = self._safe_request(self.exchange.fetch_ticker, symbol)
            
            # Validate ticker data
            required_fields = ['last', 'bid', 'ask', 'high', 'low', 'volume']
            for field in required_fields:
                if field not in ticker:
                    logger.warning(f"Ticker missing field {field}: {ticker}")
                    ticker[field] = 0.0
            
            # Ensure numeric values
            ticker['last'] = float(ticker['last'])
            ticker['bid'] = float(ticker.get('bid', ticker['last']))
            ticker['ask'] = float(ticker.get('ask', ticker['last']))
            
            # Update cache
            self._ticker_cache[cache_key] = ticker
            self._ticker_cache_time[cache_key] = current_time
            
            return ticker
            
        except Exception as e:
            logger.error(f"Failed to fetch ticker for {symbol}: {e}")
            # Return dummy ticker if failed
            return {
                'last': 0.0,
                'bid': 0.0,
                'ask': 0.0,
                'high': 0.0,
                'low': 0.0,
                'volume': 0.0,
                'symbol': symbol,
                'timestamp': int(time.time() * 1000)
            }
    
    # ==========================================================
    # BALANCE & ACCOUNT INFO
    # ==========================================================
    def get_balance(self, asset: str, use_cache: bool = True):
        """Get balance with caching and invalidation."""
        cache_timeout = 5  # seconds
        
        # Force refresh if cache disabled or cache expired
        if (not use_cache or 
            self._balance_cache is None or 
            time.time() - self._balance_cache_time > cache_timeout):
            
            try:
                self._balance_cache = self._safe_request(self.exchange.fetch_balance)
                self._balance_cache_time = time.time()
                logger.debug("Balance cache refreshed")
            except Exception as e:
                logger.error(f"Failed to fetch balance: {e}")
                # Return cached value if available, otherwise 0
                if self._balance_cache is None:
                    return 0.0
        
        # Extract balance
        asset_upper = asset.upper()
        if asset_upper in self._balance_cache.get('free', {}):
            balance = float(self._balance_cache['free'][asset_upper])
        elif asset_upper in self._balance_cache.get('total', {}):
            balance = float(self._balance_cache['total'][asset_upper])
        else:
            balance = 0.0
        
        return balance
    
    def invalidate_balance_cache(self):
        """Invalidate balance cache (call after trades)."""
        self._balance_cache = None
        logger.debug("Balance cache invalidated")
    
    # ==========================================================
    # ORDER MANAGEMENT - IMPROVED
    # ==========================================================
    def get_market_info(self, symbol: str):
        """Get market information (limits, precision)."""
        if symbol in self._market_cache:
            return self._market_cache[symbol]
        
        try:
            market = self.exchange.market(symbol)
            
            # Extract important limits
            market_info = {
                'symbol': symbol,
                'limits': {
                    'amount': {
                        'min': float(market.get('limits', {}).get('amount', {}).get('min', 0.00001)),
                        'max': float(market.get('limits', {}).get('amount', {}).get('max', 100000))
                    },
                    'cost': {
                        'min': float(market.get('limits', {}).get('cost', {}).get('min', 10.0)),  # Binance default
                        'max': float(market.get('limits', {}).get('cost', {}).get('max', 1000000))
                    },
                    'price': {
                        'min': float(market.get('limits', {}).get('price', {}).get('min', 0.01)),
                        'max': float(market.get('limits', {}).get('price', {}).get('max', 1000000))
                    }
                },
                'precision': {
                    'amount': int(market.get('precision', {}).get('amount', 8)),
                    'price': int(market.get('precision', {}).get('price', 2))
                },
                'base': market.get('base', ''),
                'quote': market.get('quote', ''),
                'active': market.get('active', True)
            }
            
            self._market_cache[symbol] = market_info
            return market_info
            
        except Exception as e:
            logger.error(f"Failed to get market info for {symbol}: {e}")
            # Return safe defaults
            return {
                'symbol': symbol,
                'limits': {
                    'amount': {'min': 0.00001, 'max': 100000},
                    'cost': {'min': 10.0, 'max': 1000000},
                    'price': {'min': 0.01, 'max': 1000000}
                },
                'precision': {'amount': 8, 'price': 2}
            }
    
    def safe_market_buy(self, symbol, quote_amount):
        """
        Market buy using quote amount with comprehensive validation.
        Returns (success, order_info_or_error_message)
        """
        try:
            # Input validation
            if quote_amount <= 0:
                return False, "Invalid quote amount (must be positive)"
            
            # Get market info
            market = self.get_market_info(symbol)
            min_notional = market['limits']['cost']['min']
            
            # Check minimum notional
            if quote_amount < min_notional:
                return False, f"Below minimum notional ${min_notional:.2f}"
            
            # Get current price
            ticker = self.get_ticker(symbol)
            price = ticker['last']
            if price <= 0:
                return False, f"Invalid price: ${price:.2f}"
            
            # Calculate base amount
            base_amount = quote_amount / price
            
            # Check minimum amount
            min_amount = market['limits']['amount']['min']
            if base_amount < min_amount:
                return False, f"Base amount {base_amount:.8f} below minimum {min_amount:.8f}"
            
            # Check balance
            quote_asset = symbol.split('/')[1]
            free_balance = self.get_balance(quote_asset)
            if free_balance < quote_amount:
                return False, f"Insufficient {quote_asset} balance: {free_balance:.2f} < {quote_amount:.2f}"
            
            # Create order
            logger.info(f"Buying {base_amount:.6f} {symbol} for ${quote_amount:.2f} @ ${price:.2f}")
            
            # Apply precision rounding
            amount_precision = market['precision']['amount']
            base_amount_rounded = round(base_amount, amount_precision)
            
            order = self._safe_request(
                self.exchange.create_market_buy_order,
                symbol,
                base_amount_rounded
            )
            
            # Invalidate caches
            self.invalidate_balance_cache()
            self._ticker_cache.pop(symbol.replace('/', '').upper(), None)
            
            logger.info(f"Buy order executed: {order.get('id', 'N/A')}")
            return True, order
            
        except ccxt.InsufficientFunds as e:
            logger.error(f"Insufficient funds: {e}")
            return False, f"Insufficient funds: {str(e)}"
        except ccxt.InvalidOrder as e:
            logger.error(f"Invalid order: {e}")
            return False, f"Invalid order: {str(e)}"
        except Exception as e:
            logger.error(f"Buy order failed: {e}")
            return False, f"Buy failed: {str(e)}"
    
    def safe_market_sell(self, symbol, amount):
        """
        Market sell with comprehensive validation.
        Returns (success, order_info_or_error_message)
        """
        try:
            # Input validation
            if amount <= 0:
                return False, "Invalid amount (must be positive)"
            
            # Get market info
            market = self.get_market_info(symbol)
            
            # Check minimum amount
            min_amount = market['limits']['amount']['min']
            if amount < min_amount:
                return False, f"Sell amount {amount:.8f} below minimum {min_amount:.8f}"
            
            # Get base asset and check balance
            base = symbol.split("/")[0]
            free_balance = self.get_balance(base)
            
            if free_balance <= 0:
                return False, f"No {base} balance to sell"
            
            # Adjust sell amount (leave 0.1% buffer for fees/slippage)
            sell_amount = min(amount, free_balance * 0.999)
            
            # Get current price and check notional
            ticker = self.get_ticker(symbol)
            price = ticker['last']
            notional_value = sell_amount * price
            
            min_notional = market['limits']['cost']['min']
            if notional_value < min_notional:
                return False, f"Notional value ${notional_value:.2f} below minimum ${min_notional:.2f}"
            
            # Apply precision rounding
            amount_precision = market['precision']['amount']
            sell_amount_rounded = round(sell_amount, amount_precision)
            
            # Create order
            logger.info(f"Selling {sell_amount_rounded:.6f} {symbol} @ ${price:.2f} (value: ${notional_value:.2f})")
            
            order = self._safe_request(
                self.exchange.create_market_sell_order,
                symbol,
                sell_amount_rounded
            )
            
            # Invalidate caches
            self.invalidate_balance_cache()
            self._ticker_cache.pop(symbol.replace('/', '').upper(), None)
            
            logger.info(f"Sell order executed: {order.get('id', 'N/A')}")
            return True, order
            
        except ccxt.InsufficientFunds as e:
            logger.error(f"Insufficient funds: {e}")
            return False, f"Insufficient funds: {str(e)}"
        except ccxt.InvalidOrder as e:
            logger.error(f"Invalid order: {e}")
            return False, f"Invalid order: {str(e)}"
        except Exception as e:
            logger.error(f"Sell order failed: {e}")
            return False, f"Sell failed: {str(e)}"
    
    # ========================================================
    # PORTFOLIO & EQUITY
    # ========================================================
    def get_equity(self, symbol="BTC/USDT"):
        """Calculate total portfolio value in quote currency."""
        try:
            # Don't use cache for equity calculation
            balance = self._safe_request(self.exchange.fetch_balance)
            base, quote = symbol.split("/")

            free_quote = balance["free"].get(quote, 0)
            free_base = balance["free"].get(base, 0)
            
            ticker = self.get_ticker(symbol, cache_timeout=1)  # Fresh price
            price = float(ticker["last"])

            equity = free_quote + free_base * price
            logger.debug(f"Equity calculation: ${equity:.2f} (${free_quote:.2f} + {free_base:.6f} × ${price:.2f})")
            
            return float(equity)
            
        except Exception as e:
            logger.error(f"Failed to calculate equity: {e}")
            return 0.0
    
    def get_portfolio_summary(self, symbol="BTC/USDT"):
        """Get detailed portfolio summary."""
        try:
            balance = self._safe_request(self.exchange.fetch_balance)
            base, quote = symbol.split("/")
            
            free_quote = float(balance["free"].get(quote, 0))
            free_base = float(balance["free"].get(base, 0))
            total_quote = float(balance["total"].get(quote, 0))
            total_base = float(balance["total"].get(base, 0))
            
            ticker = self.get_ticker(symbol)
            price = float(ticker["last"])
            
            used_quote = total_quote - free_quote
            used_base = total_base - free_base
            
            portfolio_value = free_quote + free_base * price
            total_value = total_quote + total_base * price
            
            return {
                'cash': free_quote,
                'position': free_base,
                'price': price,
                'portfolio_value': portfolio_value,
                'total_value': total_value,
                'used_cash': used_quote,
                'used_position': used_base,
                'base_asset': base,
                'quote_asset': quote
            }
            
        except Exception as e:
            logger.error(f"Failed to get portfolio summary: {e}")
            return None
    
    # ========================================================
    # PROXY MANAGEMENT
    # ========================================================
    def _apply_proxy(self, proxy):
        """Apply proxy to exchange connection."""
        if not proxy:
            # Clear any existing proxy
            self._clear_proxy()
            return
        
        try:
            # Store current proxy
            self.current_proxy = proxy
            
            # Multiple methods to apply proxy
            success = False
            
            # Method 1: CCXT built-in proxies
            if hasattr(self.exchange, 'proxies'):
                self.exchange.proxies = {
                    'http': proxy,
                    'https': proxy,
                }
                logger.debug(f"Proxy applied via ccxt.proxies: {proxy}")
                success = True
            
            # Method 2: Requests session
            if hasattr(self.exchange, 'session') and hasattr(self.exchange.session, 'proxies'):
                self.exchange.session.proxies = {
                    'http': proxy,
                    'https': proxy,
                }
                logger.debug(f"Proxy applied via session.proxies: {proxy}")
                success = True
            
            # Method 3: Environment variables (for aiohttp)
            os.environ['HTTP_PROXY'] = proxy
            os.environ['HTTPS_PROXY'] = proxy
            self.exchange.aiohttp_trust_env = True
            logger.debug(f"Proxy set via env vars: {proxy}")
            success = True
            
            if not success:
                logger.warning(f"Could not apply proxy {proxy}. Exchange backend may not support proxies.")
            else:
                logger.info(f"Proxy applied: {proxy}")
                
        except Exception as e:
            logger.error(f"Failed to apply proxy {proxy}: {e}")
            self._clear_proxy()
    
    def _clear_proxy(self):
        """Clear proxy settings."""
        try:
            self.current_proxy = None
            
            # Clear from exchange
            if hasattr(self.exchange, 'proxies'):
                self.exchange.proxies = None
            
            if hasattr(self.exchange, 'session') and hasattr(self.exchange.session, 'proxies'):
                self.exchange.session.proxies = None
            
            # Clear env vars
            os.environ.pop('HTTP_PROXY', None)
            os.environ.pop('HTTPS_PROXY', None)
            
            logger.debug("Proxy cleared")
        except Exception as e:
            logger.error(f"Failed to clear proxy: {e}")
    
    # ========================================================
    # SIMPLER ORDER METHODS (for backward compatibility)
    # ========================================================
    def market_buy(self, symbol, size):
        """Simple market buy - accepts base amount."""
        try:
            order = self._safe_request(
                self.exchange.create_market_buy_order,
                symbol,
                size
            )
            self.invalidate_balance_cache()
            return order
        except Exception as e:
            logger.error(f"Market buy failed: {e}")
            raise
    
    def market_sell(self, symbol, size):
        """Simple market sell - accepts base amount."""
        try:
            order = self._safe_request(
                self.exchange.create_market_sell_order,
                symbol,
                size
            )
            self.invalidate_balance_cache()
            return order
        except Exception as e:
            logger.error(f"Market sell failed: {e}")
            raise
    
    # ========================================================
    # CONNECTION TESTING - IMPROVED
    # ========================================================
    def test_connection(self, max_attempts=3):
        """Test connection to exchange with comprehensive checks."""
        logger.info("Testing exchange connection...")
        
        tests_passed = 0
        total_tests = 4
        
        for attempt in range(1, max_attempts + 1):
            try:
                logger.info(f"Connection test attempt {attempt}/{max_attempts}")
                
                # Test 1: Server time (lightweight)
                server_time = self._safe_request(
                    self.exchange.fetch_time,
                    retries=1,
                    backoff=2
                )
                logger.info(f"✓ Server time: {server_time}")
                tests_passed += 1
                
                # Test 2: System status
                try:
                    status = self._safe_request(
                        self.exchange.fetch_status,
                        retries=1,
                        backoff=2
                    )
                    if status and status.get('status') == 'ok':
                        logger.info("✓ System status: OK")
                        tests_passed += 1
                    else:
                        logger.warning(f"System status: {status}")
                except Exception as e:
                    logger.debug(f"System status check not available: {e}")
                
                # Test 3: Ticker (market data)
                test_symbol = "BTC/USDT"
                ticker = self._safe_request(
                    self.exchange.fetch_ticker,
                    test_symbol,
                    retries=1,
                    backoff=2
                )
                logger.info(f"✓ Ticker {test_symbol}: ${ticker['last']}")
                tests_passed += 1
                
                # Test 4: Authentication (fetch balance)
                if self.exchange.check_required_credentials():
                    balance = self._safe_request(
                        self.exchange.fetch_balance,
                        retries=1,
                        backoff=2
                    )
                    logger.info("✓ API credentials valid")
                    tests_passed += 1
                
                # Success criteria
                if tests_passed >= 3:  # At least 3/4 tests passed
                    logger.info(f"✅ Connection test passed ({tests_passed}/{total_tests} tests)")
                    return True
                else:
                    logger.warning(f"Connection test partial: {tests_passed}/{total_tests} tests passed")
                    
            except ccxt.AuthenticationError as e:
                logger.error(f"Authentication failed: {e}")
                return False
            except ccxt.NetworkError as e:
                logger.warning(f"Network error (attempt {attempt}): {e}")
                if attempt < max_attempts:
                    time.sleep(5 * attempt)
                continue
            except Exception as e:
                logger.warning(f"Connection test failed (attempt {attempt}): {type(e).__name__}: {e}")
                if attempt < max_attempts:
                    time.sleep(5 * attempt)
                continue
        
        logger.error(f"❌ Connection test failed after {max_attempts} attempts")
        return False
    
    # ========================================================
    # UTILITIES
    # ========================================================
    def get_exchange_info(self):
        """Get exchange information and capabilities."""
        try:
            info = {
                'name': self.exchange.name,
                'version': self.exchange.version,
                'timeout': self.exchange.timeout,
                'rateLimit': self.exchange.rateLimit,
                'has': self.exchange.has,
                'testnet': self.testnet,
                'proxy': self.current_proxy,
                'api_key_masked': '...' + (self.exchange.apiKey[-4:] if self.exchange.apiKey else ''),
                'required_credentials': self.exchange.requiredCredentials
            }
            return info
        except Exception as e:
            logger.error(f"Failed to get exchange info: {e}")
            return {}
    
    def __str__(self):
        """String representation."""
        info = self.get_exchange_info()
        return (f"OrderManager({info.get('name', 'Unknown')}, "
                f"testnet={info.get('testnet', False)}, "
                f"proxy={'Yes' if info.get('proxy') else 'No'})")