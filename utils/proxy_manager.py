import os
import requests
import time
from typing import List, Optional

class ProxyManager:
    def __init__(self, proxies: List[str] = None, timeout: int = 8):
        """
        PROXIES dapat di-load otomatis dari environment (.env):
        
        PROXIES=http://1.1.1.1:3128, http://2.2.2.2:8080, socks5://3.3.3.3:1080
        """

        # Load proxy list dari ENV
        env_proxies = os.getenv("PROXIES")

        if env_proxies:
            proxies = [p.strip() for p in env_proxies.split(",") if p.strip()]
            print(f"[ProxyManager] Loaded {len(proxies)} proxies from .env")

        self.proxies = proxies or []
        self.timeout = timeout
        self._good = []
        self._last_checked = 0
        self._check_interval = 60

    def is_proxy_ok(self, proxy: str) -> bool:
        """Cek apakah proxy berfungsi."""
        try:
            sess = requests.Session()
            sess.proxies.update({
                'http': proxy,
                'https': proxy
            })
            r = sess.get('https://httpbin.org/ip', timeout=self.timeout)
            return r.status_code == 200
        except Exception:
            return False

    def refresh(self):
        """Refresh daftar proxy yang terbukti bekerja."""
        now = time.time()
        # Tidak usah cek ulang jika masih dalam interval dan sudah ada working proxy
        if now - self._last_checked < self._check_interval and self._good:
            return

        self._good = []
        for p in self.proxies:
            if self.is_proxy_ok(p):
                self._good.append(p)

        self._last_checked = now

    def get_working_proxy(self) -> Optional[str]:
        """Ambil proxy yang bekerja (rotating)."""
        self.refresh()

        if not self._good:
            return None

        proxy = self._good.pop(0)
        self._good.append(proxy)
        return proxy

    def apply_to_session(self, session, proxy: Optional[str]):
        """Pasangkan proxy pada session request."""
        if proxy:
            sess
