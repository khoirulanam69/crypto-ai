import requests
import socket
import re

class DNSResolver:
    def __init__(self, primary='cloudflare', fallback_ips=None, timeout=5):
        self.primary = primary
        self.timeout = timeout
        self.fallback_ips = fallback_ips or []

        # Regex validasi IPv4
        self.ipv4_pattern = re.compile(
            r"^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$"
        )

    # --------------------------
    # VALIDATOR
    # --------------------------
    def _is_valid_ipv4(self, ip):
        if not ip:
            return False
        if not isinstance(ip, str):
            return False
        if not self.ipv4_pattern.match(ip):
            return False
        return all(0 <= int(part) <= 255 for part in ip.split("."))

    # --------------------------
    # DoH Resolver
    # --------------------------
    def doh_query(self, domain):
        try:
            if self.primary == 'google':
                url = 'https://dns.google/resolve'
                params = {'name': domain, 'type': 'A'}
                headers = {}
            else:
                url = 'https://cloudflare-dns.com/dns-query'
                params = {'name': domain, 'type': 'A'}
                headers = {'accept': 'application/dns-json'}

            r = requests.get(url, params=params, headers=headers, timeout=self.timeout)
            r.raise_for_status()

            data = r.json()
            answers = data.get('Answer') or data.get('answer') or []

            result_ip = None
            for a in answers:
                # Format dictionary
                if isinstance(a, dict):
                    ip = a.get("data")
                    if self._is_valid_ipv4(ip):
                        result_ip = ip
                        break
                # Format string
                elif isinstance(a, str):
                    if self._is_valid_ipv4(a):
                        result_ip = a
                        break

            return result_ip
        except:
            return None

    # --------------------------
    # SOCKET Resolver
    # --------------------------
    def socket_lookup(self, domain):
        try:
            ip = socket.gethostbyname(domain)
            if self._is_valid_ipv4(ip):
                return ip
        except:
            return None
        return None

    # --------------------------
    # FINAL RESOLVE FLOW
    # --------------------------
    def resolve(self, domain):
        # 1. DoH lookup
        ip = self.doh_query(domain)
        if ip:
            return ip

        # 2. Socket lookup
        ip = self.socket_lookup(domain)
        if ip:
            return ip

        # 3. Hard fallback IPs
        for fb in self.fallback_ips:
            if self._is_valid_ipv4(fb):
                return fb

        return None
