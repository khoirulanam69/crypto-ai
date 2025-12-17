def calc_position_size(
    equity: float,
    price: float,
    atr: float,
    risk_pct: float = 0.005,   # 0.5% per trade
    atr_mult: float = 2.0
):
    """
    Risk per trade = equity * risk_pct
    Stop distance = atr * atr_mult
    """
    if atr <= 0 or price <= 0:
        return 0.0

    risk_amount = equity * risk_pct
    stop_distance = atr * atr_mult

    size = risk_amount / stop_distance
    max_size = equity / price * 0.95

    return max(0.0, min(size, max_size))
