# ai/ensemble/rule.py
import numpy as np
import logging

logger = logging.getLogger(__name__)

class RuleBased:
    def __init__(self):
        self.name = "RuleBased"
        self.last_decision_info = {}
        
    def decide(self, state):
        """
        Hybrid strategy combining trend following and mean reversion
        state: normalized prices + cash_ratio + position_ratio
        """
        try:
            # Extract components from state
            prices = state[:-2]  # Normalized prices
            cash_ratio = state[-2]
            position_ratio = state[-1]
            
            # Need minimum data
            if len(prices) < 20:
                self.last_decision_info = {
                    'decision': 1,
                    'reason': f'Insufficient data: {len(prices)} < 20'
                }
                return 1  # HOLD
            
            # Calculate indicators
            current_price = prices[-1]
            
            # 1. Moving averages for trend
            ma_10 = np.mean(prices[-10:]) if len(prices) >= 10 else 1.0
            ma_20 = np.mean(prices[-20:]) if len(prices) >= 20 else 1.0
            ma_50 = np.mean(prices[-min(50, len(prices)):])
            
            # 2. Trend signals
            trend_short = current_price - ma_10  # Short term trend
            trend_medium = current_price - ma_20  # Medium term trend
            trend_long = current_price - ma_50    # Long term trend
            
            # 3. Mean reversion indicators
            # Percentage deviation from mean
            deviation_from_20 = (current_price - ma_20) / ma_20
            
            # 4. Momentum indicators
            if len(prices) >= 5:
                momentum_5 = (current_price - prices[-5]) / prices[-5] * 100  # 5-period % change
            else:
                momentum_5 = 0
                
            if len(prices) >= 10:
                momentum_10 = (current_price - prices[-10]) / prices[-10] * 100  # 10-period % change
            else:
                momentum_10 = 0
            
            # 5. Volatility adjusted thresholds
            volatility = np.std(prices[-10:]) if len(prices) >= 10 else 0.001
            # Higher volatility needs stronger signals
            volatility_multiplier = 1 + (volatility * 5)
            
            # 6. RSI-like indicator
            if len(prices) >= 14:
                gains = []
                losses = []
                for i in range(1, min(15, len(prices))):
                    change = prices[-i] - prices[-i-1]
                    if change > 0:
                        gains.append(change)
                    else:
                        losses.append(abs(change))
                
                avg_gain = np.mean(gains) if gains else 0
                avg_loss = np.mean(losses) if losses else 0.001
                
                if avg_loss == 0:
                    rs = 100
                else:
                    rs = avg_gain / avg_loss
                
                rsi = 100 - (100 / (1 + rs))
            else:
                rsi = 50
            
            # HYBRID DECISION LOGIC
            
            decision = 1  # Default HOLD
            reason = "No clear signal"
            
            # Calculate buy and sell scores
            buy_score = 0
            sell_score = 0
            
            # === BUY SIGNALS ===
            
            # 1. STRONG MEAN REVERSION BUY (oversold)
            if deviation_from_20 < -0.003:  # 0.3% below 20MA
                buy_score += 3
                
            # 2. TREND FOLLOWING BUY
            if trend_short > 0.001 and trend_medium > 0.0005:  # Uptrend
                buy_score += 2
                
            # 3. MOMENTUM BUY
            if momentum_5 > 0.1 and momentum_10 > 0.05:  # Positive momentum
                buy_score += 1
                
            # 4. RSI OVERSOLD BUY
            if rsi < 35:
                buy_score += 2
                
            # 5. GOLDEN CROSS (short MA above medium MA)
            if ma_10 > ma_20:
                buy_score += 1
            
            # === SELL SIGNALS ===
            
            # 1. STRONG MEAN REVERSION SELL (overbought)
            if deviation_from_20 > 0.003:  # 0.3% above 20MA
                sell_score += 3
                
            # 2. TREND FOLLOWING SELL
            if trend_short < -0.001 and trend_medium < -0.0005:  # Downtrend
                sell_score += 2
                
            # 3. MOMENTUM SELL
            if momentum_5 < -0.1 and momentum_10 < -0.05:  # Negative momentum
                sell_score += 1
                
            # 4. RSI OVERBOUGHT SELL
            if rsi > 65:
                sell_score += 2
                
            # 5. DEATH CROSS (short MA below medium MA)
            if ma_10 < ma_20:
                sell_score += 1
            
            # Apply account constraints
            can_buy = cash_ratio > 0.05  # Need at least 5% cash
            can_sell = position_ratio > 0.00001  # Need at least minimal position
            
            # Make final decision
            if buy_score > sell_score and buy_score >= 3 and can_buy:
                decision = 0  # BUY
                reason = (f"BUY: Strong buy signals (score={buy_score}) | "
                         f"Deviation: {deviation_from_20*100:.2f}% | "
                         f"RSI: {rsi:.1f} | Momentum: {momentum_5:.2f}%")
                
            elif sell_score > buy_score and sell_score >= 3 and can_sell:
                decision = 2  # SELL
                reason = (f"SELL: Strong sell signals (score={sell_score}) | "
                         f"Deviation: {deviation_from_20*100:.2f}% | "
                         f"RSI: {rsi:.1f} | Momentum: {momentum_5:.2f}%")
                
            else:
                # Check for weaker signals if no strong signal
                if buy_score >= 2 and can_buy:
                    decision = 0  # BUY
                    reason = f"WEAK BUY: Moderate signals (score={buy_score})"
                elif sell_score >= 2 and can_sell:
                    decision = 2  # SELL
                    reason = f"WEAK SELL: Moderate signals (score={sell_score})"
                else:
                    # Analyze why no action
                    if not can_buy and buy_score > 0:
                        reason = f"Buy signals (score={buy_score}) but insufficient cash ({cash_ratio:.2%})"
                    elif not can_sell and sell_score > 0:
                        reason = f"Sell signals (score={sell_score}) but no significant position"
                    else:
                        reason = (f"HOLD: No clear signals | "
                                 f"Buy score: {buy_score}, Sell score: {sell_score} | "
                                 f"Deviation: {deviation_from_20*100:.2f}% | RSI: {rsi:.1f}")
            
            # Store detailed debug info
            self.last_decision_info = {
                'decision': decision,
                'reason': reason,
                'current_price': float(current_price),
                'ma_10': float(ma_10),
                'ma_20': float(ma_20),
                'ma_50': float(ma_50),
                'trend_short': float(trend_short),
                'trend_medium': float(trend_medium),
                'deviation_from_20': float(deviation_from_20 * 100),  # in percentage
                'momentum_5': float(momentum_5),
                'momentum_10': float(momentum_10),
                'volatility': float(volatility),
                'rsi': float(rsi),
                'cash_ratio': float(cash_ratio),
                'position_ratio': float(position_ratio),
                'buy_score': buy_score,
                'sell_score': sell_score,
                'can_buy': can_buy,
                'can_sell': can_sell
            }
            
            # Log the decision
            action_names = {0: "BUY", 1: "HOLD", 2: "SELL"}
            logger.info(f"RuleBased → {action_names[decision]}: {reason}")
            
            return decision
            
        except Exception as e:
            error_msg = f"RuleBased error: {str(e)}"
            logger.error(error_msg)
            self.last_decision_info = {
                'decision': 1,
                'reason': error_msg
            }
            return 1  # HOLD on error
    
    def get_last_decision_info(self):
        """Get detailed info about the last decision."""
        return self.last_decision_info
    
    # Add predict method for compatibility with aggregator if needed
    def predict(self, state):
        """
        Alias for decide() for compatibility with EnsembleAggregator
        Returns: (action, confidence)
        """
        action = self.decide(state)
        # Calculate confidence based on scores
        info = self.get_last_decision_info()
        buy_score = info.get('buy_score', 0)
        sell_score = info.get('sell_score', 0)
        
        if action == 0:  # BUY
            confidence = min(0.9, buy_score / 10.0) if buy_score > 0 else 0.5
        elif action == 2:  # SELL
            confidence = min(0.9, sell_score / 10.0) if sell_score > 0 else 0.5
        else:  # HOLD
            confidence = 0.3
            
        return action, confidence