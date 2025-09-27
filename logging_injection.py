
# LOGGING INJECTION SCRIPT - Paste this into your bot code
# Add this at the beginning of your main analysis loop

import logging
from datetime import datetime

# Setup enhanced console logging
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '%(asctime)s | 🔍 %(levelname)-8s | %(name)-20s | %(message)s'
)
console.setFormatter(formatter)

# Get root logger and add console handler
root_logger = logging.getLogger()
if not any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers):
    root_logger.addHandler(console)
    root_logger.setLevel(logging.DEBUG)

# Create specific loggers for trading components
ml_logger = logging.getLogger('MLPredictor')
signal_logger = logging.getLogger('SignalGenerator') 
confidence_logger = logging.getLogger('ConfidenceManager')
master_logger = logging.getLogger('MasterAgent')
decision_logger = logging.getLogger('TradeDecision')

# Enable debug logging for all trading loggers
for logger in [ml_logger, signal_logger, confidence_logger, master_logger, decision_logger]:
    logger.setLevel(logging.DEBUG)

print("🚀 [Enhanced Logging] Injection script applied - predictions will now be visible!")

# Add these logging calls in your prediction functions:
# ml_logger.debug(f"🔮 [Prediction] {symbol} | Direction: {direction} | Confidence: {confidence:.3f}")
# signal_logger.debug(f"📡 [Signal] {symbol} | Type: {signal_type} | Strength: {strength:.3f}")
# confidence_logger.debug(f"🎚️ [Confidence] {symbol} | Final: {confidence:.3f}")
# master_logger.debug(f"👑 [Master] {symbol} | Decision: {decision} | Reason: {reason}")
# decision_logger.debug(f"⚖️ [Decision] {symbol} | Action: {action} | Risk: {risk:.3f}")
