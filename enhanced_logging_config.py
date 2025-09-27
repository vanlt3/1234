#!/usr/bin/env python3
"""
Enhanced Logging Configuration for Trading Bot
Enables detailed logging for predictions, signals, and trading decisions
"""

import os
import logging
from datetime import datetime
from pathlib import Path

# ==============================================================================
# ENHANCED LOGGING CONFIGURATION
# ==============================================================================

ENHANCED_LOGGING_CONFIG = {
    # Global logging settings
    'ENABLE_DEBUG_MODE': True,          # Enable debug mode for detailed logs
    'ENABLE_VERBOSE_LOGGING': True,     # Enable verbose logging
    'CONSOLE_LOG_LEVEL': 'DEBUG',       # Show all logs in console
    'FILE_LOG_LEVEL': 'DEBUG',          # Save all logs to file
    
    # Trading pipeline specific logging
    'ENABLE_PREDICTION_LOGS': True,     # Log ML model predictions
    'ENABLE_SIGNAL_LOGS': True,         # Log trading signals generation
    'ENABLE_CONFIDENCE_LOGS': True,     # Log confidence calculations
    'ENABLE_DECISION_LOGS': True,       # Log Master Agent decisions
    'ENABLE_ENTRY_LOGS': True,          # Log trade entry decisions
    'ENABLE_FEATURE_LOGS': True,        # Log feature engineering details
    'ENABLE_RL_LOGS': True,             # Log RL Agent actions
    'ENABLE_PORTFOLIO_LOGS': True,      # Log portfolio management
    
    # Performance logging
    'ENABLE_TIMING_LOGS': True,         # Log execution times
    'ENABLE_API_LOGS': True,            # Log API calls and responses
    'ENABLE_ERROR_DETAIL_LOGS': True,   # Detailed error logging
    
    # Log formatting
    'USE_COLORED_CONSOLE': True,        # Use colors in console
    'USE_EMOJIS': True,                 # Use emojis for better visibility
    'LOG_TIMESTAMP_FORMAT': '%Y-%m-%d %H:%M:%S.%f',  # Detailed timestamps
    
    # File logging
    'ROTATE_LOG_FILES': True,           # Rotate log files
    'MAX_LOG_FILE_SIZE': '50MB',        # Max size before rotation
    'KEEP_LOG_FILES': 7,                # Keep 7 days of logs
}

# ==============================================================================
# SPECIFIC LOGGER CONFIGURATIONS
# ==============================================================================

LOGGER_CONFIGS = {
    # Core trading loggers
    'TradingBot': {
        'level': 'DEBUG',
        'enable_console': True,
        'enable_file': True,
        'prefix': '🤖 [Bot]',
    },
    
    # ML/AI Prediction loggers
    'MLPredictor': {
        'level': 'DEBUG',
        'enable_console': True,
        'enable_file': True,
        'prefix': '🧠 [ML]',
    },
    
    'RLAgent': {
        'level': 'DEBUG',
        'enable_console': True,
        'enable_file': True,
        'prefix': '🎯 [RL]',
    },
    
    'EnsembleModel': {
        'level': 'DEBUG',
        'enable_console': True,
        'enable_file': True,
        'prefix': '🎪 [Ensemble]',
    },
    
    # Signal and Decision loggers
    'SignalGenerator': {
        'level': 'DEBUG',
        'enable_console': True,
        'enable_file': True,
        'prefix': '📡 [Signal]',
    },
    
    'ConfidenceManager': {
        'level': 'DEBUG',
        'enable_console': True,
        'enable_file': True,
        'prefix': '🎚️ [Confidence]',
    },
    
    'MasterAgent': {
        'level': 'DEBUG',
        'enable_console': True,
        'enable_file': True,
        'prefix': '👑 [Master]',
    },
    
    'TradeDecision': {
        'level': 'DEBUG',
        'enable_console': True,
        'enable_file': True,
        'prefix': '⚖️ [Decision]',
    },
    
    # Feature and Data loggers
    'FeatureEngineer': {
        'level': 'INFO',  # Less verbose for features
        'enable_console': True,
        'enable_file': True,
        'prefix': '⚙️ [Features]',
    },
    
    'DataManager': {
        'level': 'INFO',
        'enable_console': True,
        'enable_file': True,
        'prefix': '📊 [Data]',
    },
    
    # Portfolio and Risk loggers
    'PortfolioManager': {
        'level': 'DEBUG',
        'enable_console': True,
        'enable_file': True,
        'prefix': '💼 [Portfolio]',
    },
    
    'RiskManager': {
        'level': 'DEBUG',
        'enable_console': True,
        'enable_file': True,
        'prefix': '🛡️ [Risk]',
    },
}

# ==============================================================================
# ENHANCED FORMATTER CLASS
# ==============================================================================

class TradingBotFormatter(logging.Formatter):
    """Enhanced formatter specifically for trading bot logs"""
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'       # Reset
    }
    
    LEVEL_EMOJIS = {
        'DEBUG': '🔍',
        'INFO': 'ℹ️',
        'WARNING': '⚠️',
        'ERROR': '❌',
        'CRITICAL': '🚨'
    }
    
    # Special emojis for trading-specific logs
    TRADING_EMOJIS = {
        'prediction': '🔮',
        'signal': '📡',
        'confidence': '🎚️',
        'entry': '🚪',
        'exit': '🚀',
        'profit': '💰',
        'loss': '📉',
        'buy': '🟢',
        'sell': '🔴',
        'hold': '⏸️',
        'success': '✅',
        'failure': '❌',
        'warning': '⚠️',
    }
    
    def format(self, record):
        # Add timestamp
        record.asctime = datetime.fromtimestamp(record.created).strftime(
            ENHANCED_LOGGING_CONFIG['LOG_TIMESTAMP_FORMAT'][:-3]  # Remove microseconds for readability
        )
        
        # Add level emoji and color
        if ENHANCED_LOGGING_CONFIG['USE_EMOJIS']:
            level_emoji = self.LEVEL_EMOJIS.get(record.levelname, '📝')
        else:
            level_emoji = ''
            
        if ENHANCED_LOGGING_CONFIG['USE_COLORED_CONSOLE']:
            color = self.COLORS.get(record.levelname, '')
            reset = self.COLORS['RESET']
            record.levelname = f"{color}{level_emoji} {record.levelname}{reset}"
        else:
            record.levelname = f"{level_emoji} {record.levelname}"
        
        # Add trading-specific emojis to message
        if ENHANCED_LOGGING_CONFIG['USE_EMOJIS']:
            message = str(record.msg).lower()
            for keyword, emoji in self.TRADING_EMOJIS.items():
                if keyword in message:
                    record.msg = f"{emoji} {record.msg}"
                    break
        
        return super().format(record)

# ==============================================================================
# SETUP FUNCTIONS
# ==============================================================================

def setup_enhanced_logging():
    """Setup enhanced logging configuration for trading bot"""
    
    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Clear existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    
    # Set root logger level
    root_logger.setLevel(logging.DEBUG)
    
    # Setup console handler
    if ENHANCED_LOGGING_CONFIG['ENABLE_DEBUG_MODE']:
        console_handler = logging.StreamHandler()
        console_formatter = TradingBotFormatter(
            '%(asctime)s | %(levelname)-15s | %(name)-20s | %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(getattr(logging, ENHANCED_LOGGING_CONFIG['CONSOLE_LOG_LEVEL']))
        root_logger.addHandler(console_handler)
    
    # Setup file handler
    log_filename = f"trading_bot_detailed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_dir / log_filename, encoding='utf-8')
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)-20s | %(funcName)-20s:%(lineno)-4d | %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(getattr(logging, ENHANCED_LOGGING_CONFIG['FILE_LOG_LEVEL']))
    root_logger.addHandler(file_handler)
    
    # Setup specific loggers
    configured_loggers = {}
    for logger_name, config in LOGGER_CONFIGS.items():
        logger = logging.getLogger(logger_name)
        logger.setLevel(getattr(logging, config['level']))
        logger.propagate = True
        configured_loggers[logger_name] = logger
    
    # Suppress noisy external libraries
    external_libs = ['urllib3', 'requests', 'tensorflow', 'torch', 'sklearn', 'matplotlib', 'numpy', 'pandas']
    for lib in external_libs:
        logging.getLogger(lib).setLevel(logging.WARNING)
    
    print("🚀 [Enhanced Logging] Detailed trading bot logging configured!")
    print(f"📁 [Enhanced Logging] Logs will be saved to: {log_dir / log_filename}")
    
    return configured_loggers

def get_trading_logger(name: str) -> logging.Logger:
    """Get a configured logger for trading components"""
    config = LOGGER_CONFIGS.get(name, {
        'level': 'INFO',
        'prefix': f'🔧 [{name}]'
    })
    
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, config['level']))
    
    return logger

# ==============================================================================
# LOGGING HELPER FUNCTIONS
# ==============================================================================

def log_prediction(logger: logging.Logger, symbol: str, prediction: dict):
    """Log ML model prediction with enhanced formatting"""
    if ENHANCED_LOGGING_CONFIG['ENABLE_PREDICTION_LOGS']:
        direction = prediction.get('direction', 'UNKNOWN')
        confidence = prediction.get('confidence', 0.0)
        model_name = prediction.get('model', 'Unknown')
        
        logger.debug(f"🔮 [Prediction] {symbol} | Model: {model_name} | Direction: {direction} | Confidence: {confidence:.3f}")

def log_signal(logger: logging.Logger, symbol: str, signal: dict):
    """Log trading signal generation with enhanced formatting"""
    if ENHANCED_LOGGING_CONFIG['ENABLE_SIGNAL_LOGS']:
        signal_type = signal.get('type', 'UNKNOWN')
        strength = signal.get('strength', 0.0)
        source = signal.get('source', 'Unknown')
        
        emoji = '🟢' if signal_type == 'BUY' else '🔴' if signal_type == 'SELL' else '⏸️'
        logger.debug(f"{emoji} [Signal] {symbol} | Type: {signal_type} | Strength: {strength:.3f} | Source: {source}")

def log_confidence(logger: logging.Logger, symbol: str, confidence_data: dict):
    """Log confidence calculation with enhanced formatting"""
    if ENHANCED_LOGGING_CONFIG['ENABLE_CONFIDENCE_LOGS']:
        final_confidence = confidence_data.get('final', 0.0)
        method = confidence_data.get('method', 'Unknown')
        components = confidence_data.get('components', {})
        
        logger.debug(f"🎚️ [Confidence] {symbol} | Final: {final_confidence:.3f} | Method: {method} | Components: {components}")

def log_trade_decision(logger: logging.Logger, symbol: str, decision: dict):
    """Log Master Agent trade decision with enhanced formatting"""
    if ENHANCED_LOGGING_CONFIG['ENABLE_DECISION_LOGS']:
        action = decision.get('action', 'UNKNOWN')
        reasoning = decision.get('reasoning', 'No reason provided')
        risk_score = decision.get('risk_score', 0.0)
        
        emoji = '✅' if action in ['BUY', 'SELL'] else '❌' if action == 'REJECT' else '⏸️'
        logger.debug(f"{emoji} [Decision] {symbol} | Action: {action} | Risk: {risk_score:.3f} | Reason: {reasoning}")

def log_entry_execution(logger: logging.Logger, symbol: str, entry_data: dict):
    """Log trade entry execution with enhanced formatting"""
    if ENHANCED_LOGGING_CONFIG['ENABLE_ENTRY_LOGS']:
        entry_price = entry_data.get('price', 0.0)
        quantity = entry_data.get('quantity', 0.0)
        sl_price = entry_data.get('sl', 0.0)
        tp_price = entry_data.get('tp', 0.0)
        
        logger.info(f"🚪 [Entry] {symbol} | Price: {entry_price:.5f} | Qty: {quantity} | SL: {sl_price:.5f} | TP: {tp_price:.5f}")

# ==============================================================================
# CONFIGURATION OVERRIDE FOR PRODUCTION
# ==============================================================================

def apply_enhanced_logging_to_bot():
    """Apply enhanced logging configuration to existing bot"""
    
    # Override production config settings
    import production_config
    
    # Enable debug and verbose logging
    production_config.DEPLOYMENT_CONFIG['DEBUG_MODE'] = True
    production_config.DEPLOYMENT_CONFIG['VERBOSE_LOGGING'] = True
    
    # Setup enhanced logging
    setup_enhanced_logging()
    
    print("🔧 [Config Override] Enhanced logging applied to production bot!")
    print("📊 [Config Override] Debug mode and verbose logging enabled!")

if __name__ == "__main__":
    # Test the enhanced logging setup
    print("🧪 [Test] Testing Enhanced Logging Configuration...")
    
    # Setup logging
    loggers = setup_enhanced_logging()
    
    # Test different logger types
    ml_logger = get_trading_logger('MLPredictor')
    signal_logger = get_trading_logger('SignalGenerator')
    master_logger = get_trading_logger('MasterAgent')
    
    # Test log functions
    log_prediction(ml_logger, 'BTCUSD', {
        'direction': 'BUY',
        'confidence': 0.85,
        'model': 'RandomForest'
    })
    
    log_signal(signal_logger, 'BTCUSD', {
        'type': 'BUY',
        'strength': 0.78,
        'source': 'Technical Analysis'
    })
    
    log_trade_decision(master_logger, 'BTCUSD', {
        'action': 'BUY',
        'reasoning': 'Strong bullish signals with high confidence',
        'risk_score': 0.65
    })
    
    print("✅ [Test] Enhanced logging test completed!")