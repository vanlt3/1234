# Production Configuration for Trading Bot
# This file contains optimized settings for production environment

import os
from typing import Dict, Any

# ==============================================================================
# PRODUCTION API KEYS - REPLACE WITH ACTUAL KEYS
# ==============================================================================

PRODUCTION_API_KEYS = {
    # Trading APIs
    'FINHUB': os.getenv('FINNHUB_API_KEY', 'your_finnhub_api_key_here'),
    'MARKETAUX': os.getenv('MARKETAUX_API_KEY', 'your_marketaux_api_key_here'),
    'NEWSAPI': os.getenv('NEWSAPI_KEY', 'your_newsapi_key_here'),
    'EODHD': os.getenv('EODHD_API_KEY', 'your_eodhd_api_key_here'),
    
    # Economic Data
    'TRADING_ECONOMICS': os.getenv('TRADING_ECONOMICS_API_KEY', 'your_te_api_key_here'),
    'ALPHA_VANTAGE': os.getenv('ALPHA_VANTAGE_API_KEY', 'your_av_api_key_here'),
    
    # AI/ML Services
    'GOOGLE_AI': os.getenv('GOOGLE_AI_API_KEY', 'your_google_ai_key_here'),
    
    # Notifications
    'DISCORD_WEBHOOK': os.getenv('DISCORD_WEBHOOK_URL', 'your_discord_webhook_url_here'),
}

# ==============================================================================
# PRODUCTION CONFIDENCE SETTINGS
# ==============================================================================

PRODUCTION_CONFIDENCE_CONFIG = {
    # Enhanced thresholds for production
    'BUY_THRESHOLD': 0.12,      # Very low threshold for buy signals
    'SELL_THRESHOLD': 0.12,     # Very low threshold for sell signals  
    'HOLD_THRESHOLD': 0.30,     # Higher threshold for no action
    
    # Confidence calculation method
    'CALCULATION_METHOD': 'adaptive_dynamic',  # Options: weighted_average, bayesian_fusion, ensemble_voting, adaptive_dynamic
    
    # Market condition adjustments
    'VOLATILITY_BOOST_HIGH': 1.15,     # Boost trading signals in high volatility
    'VOLATILITY_REDUCE_LOW': 0.90,     # Reduce trading signals in low volatility
    'TREND_STRENGTH_BOOST': 1.10,      # Boost signals in strong trends
    
    # Performance-based adjustments
    'GOOD_PERFORMANCE_BOOST': 1.05,    # Boost for symbols with >70% success rate
    'POOR_PERFORMANCE_REDUCE': 0.85,   # Reduce for symbols with <30% success rate
    
    # Agreement bonuses
    'STRONG_AGREEMENT_BOOST': 1.10,    # Boost when >75% signals agree
    'POOR_AGREEMENT_REDUCE': 0.80,     # Reduce when <40% signals agree
}

# ==============================================================================
# ONLINE LEARNING BOOTSTRAP CONFIGURATION
# ==============================================================================

ONLINE_LEARNING_BOOTSTRAP_CONFIG = {
    # Bootstrap training settings
    'ENABLE_BOOTSTRAP': True,                    # Enable bootstrap initial training
    'BOOTSTRAP_METHOD': 'consensus',             # Options: 'consensus', 'historical', 'hybrid'
    'BOOTSTRAP_SAMPLES': 150,                    # Number of bootstrap samples to generate
    'MIN_BOOTSTRAP_SAMPLES': 50,                 # Minimum samples required for bootstrap
    
    # Consensus method settings
    'CONSENSUS_THRESHOLD': 0.6,                  # 60% agreement required for consensus
    'CONSENSUS_WEIGHTS': {                       # Weights for different models in consensus
        'rl': 0.35,
        'master': 0.30,
        'ensemble': 0.25,
        'online': 0.10
    },
    
    # Historical method settings
    'HISTORICAL_LOOKBACK': 500,                  # Number of historical candles to analyze
    'FUTURE_RETURN_PERIODS': 10,                 # Look ahead periods for labeling
    'BUY_THRESHOLD': 0.015,                      # 1.5% return threshold for BUY label
    'SELL_THRESHOLD': -0.015,                    # -1.5% return threshold for SELL label
    
    # Hybrid method settings
    'CONSENSUS_RATIO': 0.7,                      # 70% consensus, 30% historical
    'HISTORICAL_RATIO': 0.3,
    
    # Quality control
    'ENABLE_SAMPLE_VALIDATION': True,            # Validate bootstrap samples quality
    'MIN_CONFIDENCE_THRESHOLD': 0.4,             # Minimum confidence for valid samples
    'MAX_IMBALANCE_RATIO': 0.8,                  # Maximum class imbalance allowed
    
    # Performance settings
    'PARALLEL_BOOTSTRAP': True,                  # Enable parallel bootstrap processing
    'BOOTSTRAP_TIMEOUT': 30,                     # Timeout for bootstrap process (seconds)
    'ENABLE_BOOTSTRAP_LOGGING': True,            # Enable detailed bootstrap logging
}

# ==============================================================================
# PERFORMANCE OPTIMIZATION
# ==============================================================================

PERFORMANCE_CONFIG = {
    # Parallel processing
    'MAX_WORKERS': 4,                   # Maximum parallel workers
    'FEATURE_CACHE_TTL': 300,          # Feature cache TTL in seconds
    'DATA_CACHE_TTL': 180,             # Data cache TTL in seconds
    
    # Memory management
    'MAX_HISTORY_LENGTH': 1000,        # Maximum history entries per symbol
    'CLEANUP_INTERVAL': 3600,          # Cleanup interval in seconds
    
    # API rate limiting
    'API_RATE_LIMITS': {
        'FINHUB': 60,
        'MARKETAUX': 100, 
        'NEWSAPI': 1000,
        'EODHD': 20,
        'TRADING_ECONOMICS': 100
    },
    
    # Monitoring
    'HEALTH_CHECK_INTERVAL': 300,      # Health check interval in seconds
    'METRICS_RETENTION_HOURS': 168,    # Keep metrics for 1 week
}

# ==============================================================================
# RISK MANAGEMENT
# ==============================================================================

RISK_MANAGEMENT_CONFIG = {
    # Position sizing
    'MAX_RISK_PER_TRADE': 0.015,       # 1.5% max risk per trade (reduced)
    'MAX_PORTFOLIO_RISK': 0.08,        # 8% max total portfolio risk (reduced)
    'MAX_OPEN_POSITIONS': 3,           # Reduced from 5 to 3
    
    # Stop loss and take profit
    'SL_ATR_MULTIPLIER': 1.2,          # Tighter stop losses
    'BASE_RR_RATIO': 2.0,              # Better risk-reward ratio
    'TRAILING_STOP_MULTIPLIER': 0.8,   # Tighter trailing stops
    
    # Volatility adjustments
    'HIGH_VOLATILITY_THRESHOLD': 0.8,
    'LOW_VOLATILITY_THRESHOLD': 0.2,
    'VOLATILITY_POSITION_SCALING': True,
}

# ==============================================================================
# MONITORING AND ALERTING
# ==============================================================================

MONITORING_CONFIG = {
    # Discord notifications
    'ENABLE_DISCORD_ALERTS': True,
    'ALERT_LEVELS': ['ERROR', 'WARNING', 'TRADE'],
    'MAX_ALERTS_PER_HOUR': 20,
    
    # Performance monitoring
    'MONITOR_CONFIDENCE_TRENDS': True,
    'MONITOR_API_HEALTH': True,
    'MONITOR_TRADE_PERFORMANCE': True,
    
    # Health checks
    'MIN_SUCCESS_RATE_THRESHOLD': 0.4,     # Alert if success rate < 40%
    'MAX_ERROR_RATE_THRESHOLD': 0.1,       # Alert if error rate > 10%
    'MAX_RESPONSE_TIME_THRESHOLD': 15.0,   # Alert if response time > 15s
}

# ==============================================================================
# SYMBOL-SPECIFIC CONFIGURATIONS
# ==============================================================================

SYMBOL_CONFIGS = {
    # Crypto pairs - higher volatility tolerance
    'BTCUSD': {
        'confidence_multiplier': 1.0,
        'volatility_tolerance': 0.9,
        'max_position_size': 0.3,
    },
    'ETHUSD': {
        'confidence_multiplier': 1.0, 
        'volatility_tolerance': 0.85,
        'max_position_size': 0.25,
    },
    
    # Forex pairs - standard settings
    'EURUSD': {
        'confidence_multiplier': 1.1,  # Slightly higher confidence for major pairs
        'volatility_tolerance': 0.7,
        'max_position_size': 0.4,
    },
    'AUDUSD': {
        'confidence_multiplier': 1.05,
        'volatility_tolerance': 0.75,
        'max_position_size': 0.35,
    },
    'AUDNZD': {
        'confidence_multiplier': 0.95, # Slightly lower for minor pairs
        'volatility_tolerance': 0.8,
        'max_position_size': 0.3,
    },
    
    # Commodities - moderate settings
    'XAUUSD': {
        'confidence_multiplier': 1.0,
        'volatility_tolerance': 0.8,
        'max_position_size': 0.35,
    },
    'USOIL': {
        'confidence_multiplier': 0.9,  # Lower due to higher volatility
        'volatility_tolerance': 0.9,
        'max_position_size': 0.25,
    },
    
    # Indices - conservative settings
    'SPX500': {
        'confidence_multiplier': 1.15, # Higher confidence for stable indices
        'volatility_tolerance': 0.6,
        'max_position_size': 0.4,
    },
    'DE40': {
        'confidence_multiplier': 1.1,
        'volatility_tolerance': 0.65,
        'max_position_size': 0.35,
    },
}

# ==============================================================================
# PRODUCTION DEPLOYMENT SETTINGS
# ==============================================================================

DEPLOYMENT_CONFIG = {
    # Environment
    'ENVIRONMENT': 'PRODUCTION',
    'DEBUG_MODE': False,
    'VERBOSE_LOGGING': False,
    
    # Startup behavior
    'WARM_UP_CYCLES': 3,               # Number of cycles to warm up before trading
    'INITIAL_CONFIDENCE_BOOST': 1.05,  # Small boost during initial cycles
    
    # Failsafe mechanisms
    'ENABLE_CIRCUIT_BREAKER': True,    # Stop trading if too many errors
    'MAX_CONSECUTIVE_LOSSES': 5,       # Circuit breaker threshold
    'RECOVERY_TIME_HOURS': 2,          # Time to wait before resuming
    
    # Backup and recovery
    'AUTO_BACKUP_INTERVAL': 3600,      # Backup models every hour
    'BACKUP_RETENTION_DAYS': 7,        # Keep backups for 1 week
}

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def get_production_config() -> Dict[str, Any]:
    """Get complete production configuration"""
    return {
        'api_keys': PRODUCTION_API_KEYS,
        'confidence': PRODUCTION_CONFIDENCE_CONFIG,
        'performance': PERFORMANCE_CONFIG,
        'risk_management': RISK_MANAGEMENT_CONFIG,
        'monitoring': MONITORING_CONFIG,
        'symbols': SYMBOL_CONFIGS,
        'deployment': DEPLOYMENT_CONFIG,
    }

def validate_api_keys() -> Dict[str, bool]:
    """Validate that all required API keys are configured"""
    validation_results = {}
    
    required_keys = ['FINHUB', 'GOOGLE_AI', 'DISCORD_WEBHOOK']
    optional_keys = ['MARKETAUX', 'NEWSAPI', 'EODHD', 'TRADING_ECONOMICS', 'ALPHA_VANTAGE']
    
    for key in required_keys:
        api_key = PRODUCTION_API_KEYS.get(key, '')
        validation_results[key] = bool(api_key and 'your_' not in api_key.lower())
    
    for key in optional_keys:
        api_key = PRODUCTION_API_KEYS.get(key, '')
        validation_results[f"{key}_optional"] = bool(api_key and 'your_' not in api_key.lower())
    
    return validation_results

def get_symbol_config(symbol: str) -> Dict[str, Any]:
    """Get configuration for specific symbol"""
    return SYMBOL_CONFIGS.get(symbol, {
        'confidence_multiplier': 1.0,
        'volatility_tolerance': 0.75,
        'max_position_size': 0.3,
    })

if __name__ == "__main__":
    # Quick validation check
    print("🔍 Production Configuration Validation")
    print("=" * 50)
    
    config = get_production_config()
    validation = validate_api_keys()
    
    print(f"Environment: {config['deployment']['ENVIRONMENT']}")
    print(f"Confidence Method: {config['confidence']['CALCULATION_METHOD']}")
    print(f"Max Workers: {config['performance']['MAX_WORKERS']}")
    print(f"Max Risk per Trade: {config['risk_management']['MAX_RISK_PER_TRADE']*100}%")
    
    print("\n🔑 API Key Validation:")
    for key, valid in validation.items():
        status = "✅" if valid else "❌"
        print(f"  {status} {key}: {'Valid' if valid else 'Missing/Invalid'}")
    
    print(f"\n📊 Configured Symbols: {len(SYMBOL_CONFIGS)}")
    for symbol in SYMBOL_CONFIGS:
        print(f"  - {symbol}")