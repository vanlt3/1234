#!/usr/bin/env python3
"""
Script to apply enhanced logging configuration to the trading bot
This will patch the existing bot to show detailed predictions, signals, and decisions
"""

import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

def patch_bot_logging():
    """Patch the main bot file to include enhanced logging"""
    
    bot_file = Path("Bot-Trading_Swing.py")
    if not bot_file.exists():
        print("❌ Bot file not found!")
        return False
    
    # Read the current bot file
    with open(bot_file, 'r', encoding='utf-8') as f:
        bot_content = f.read()
    
    # Create backup
    backup_file = f"Bot-Trading_Swing_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"
    with open(backup_file, 'w', encoding='utf-8') as f:
        f.write(bot_content)
    print(f"✅ Backup created: {backup_file}")
    
    # Patches to apply
    patches = []
    
    # 1. Import enhanced logging at the top
    import_patch = '''
# Enhanced Logging Configuration
from enhanced_logging_config import (
    setup_enhanced_logging, get_trading_logger,
    log_prediction, log_signal, log_confidence, 
    log_trade_decision, log_entry_execution,
    ENHANCED_LOGGING_CONFIG
)
'''
    
    # 2. Enable debug mode in production config
    config_patch = '''
# Override production config for enhanced logging
DEPLOYMENT_CONFIG = {
    'ENVIRONMENT': 'PRODUCTION',
    'DEBUG_MODE': True,          # ✅ Enable debug mode
    'VERBOSE_LOGGING': True,     # ✅ Enable verbose logging
    'WARM_UP_CYCLES': 3,
    'INITIAL_CONFIDENCE_BOOST': 1.05,
    'ENABLE_CIRCUIT_BREAKER': True,
    'MAX_CONSECUTIVE_LOSSES': 5,
    'RECOVERY_TIME_HOURS': 2,
    'AUTO_BACKUP_INTERVAL': 3600,
    'BACKUP_RETENTION_DAYS': 7,
}
'''
    
    # 3. Patch console handler log level
    console_patch = '''
        console_handler.setLevel(logging.DEBUG)  # ✅ Show DEBUG logs in console
'''
    
    # Apply patches
    modified_content = bot_content
    
    # Find import section and add enhanced logging import
    import_insertion_point = "import logging"
    if import_insertion_point in modified_content:
        modified_content = modified_content.replace(
            import_insertion_point,
            import_insertion_point + import_patch
        )
        patches.append("✅ Enhanced logging import added")
    
    # Find DEPLOYMENT_CONFIG and replace it
    if "'DEBUG_MODE': False" in modified_content:
        modified_content = modified_content.replace(
            "'DEBUG_MODE': False",
            "'DEBUG_MODE': True"
        )
        patches.append("✅ DEBUG_MODE enabled")
    
    if "'VERBOSE_LOGGING': False" in modified_content:
        modified_content = modified_content.replace(
            "'VERBOSE_LOGGING': False",
            "'VERBOSE_LOGGING': True"
        )
        patches.append("✅ VERBOSE_LOGGING enabled")
    
    # Find console handler and change log level
    if "console_handler.setLevel(logging.INFO)" in modified_content:
        modified_content = modified_content.replace(
            "console_handler.setLevel(logging.INFO)",
            "console_handler.setLevel(logging.DEBUG)"
        )
        patches.append("✅ Console log level changed to DEBUG")
    
    # Write the modified file
    with open(bot_file, 'w', encoding='utf-8') as f:
        f.write(modified_content)
    
    print("🔧 Applied patches:")
    for patch in patches:
        print(f"   {patch}")
    
    return len(patches) > 0

def add_prediction_logging_patches():
    """Add specific logging calls for predictions and signals"""
    
    bot_file = Path("Bot-Trading_Swing.py")
    with open(bot_file, 'r', encoding='utf-8') as f:
        bot_content = f.read()
    
    # Find key functions and add logging
    prediction_patches = [
        # Add logging after model predictions
        {
            'search': 'prediction = model.predict(X_latest)',
            'replace': '''prediction = model.predict(X_latest)
            # Enhanced logging for predictions
            pred_logger = get_trading_logger('MLPredictor')
            log_prediction(pred_logger, symbol, {
                'direction': 'BUY' if prediction[0] > 0.5 else 'SELL',
                'confidence': float(prediction[0]),
                'model': model.__class__.__name__
            })'''
        },
        
        # Add logging for ensemble predictions
        {
            'search': 'ensemble_prediction = ensemble_model.predict',
            'replace': '''ensemble_prediction = ensemble_model.predict
            # Enhanced logging for ensemble predictions
            ens_logger = get_trading_logger('EnsembleModel')
            ens_logger.debug(f"🎪 [Ensemble] {symbol} prediction: {ensemble_prediction}")'''
        },
        
        # Add logging for confidence calculations
        {
            'search': 'final_confidence =',
            'replace': '''final_confidence =
            # Enhanced logging for confidence
            conf_logger = get_trading_logger('ConfidenceManager')
            log_confidence(conf_logger, symbol, {
                'final': final_confidence,
                'method': 'production_confidence',
                'components': {'base': final_confidence}
            })'''
        },
        
        # Add logging for RL agent decisions
        {
            'search': 'rl_action = rl_agent.predict',
            'replace': '''rl_action = rl_agent.predict
            # Enhanced logging for RL decisions
            rl_logger = get_trading_logger('RLAgent')
            rl_logger.debug(f"🎯 [RL] {symbol} action: {rl_action}")'''
        },
    ]
    
    modified_content = bot_content
    applied_patches = []
    
    for patch in prediction_patches:
        if patch['search'] in modified_content:
            modified_content = modified_content.replace(
                patch['search'],
                patch['replace']
            )
            applied_patches.append(f"✅ Added logging for: {patch['search'][:30]}...")
    
    # Write the modified file
    with open(bot_file, 'w', encoding='utf-8') as f:
        f.write(modified_content)
    
    print("🔮 Applied prediction logging patches:")
    for patch in applied_patches:
        print(f"   {patch}")
    
    return len(applied_patches)

def create_logging_injection_script():
    """Create a script that can be injected into the running bot"""
    
    injection_script = '''
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
'''
    
    with open('logging_injection.py', 'w', encoding='utf-8') as f:
        f.write(injection_script)
    
    print("💉 Created logging injection script: logging_injection.py")

if __name__ == "__main__":
    from datetime import datetime
    
    print("🔧 [Enhanced Logging Patcher] Starting bot logging enhancement...")
    
    # Apply basic logging patches
    if patch_bot_logging():
        print("✅ Basic logging patches applied successfully!")
    else:
        print("⚠️ Some basic patches may have failed")
    
    # Add prediction-specific logging
    prediction_patches = add_prediction_logging_patches()
    if prediction_patches > 0:
        print(f"✅ Applied {prediction_patches} prediction logging patches!")
    else:
        print("⚠️ No prediction logging patches could be applied")
    
    # Create injection script as backup
    create_logging_injection_script()
    
    print("\n🎯 [Summary] Enhanced logging setup complete!")
    print("📊 [Next Steps]:")
    print("   1. Restart your bot to see detailed logs")
    print("   2. Look for logs with emojis: 🔮 🎯 📡 🎚️ 👑 ⚖️")
    print("   3. Check the detailed log file in logs/ directory")
    print("   4. If patches don't work, use the injection script manually")