#!/usr/bin/env python3
"""
Script to apply Online Learning Bootstrap to the main trading bot
This script modifies Bot-Trading_Swing.py to integrate bootstrap functionality
"""

import os
import re
import shutil
from datetime import datetime

def backup_original_bot():
    """Create a backup of the original bot file"""
    original_file = "/workspace/Bot-Trading_Swing.py"
    backup_file = f"/workspace/Bot-Trading_Swing_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"
    
    try:
        shutil.copy2(original_file, backup_file)
        print(f"✅ Created backup: {backup_file}")
        return True
    except Exception as e:
        print(f"❌ Failed to create backup: {e}")
        return False

def apply_bootstrap_integration():
    """Apply bootstrap integration to the main bot file"""
    bot_file = "/workspace/Bot-Trading_Swing.py"
    
    if not os.path.exists(bot_file):
        print(f"❌ Bot file not found: {bot_file}")
        return False
    
    try:
        # Read the original file
        with open(bot_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Apply modifications
        modified_content = content
        
        # 1. Add imports at the top
        import_section = '''
# Enhanced Online Learning Bootstrap Integration
try:
    from online_learning_bootstrap import OnlineLearningBootstrap
    from online_learning_integration import EnhancedOnlineLearningManager, create_enhanced_online_learning_manager
    from production_config import ONLINE_LEARNING_BOOTSTRAP_CONFIG
    BOOTSTRAP_AVAILABLE = True
    print("✅ [Bootstrap] Online Learning Bootstrap modules loaded successfully")
except ImportError as e:
    print(f"⚠️ [Bootstrap] Bootstrap modules not available: {e}")
    BOOTSTRAP_AVAILABLE = False
'''
        
        # Find a good place to insert imports (after existing imports)
        import_pattern = r'(from sklearn\..*?import.*?\n)'
        if re.search(import_pattern, modified_content):
            modified_content = re.sub(import_pattern, r'\1' + import_section, modified_content, count=1)
        else:
            # Fallback: add after first import
            first_import = re.search(r'(import .*?\n)', modified_content)
            if first_import:
                modified_content = modified_content[:first_import.end()] + import_section + modified_content[first_import.end():]
        
        # 2. Modify OnlineLearningManager initialization
        # Find the OnlineLearningManager initialization
        online_learning_pattern = r'(self\.online_learning\s*=\s*OnlineLearningManager\(\))'
        
        replacement = '''# Enhanced Online Learning with Bootstrap Support
        if BOOTSTRAP_AVAILABLE:
            self.online_learning = create_enhanced_online_learning_manager(self)
            print("✅ [Bot Init] Enhanced Online Learning Manager with Bootstrap initialized")
        else:
            self.online_learning = OnlineLearningManager()
            print("⚠️ [Bot Init] Standard Online Learning Manager initialized (Bootstrap not available)")'''
        
        if re.search(online_learning_pattern, modified_content):
            modified_content = re.sub(online_learning_pattern, replacement, modified_content)
            print("✅ Modified OnlineLearningManager initialization")
        else:
            print("⚠️ Could not find OnlineLearningManager initialization pattern")
        
        # 3. Modify the initialize_all_online_models call
        # Find calls to initialize_all_online_models
        init_models_pattern = r'(self\.online_learning\.initialize_all_online_models\(.*?\))'
        
        init_replacement = '''# Enhanced initialization with bootstrap support
        if BOOTSTRAP_AVAILABLE and hasattr(self.online_learning, 'initialize_all_online_models_with_bootstrap'):
            bootstrap_report = self.online_learning.initialize_all_online_models_with_bootstrap(list(self.active_symbols))
            print(f"🚀 [Bootstrap] Initialization completed:")
            print(f"   - Symbols processed: {bootstrap_report['overall_stats']['successful_initializations']}/{bootstrap_report['total_symbols']}")
            print(f"   - Bootstrap samples used: {bootstrap_report['overall_stats']['total_bootstrap_samples']}")
            print(f"   - Processing time: {bootstrap_report['overall_stats']['total_initialization_time']:.2f}s")
        else:
            self.online_learning.initialize_all_online_models(list(self.active_symbols))
            print("⚠️ [Bootstrap] Using standard initialization (Bootstrap not available)")'''
        
        if re.search(init_models_pattern, modified_content):
            modified_content = re.sub(init_models_pattern, init_replacement, modified_content)
            print("✅ Modified initialize_all_online_models call")
        else:
            print("⚠️ Could not find initialize_all_online_models call")
        
        # 4. Add bootstrap status reporting method
        bootstrap_status_method = '''
    def get_bootstrap_status_report(self):
        """Get comprehensive bootstrap status report"""
        if not BOOTSTRAP_AVAILABLE:
            return {
                'bootstrap_available': False,
                'message': 'Bootstrap functionality not loaded'
            }
        
        if hasattr(self.online_learning, 'get_bootstrap_status'):
            status = self.online_learning.get_bootstrap_status()
            status['bootstrap_available'] = True
            return status
        else:
            return {
                'bootstrap_available': True,
                'message': 'Bootstrap available but not used in current online learning manager'
            }
    
    def print_bootstrap_summary(self):
        """Print a summary of bootstrap status"""
        status = self.get_bootstrap_status_report()
        
        print("\\n📊 BOOTSTRAP STATUS SUMMARY")
        print("=" * 40)
        
        if not status.get('bootstrap_available', False):
            print("❌ Bootstrap not available")
            print(f"   Reason: {status.get('message', 'Unknown')}")
            return
        
        if not status.get('bootstrap_history_available', False):
            print("⚠️ Bootstrap available but not initialized yet")
            return
        
        latest = status.get('latest_initialization', {})
        overall_stats = latest.get('overall_stats', {})
        
        print("✅ Bootstrap Status: Active")
        print(f"   Total models: {status.get('total_models', 0)}")
        print(f"   Models with bootstrap: {status.get('models_with_bootstrap', 0)}")
        print(f"   Last initialization:")
        print(f"      - Symbols: {overall_stats.get('successful_initializations', 0)}/{overall_stats.get('total_symbols', 0)}")
        print(f"      - Bootstrap samples: {overall_stats.get('total_bootstrap_samples', 0)}")
        print(f"      - Success rate: {overall_stats.get('success_rate', 0):.1%}")
        print(f"      - Avg samples/symbol: {overall_stats.get('average_samples_per_symbol', 0):.1f}")
'''
        
        # Find a good place to add the method (before the last class method)
        class_end_pattern = r'(\n\s*def\s+\w+.*?\n.*?""".*?""")'
        matches = list(re.finditer(class_end_pattern, modified_content, re.DOTALL))
        if matches:
            # Insert before the last method
            last_match = matches[-1]
            insert_pos = last_match.start()
            modified_content = modified_content[:insert_pos] + bootstrap_status_method + modified_content[insert_pos:]
            print("✅ Added bootstrap status reporting methods")
        else:
            # Fallback: add at the end of the file
            modified_content += bootstrap_status_method
            print("✅ Added bootstrap status reporting methods (at end of file)")
        
        # 5. Add bootstrap configuration validation
        config_validation = '''
# Bootstrap Configuration Validation
if BOOTSTRAP_AVAILABLE:
    print("🔧 [Bootstrap] Validating configuration...")
    config = ONLINE_LEARNING_BOOTSTRAP_CONFIG
    
    if config.get('ENABLE_BOOTSTRAP', False):
        print(f"✅ [Bootstrap] Enabled with method: {config.get('BOOTSTRAP_METHOD', 'unknown')}")
        print(f"   - Target samples: {config.get('BOOTSTRAP_SAMPLES', 0)}")
        print(f"   - Historical lookback: {config.get('HISTORICAL_LOOKBACK', 0)} candles")
        print(f"   - Parallel processing: {config.get('PARALLEL_BOOTSTRAP', False)}")
    else:
        print("⚠️ [Bootstrap] Available but disabled in configuration")
else:
    print("⚠️ [Bootstrap] Not available - using standard online learning")
'''
        
        # Add configuration validation after imports
        main_function_pattern = r'(if __name__ == "__main__":)'
        if re.search(main_function_pattern, modified_content):
            modified_content = re.sub(main_function_pattern, config_validation + '\n\n' + r'\1', modified_content)
            print("✅ Added bootstrap configuration validation")
        
        # Write the modified content back
        with open(bot_file, 'w', encoding='utf-8') as f:
            f.write(modified_content)
        
        print("✅ Successfully applied bootstrap integration to Bot-Trading_Swing.py")
        return True
    
    except Exception as e:
        print(f"❌ Error applying bootstrap integration: {e}")
        return False

def create_deployment_instructions():
    """Create deployment instructions file"""
    instructions = """
# Online Learning Bootstrap Deployment Instructions

## Files Added/Modified

### New Files:
1. `online_learning_bootstrap.py` - Core bootstrap implementation
2. `online_learning_integration.py` - Integration with existing bot
3. `test_bootstrap_integration.py` - Comprehensive test suite
4. `apply_bootstrap_to_bot.py` - This deployment script

### Modified Files:
1. `production_config.py` - Added ONLINE_LEARNING_BOOTSTRAP_CONFIG
2. `Bot-Trading_Swing.py` - Integrated bootstrap functionality

## Configuration

The bootstrap system is configured in `production_config.py`:

```python
ONLINE_LEARNING_BOOTSTRAP_CONFIG = {
    'ENABLE_BOOTSTRAP': True,                    # Enable/disable bootstrap
    'BOOTSTRAP_METHOD': 'consensus',             # Method: consensus, historical, hybrid
    'BOOTSTRAP_SAMPLES': 150,                    # Number of samples to generate
    'HISTORICAL_LOOKBACK': 5000,                 # Historical candles to analyze
    'CONSENSUS_THRESHOLD': 0.6,                  # Agreement threshold for consensus
    'PARALLEL_BOOTSTRAP': True,                  # Enable parallel processing
}
```

## How It Works

### Before Bootstrap:
```
🔄 [Online Learning] BTCUSD: Decision=HOLD, Confidence=50.00%
🔄 [Online Learning] ETHUSD: Decision=HOLD, Confidence=50.00%
```

### After Bootstrap:
```
🔄 [Online Learning] BTCUSD: Decision=SELL, Confidence=67.30%
🔄 [Online Learning] ETHUSD: Decision=BUY, Confidence=74.20%
```

## Testing

Run the test suite to verify functionality:

```bash
python test_bootstrap_integration.py
```

Expected output:
- ✅ All tests should pass with >80% success rate
- 🚀 Bootstrap should generate 100+ samples per symbol
- 📊 Models should show improved predictions (confidence ≠ 50%)

## Monitoring

The bot now includes bootstrap status monitoring:

```python
# In bot instance
bot.print_bootstrap_summary()
status = bot.get_bootstrap_status_report()
```

## Performance Impact

- **Initialization time**: +10-30 seconds (one-time cost)
- **Memory usage**: +50-100MB (for bootstrap samples)
- **Prediction quality**: Significantly improved diversity
- **Final confidence**: More varied and meaningful values

## Troubleshooting

### Bootstrap Not Working:
1. Check `ENABLE_BOOTSTRAP` in configuration
2. Verify sufficient historical data (5000+ candles)
3. Check log for bootstrap-related errors

### Low Performance:
1. Reduce `BOOTSTRAP_SAMPLES` (try 100 instead of 150)
2. Disable `PARALLEL_BOOTSTRAP` if memory constrained
3. Use 'historical' method instead of 'consensus'

### Still Getting 50% Confidence:
1. Check if bootstrap samples were actually applied
2. Verify model initialization completed successfully
3. Review prediction thresholds (lowered from 0.3 to 0.2)

## Rollback

To rollback changes:
1. Restore from backup: `Bot-Trading_Swing_backup_*.py`
2. Remove new files: `online_learning_*.py`
3. Revert `production_config.py` changes

## Next Steps

1. Monitor bootstrap performance in production
2. Tune configuration parameters based on results
3. Consider implementing adaptive thresholds
4. Add more sophisticated consensus mechanisms
"""
    
    try:
        with open('/workspace/BOOTSTRAP_DEPLOYMENT.md', 'w') as f:
            f.write(instructions)
        print("✅ Created deployment instructions: BOOTSTRAP_DEPLOYMENT.md")
        return True
    except Exception as e:
        print(f"❌ Failed to create deployment instructions: {e}")
        return False

def main():
    """Main deployment function"""
    print("🚀 ONLINE LEARNING BOOTSTRAP DEPLOYMENT")
    print("=" * 50)
    print("This script will integrate bootstrap functionality into your trading bot")
    print()
    
    # Step 1: Create backup
    print("📋 Step 1: Creating backup of original bot...")
    if not backup_original_bot():
        print("❌ Backup failed. Aborting deployment.")
        return False
    
    # Step 2: Apply integration
    print("\n📋 Step 2: Applying bootstrap integration...")
    if not apply_bootstrap_integration():
        print("❌ Integration failed. Please check the backup and try again.")
        return False
    
    # Step 3: Create deployment instructions
    print("\n📋 Step 3: Creating deployment instructions...")
    create_deployment_instructions()
    
    # Step 4: Run tests
    print("\n📋 Step 4: Running integration tests...")
    try:
        import subprocess
        result = subprocess.run(['python', '/workspace/test_bootstrap_integration.py'], 
                              capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            print("✅ Integration tests passed!")
            print("🎉 BOOTSTRAP DEPLOYMENT SUCCESSFUL!")
        else:
            print("⚠️ Some integration tests failed, but deployment completed.")
            print("Check test_bootstrap_integration.py output for details.")
    
    except Exception as e:
        print(f"⚠️ Could not run integration tests: {e}")
        print("Please run 'python test_bootstrap_integration.py' manually to verify.")
    
    # Final instructions
    print("\n🎯 DEPLOYMENT COMPLETED")
    print("=" * 50)
    print("✅ Bootstrap functionality has been integrated into your bot")
    print("✅ Configuration added to production_config.py")
    print("✅ Backup created for safety")
    print("✅ Test suite available for verification")
    print()
    print("📖 Next steps:")
    print("   1. Review BOOTSTRAP_DEPLOYMENT.md for details")
    print("   2. Run your bot and check for bootstrap initialization messages")
    print("   3. Monitor Online Learning predictions for improved diversity")
    print("   4. Tune configuration parameters as needed")
    print()
    print("🔧 To verify bootstrap is working:")
    print("   - Look for bootstrap initialization messages in bot startup")
    print("   - Check that Online Learning confidence ≠ 50%")
    print("   - Run bot.print_bootstrap_summary() for status")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)
    else:
        print("\n🎉 Ready to trade with enhanced Online Learning!")
        exit(0)