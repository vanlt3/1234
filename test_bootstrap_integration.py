#!/usr/bin/env python3
"""
Test script for Online Learning Bootstrap Integration
Tests the bootstrap functionality with mock data and real integration
"""

import sys
import os
import numpy as np
import pandas as pd
import time
from typing import Dict, List, Any
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from online_learning_bootstrap import OnlineLearningBootstrap, BootstrapSample
    from online_learning_integration import EnhancedOnlineLearningManager
    from production_config import ONLINE_LEARNING_BOOTSTRAP_CONFIG
    print("✅ Successfully imported bootstrap modules")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

class MockBot:
    """Mock bot instance for testing"""
    
    def __init__(self):
        self.active_symbols = ['BTCUSD', 'ETHUSD', 'XAUUSD']
        self.data_cache = {}
        self._generate_mock_data()
    
    def _generate_mock_data(self):
        """Generate mock historical data for testing"""
        print("🔧 Generating mock historical data...")
        
        for symbol in self.active_symbols:
            # Generate 5000 candles of mock data
            dates = pd.date_range(start='2020-01-01', periods=5000, freq='H')
            
            # Generate realistic price data with trends
            np.random.seed(42)  # For reproducible results
            base_price = {'BTCUSD': 50000, 'ETHUSD': 3000, 'XAUUSD': 2000}[symbol]
            
            # Generate price series with some trend and volatility
            returns = np.random.normal(0, 0.01, 5000)  # 1% volatility
            prices = [base_price]
            
            for ret in returns:
                new_price = prices[-1] * (1 + ret)
                prices.append(new_price)
            
            prices = prices[1:]  # Remove the initial price
            
            # Create OHLCV data
            data = pd.DataFrame({
                'datetime': dates,
                'open': prices,
                'high': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
                'low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
                'close': prices,
                'volume': np.random.randint(1000, 10000, 5000)
            })
            
            self.data_cache[symbol] = data
            print(f"   ✅ Generated {len(data)} candles for {symbol}")
    
    def get_symbol_data(self, symbol: str, count: int = 5000) -> pd.DataFrame:
        """Mock method to get symbol data"""
        if symbol in self.data_cache:
            data = self.data_cache[symbol]
            return data.tail(count).copy()
        else:
            print(f"⚠️ No mock data available for {symbol}")
            return pd.DataFrame()
    
    # Mock other bot methods that might be called
    def __getattr__(self, name):
        """Mock any missing attributes"""
        return None

class BootstrapTester:
    """Comprehensive tester for bootstrap functionality"""
    
    def __init__(self):
        self.mock_bot = MockBot()
        self.test_results = {}
        self.logger = logging.getLogger(__name__)
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all bootstrap tests"""
        print("\n🧪 STARTING COMPREHENSIVE BOOTSTRAP TESTS")
        print("=" * 60)
        
        start_time = time.time()
        
        # Test 1: Basic Bootstrap Functionality
        print("\n📋 Test 1: Basic Bootstrap Functionality")
        self.test_results['basic_bootstrap'] = self.test_basic_bootstrap()
        
        # Test 2: Bootstrap Methods Comparison
        print("\n📋 Test 2: Bootstrap Methods Comparison")
        self.test_results['methods_comparison'] = self.test_bootstrap_methods()
        
        # Test 3: Enhanced Online Learning Manager
        print("\n📋 Test 3: Enhanced Online Learning Manager")
        self.test_results['enhanced_manager'] = self.test_enhanced_manager()
        
        # Test 4: Performance Testing
        print("\n📋 Test 4: Performance Testing")
        self.test_results['performance'] = self.test_performance()
        
        # Test 5: Integration Testing
        print("\n📋 Test 5: Integration Testing")
        self.test_results['integration'] = self.test_integration()
        
        total_time = time.time() - start_time
        
        # Generate comprehensive report
        report = self.generate_test_report(total_time)
        
        print("\n🎯 BOOTSTRAP TESTING COMPLETED")
        print("=" * 60)
        print(f"Total testing time: {total_time:.2f}s")
        
        return report
    
    def test_basic_bootstrap(self) -> Dict[str, Any]:
        """Test basic bootstrap functionality"""
        print("   🔍 Testing basic bootstrap functionality...")
        
        bootstrap = OnlineLearningBootstrap(self.mock_bot)
        test_symbol = 'BTCUSD'
        
        try:
            # Test single symbol bootstrap
            samples, report = bootstrap.bootstrap_symbol(test_symbol)
            
            result = {
                'success': len(samples) > 0,
                'samples_generated': len(samples),
                'processing_time': report.get('processing_time', 0),
                'method_used': report.get('method', 'unknown'),
                'quality_score': report.get('quality_score', 0),
                'error': report.get('error', None)
            }
            
            if result['success']:
                print(f"   ✅ Generated {result['samples_generated']} samples in {result['processing_time']:.2f}s")
                
                # Test sample quality
                if samples:
                    sample = samples[0]
                    print(f"   📊 Sample quality check:")
                    print(f"      - Features shape: {sample.features.shape}")
                    print(f"      - Label: {sample.label}")
                    print(f"      - Confidence: {sample.confidence:.3f}")
                    print(f"      - Source: {sample.source}")
            else:
                print(f"   ❌ Bootstrap failed: {result.get('error', 'Unknown error')}")
            
            return result
        
        except Exception as e:
            print(f"   ❌ Exception in basic bootstrap test: {e}")
            return {'success': False, 'error': str(e)}
    
    def test_bootstrap_methods(self) -> Dict[str, Any]:
        """Test different bootstrap methods"""
        print("   🔍 Testing different bootstrap methods...")
        
        methods = ['consensus', 'historical', 'hybrid']
        method_results = {}
        
        for method in methods:
            print(f"      Testing {method} method...")
            
            # Create bootstrap instance with specific method
            config = ONLINE_LEARNING_BOOTSTRAP_CONFIG.copy()
            config['BOOTSTRAP_METHOD'] = method
            config['BOOTSTRAP_SAMPLES'] = 50  # Smaller for testing
            
            bootstrap = OnlineLearningBootstrap(self.mock_bot)
            bootstrap.config = config
            
            try:
                samples, report = bootstrap.bootstrap_symbol('ETHUSD')
                
                method_results[method] = {
                    'success': len(samples) > 0,
                    'samples_generated': len(samples),
                    'processing_time': report.get('processing_time', 0),
                    'quality_score': report.get('quality_score', 0),
                    'error': report.get('error', None)
                }
                
                if method_results[method]['success']:
                    print(f"         ✅ {method}: {len(samples)} samples in {report.get('processing_time', 0):.2f}s")
                else:
                    print(f"         ❌ {method}: Failed - {report.get('error', 'Unknown')}")
            
            except Exception as e:
                print(f"         ❌ {method}: Exception - {e}")
                method_results[method] = {'success': False, 'error': str(e)}
        
        # Compare methods
        successful_methods = [m for m, r in method_results.items() if r['success']]
        
        return {
            'methods_tested': methods,
            'successful_methods': successful_methods,
            'method_results': method_results,
            'best_method': max(successful_methods, key=lambda m: method_results[m]['samples_generated']) if successful_methods else None
        }
    
    def test_enhanced_manager(self) -> Dict[str, Any]:
        """Test enhanced online learning manager"""
        print("   🔍 Testing enhanced online learning manager...")
        
        try:
            # Create enhanced manager
            manager = EnhancedOnlineLearningManager(self.mock_bot)
            
            # Test initialization with bootstrap
            test_symbols = ['BTCUSD', 'ETHUSD']
            initialization_report = manager.initialize_all_online_models_with_bootstrap(test_symbols)
            
            result = {
                'success': initialization_report['overall_stats']['successful_initializations'] > 0,
                'symbols_processed': initialization_report['total_symbols'],
                'successful_initializations': initialization_report['overall_stats']['successful_initializations'],
                'total_bootstrap_samples': initialization_report['overall_stats']['total_bootstrap_samples'],
                'initialization_time': initialization_report['overall_stats']['total_initialization_time']
            }
            
            if result['success']:
                print(f"   ✅ Initialized {result['successful_initializations']}/{result['symbols_processed']} symbols")
                print(f"      - Total bootstrap samples: {result['total_bootstrap_samples']}")
                print(f"      - Initialization time: {result['initialization_time']:.2f}s")
                
                # Test predictions from bootstrap-trained models
                prediction_results = {}
                for symbol in test_symbols:
                    if symbol in manager.models:
                        try:
                            # Create mock market data
                            mock_data = {'close': 50000, 'volume': 1000, 'high': 50100, 'low': 49900, 'open': 50050}
                            decision, confidence = manager.get_online_prediction_enhanced(symbol, mock_data)
                            
                            prediction_results[symbol] = {
                                'decision': decision,
                                'confidence': confidence,
                                'success': confidence != 0.5  # Success if not default
                            }
                            
                            print(f"      - {symbol}: {decision} ({confidence:.2%})")
                        
                        except Exception as e:
                            prediction_results[symbol] = {'success': False, 'error': str(e)}
                
                result['prediction_results'] = prediction_results
                result['predictions_working'] = sum(1 for r in prediction_results.values() if r.get('success', False))
            else:
                print("   ❌ Enhanced manager initialization failed")
            
            return result
        
        except Exception as e:
            print(f"   ❌ Exception in enhanced manager test: {e}")
            return {'success': False, 'error': str(e)}
    
    def test_performance(self) -> Dict[str, Any]:
        """Test bootstrap performance with larger datasets"""
        print("   🔍 Testing bootstrap performance...")
        
        try:
            # Test with all symbols
            bootstrap = OnlineLearningBootstrap(self.mock_bot)
            
            start_time = time.time()
            results = bootstrap.bootstrap_multiple_symbols(self.mock_bot.active_symbols)
            processing_time = time.time() - start_time
            
            # Generate performance report
            overall_report = bootstrap.generate_bootstrap_report(results)
            
            performance_result = {
                'success': overall_report['successful_symbols'] > 0,
                'symbols_processed': overall_report['total_symbols'],
                'successful_symbols': overall_report['successful_symbols'],
                'total_samples': overall_report['total_samples'],
                'processing_time': processing_time,
                'samples_per_second': overall_report['total_samples'] / processing_time if processing_time > 0 else 0,
                'average_quality_score': overall_report['average_quality_score']
            }
            
            if performance_result['success']:
                print(f"   ✅ Performance test completed:")
                print(f"      - Symbols: {performance_result['successful_symbols']}/{performance_result['symbols_processed']}")
                print(f"      - Total samples: {performance_result['total_samples']}")
                print(f"      - Processing time: {performance_result['processing_time']:.2f}s")
                print(f"      - Samples/second: {performance_result['samples_per_second']:.1f}")
                print(f"      - Avg quality score: {performance_result['average_quality_score']:.3f}")
            else:
                print("   ❌ Performance test failed")
            
            return performance_result
        
        except Exception as e:
            print(f"   ❌ Exception in performance test: {e}")
            return {'success': False, 'error': str(e)}
    
    def test_integration(self) -> Dict[str, Any]:
        """Test full integration scenario"""
        print("   🔍 Testing full integration scenario...")
        
        try:
            # Create enhanced manager and run full workflow
            manager = EnhancedOnlineLearningManager(self.mock_bot)
            
            # Step 1: Initialize with bootstrap
            print("      Step 1: Initializing with bootstrap...")
            init_report = manager.initialize_all_online_models_with_bootstrap(self.mock_bot.active_symbols)
            
            # Step 2: Test predictions for all symbols
            print("      Step 2: Testing predictions...")
            prediction_tests = {}
            
            for symbol in self.mock_bot.active_symbols:
                mock_data = self.mock_bot.get_symbol_data(symbol, count=1).iloc[-1]
                decision, confidence = manager.get_online_prediction_enhanced(symbol, mock_data)
                
                prediction_tests[symbol] = {
                    'decision': decision,
                    'confidence': confidence,
                    'improved': confidence != 0.5  # Check if better than default
                }
            
            # Step 3: Get bootstrap status
            print("      Step 3: Checking bootstrap status...")
            status = manager.get_bootstrap_status()
            
            # Calculate results
            successful_predictions = sum(1 for test in prediction_tests.values() if test['improved'])
            
            integration_result = {
                'success': init_report['overall_stats']['successful_initializations'] > 0 and successful_predictions > 0,
                'initialization_success': init_report['overall_stats']['successful_initializations'],
                'prediction_improvements': successful_predictions,
                'total_symbols': len(self.mock_bot.active_symbols),
                'bootstrap_status': status,
                'prediction_details': prediction_tests
            }
            
            if integration_result['success']:
                print(f"   ✅ Integration test passed:")
                print(f"      - Initialized: {integration_result['initialization_success']} symbols")
                print(f"      - Improved predictions: {integration_result['prediction_improvements']} symbols")
                
                # Show prediction improvements
                for symbol, test in prediction_tests.items():
                    status_icon = "✅" if test['improved'] else "⚠️"
                    print(f"      - {symbol}: {status_icon} {test['decision']} ({test['confidence']:.2%})")
            else:
                print("   ❌ Integration test failed")
            
            return integration_result
        
        except Exception as e:
            print(f"   ❌ Exception in integration test: {e}")
            return {'success': False, 'error': str(e)}
    
    def generate_test_report(self, total_time: float) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        
        # Count successful tests
        successful_tests = sum(1 for result in self.test_results.values() if result.get('success', False))
        total_tests = len(self.test_results)
        
        # Calculate overall statistics
        total_samples = 0
        total_processing_time = 0
        
        for test_name, result in self.test_results.items():
            if 'samples_generated' in result:
                total_samples += result['samples_generated']
            if 'processing_time' in result:
                total_processing_time += result['processing_time']
            elif 'total_bootstrap_samples' in result:
                total_samples += result['total_bootstrap_samples']
            elif 'initialization_time' in result:
                total_processing_time += result['initialization_time']
        
        report = {
            'test_summary': {
                'total_tests': total_tests,
                'successful_tests': successful_tests,
                'success_rate': successful_tests / total_tests if total_tests > 0 else 0,
                'total_execution_time': total_time
            },
            'performance_metrics': {
                'total_samples_generated': total_samples,
                'total_processing_time': total_processing_time,
                'average_samples_per_second': total_samples / total_processing_time if total_processing_time > 0 else 0
            },
            'detailed_results': self.test_results,
            'recommendations': self._generate_recommendations()
        }
        
        # Print summary
        print(f"\n📊 TEST SUMMARY:")
        print(f"   Tests passed: {successful_tests}/{total_tests} ({report['test_summary']['success_rate']:.1%})")
        print(f"   Total samples generated: {total_samples}")
        print(f"   Average processing speed: {report['performance_metrics']['average_samples_per_second']:.1f} samples/sec")
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        # Check basic bootstrap
        if not self.test_results.get('basic_bootstrap', {}).get('success', False):
            recommendations.append("❌ Basic bootstrap failed - check data availability and feature extraction")
        
        # Check methods comparison
        methods_result = self.test_results.get('methods_comparison', {})
        if methods_result.get('successful_methods'):
            best_method = methods_result.get('best_method')
            if best_method:
                recommendations.append(f"✅ Use '{best_method}' method for best results")
        else:
            recommendations.append("⚠️ No bootstrap methods worked - check configuration and data quality")
        
        # Check enhanced manager
        manager_result = self.test_results.get('enhanced_manager', {})
        if manager_result.get('success', False):
            predictions_working = manager_result.get('predictions_working', 0)
            if predictions_working > 0:
                recommendations.append(f"✅ Enhanced manager working - {predictions_working} symbols have improved predictions")
            else:
                recommendations.append("⚠️ Enhanced manager initialized but predictions not improved")
        
        # Check performance
        perf_result = self.test_results.get('performance', {})
        if perf_result.get('success', False):
            samples_per_sec = perf_result.get('samples_per_second', 0)
            if samples_per_sec > 10:
                recommendations.append("✅ Performance is good - ready for production")
            else:
                recommendations.append("⚠️ Performance may be slow - consider optimizing or reducing sample count")
        
        # Check integration
        integration_result = self.test_results.get('integration', {})
        if integration_result.get('success', False):
            recommendations.append("✅ Full integration working - ready to deploy")
        else:
            recommendations.append("❌ Integration issues detected - review configuration and dependencies")
        
        return recommendations

def main():
    """Main test execution"""
    print("🚀 ONLINE LEARNING BOOTSTRAP TEST SUITE")
    print("=" * 60)
    print("This script tests the bootstrap functionality for Online Learning models")
    print()
    
    # Run tests
    tester = BootstrapTester()
    report = tester.run_all_tests()
    
    # Save report
    import json
    report_file = 'bootstrap_test_report.json'
    
    try:
        with open(report_file, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            def convert_numpy_types(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(v) for v in obj]
                else:
                    return obj
            
            json.dump(convert_numpy_types(report), f, indent=2, default=str)
        
        print(f"\n💾 Test report saved to: {report_file}")
    
    except Exception as e:
        print(f"\n⚠️ Could not save test report: {e}")
    
    # Print final recommendations
    if report['recommendations']:
        print(f"\n🎯 RECOMMENDATIONS:")
        for rec in report['recommendations']:
            print(f"   {rec}")
    
    # Return exit code based on success
    success_rate = report['test_summary']['success_rate']
    if success_rate >= 0.8:
        print(f"\n🎉 BOOTSTRAP TESTING SUCCESSFUL ({success_rate:.1%} pass rate)")
        return 0
    else:
        print(f"\n⚠️ BOOTSTRAP TESTING NEEDS ATTENTION ({success_rate:.1%} pass rate)")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)