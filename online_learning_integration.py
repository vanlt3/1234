"""
Integration module for Online Learning Bootstrap
Updates the existing OnlineLearningManager to support bootstrap initialization
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Any, Optional
import time

# Import the bootstrap implementation
from online_learning_bootstrap import OnlineLearningBootstrap, BootstrapSample
from production_config import ONLINE_LEARNING_BOOTSTRAP_CONFIG

class EnhancedOnlineLearningManager:
    """
    Enhanced Online Learning Manager with Bootstrap Support
    This extends the existing OnlineLearningManager with bootstrap capabilities
    """
    
    def __init__(self, bot_instance=None):
        # Initialize existing attributes
        self.bot = bot_instance
        self.models = {}  # Store online learning models per symbol
        self.performance_history = {}
        self.drift_detector = None
        
        # Bootstrap-related attributes
        self.bootstrap = OnlineLearningBootstrap(bot_instance)
        self.bootstrap_config = ONLINE_LEARNING_BOOTSTRAP_CONFIG
        self.bootstrap_history = {}  # Track bootstrap performance
        self.logger = logging.getLogger(__name__)
        
        # Try to import River for advanced online learning
        try:
            from river import linear_model, forest, tree
            self.river_available = True
            self.river_models = {
                'logistic': linear_model.LogisticRegression(),
                'ensemble': forest.ARFRegressor(),
                'tree': tree.HoeffdingTreeRegressor()
            }
            print("River models initialized for online learning")
        except ImportError:
            self.river_available = False
            print("⚠️ River not available. Using sklearn SGDClassifier.")
        
        # Fallback: sklearn SGDClassifier
        from sklearn.linear_model import SGDClassifier
        self.sgd_model = SGDClassifier(loss='log', learning_rate='adaptive', eta0=0.01)
    
    def initialize_all_online_models_with_bootstrap(self, active_symbols: List[str]) -> Dict[str, Any]:
        """
        Initialize online learning models for ALL active symbols with bootstrap training
        Returns comprehensive bootstrap report
        """
        print("[Enhanced Online Learning] Initializing models with bootstrap training...")
        start_time = time.time()
        
        # Check if bootstrap is enabled
        if not self.bootstrap_config.get('ENABLE_BOOTSTRAP', True):
            print("⚠️ [Bootstrap] Bootstrap training disabled, using standard initialization")
            return self._initialize_standard_models(active_symbols)
        
        # Generate bootstrap data for all symbols
        print(f"🚀 [Bootstrap] Generating bootstrap data for {len(active_symbols)} symbols...")
        bootstrap_results = self.bootstrap.bootstrap_multiple_symbols(active_symbols)
        
        # Initialize models with bootstrap data
        initialization_report = {
            'total_symbols': len(active_symbols),
            'bootstrap_enabled': True,
            'start_time': start_time,
            'symbol_reports': {},
            'overall_stats': {}
        }
        
        successful_initializations = 0
        total_bootstrap_samples = 0
        
        for symbol in active_symbols:
            symbol_report = self._initialize_symbol_with_bootstrap(symbol, bootstrap_results.get(symbol, ([], {})))
            initialization_report['symbol_reports'][symbol] = symbol_report
            
            if symbol_report.get('success', False):
                successful_initializations += 1
                total_bootstrap_samples += symbol_report.get('bootstrap_samples_used', 0)
        
        # Calculate overall statistics
        total_time = time.time() - start_time
        initialization_report['overall_stats'] = {
            'successful_initializations': successful_initializations,
            'success_rate': successful_initializations / len(active_symbols),
            'total_bootstrap_samples': total_bootstrap_samples,
            'average_samples_per_symbol': total_bootstrap_samples / successful_initializations if successful_initializations > 0 else 0,
            'total_initialization_time': total_time,
            'average_time_per_symbol': total_time / len(active_symbols)
        }
        
        # Store bootstrap history for monitoring
        self.bootstrap_history[pd.Timestamp.now()] = initialization_report
        
        # Print summary
        print(f"✅ [Enhanced Online Learning] Initialization completed:")
        print(f"   - Symbols processed: {len(active_symbols)}")
        print(f"   - Successful initializations: {successful_initializations}")
        print(f"   - Total bootstrap samples: {total_bootstrap_samples}")
        print(f"   - Processing time: {total_time:.2f}s")
        
        return initialization_report
    
    def _initialize_symbol_with_bootstrap(self, symbol: str, bootstrap_data: Tuple[List[BootstrapSample], Dict[str, Any]]) -> Dict[str, Any]:
        """Initialize a single symbol with bootstrap data"""
        bootstrap_samples, bootstrap_report = bootstrap_data
        
        symbol_report = {
            'symbol': symbol,
            'bootstrap_report': bootstrap_report,
            'success': False,
            'bootstrap_samples_available': len(bootstrap_samples),
            'bootstrap_samples_used': 0,
            'model_type': 'unknown'
        }
        
        try:
            # Initialize the model first
            model_type = 'logistic'  # Default model type
            
            if self.river_available:
                from river import linear_model
                self.models[symbol] = linear_model.LogisticRegression()
                symbol_report['model_type'] = 'river_logistic'
                print(f"[Enhanced Online Learning] ✅ Initialized River logistic model for {symbol}")
            else:
                # Fallback to sklearn
                from sklearn.linear_model import SGDClassifier
                self.models[symbol] = SGDClassifier(loss='log', learning_rate='adaptive', eta0=0.01)
                symbol_report['model_type'] = 'sklearn_sgd'
                print(f"[Enhanced Online Learning] ✅ Initialized SGD model for {symbol}")
            
            # Apply bootstrap training if samples are available
            if bootstrap_samples:
                samples_used = self._apply_bootstrap_training(symbol, bootstrap_samples)
                symbol_report['bootstrap_samples_used'] = samples_used
                
                if samples_used > 0:
                    print(f"[Bootstrap Training] ✅ {symbol}: Trained with {samples_used} bootstrap samples")
                    
                    # Test the trained model
                    test_result = self._test_bootstrap_model(symbol, bootstrap_samples[:5])  # Test with first 5 samples
                    symbol_report['test_result'] = test_result
                    
                    if test_result.get('success', False):
                        symbol_report['success'] = True
                        print(f"[Bootstrap Test] ✅ {symbol}: Model test passed (avg confidence: {test_result.get('avg_confidence', 0):.2%})")
                    else:
                        print(f"[Bootstrap Test] ⚠️ {symbol}: Model test failed")
                else:
                    print(f"[Bootstrap Training] ⚠️ {symbol}: No bootstrap samples could be applied")
            else:
                print(f"[Bootstrap Training] ⚠️ {symbol}: No bootstrap samples available, using default initialization")
                symbol_report['success'] = True  # Standard initialization is still successful
        
        except Exception as e:
            self.logger.error(f"Error initializing {symbol} with bootstrap: {e}")
            symbol_report['error'] = str(e)
        
        return symbol_report
    
    def _apply_bootstrap_training(self, symbol: str, bootstrap_samples: List[BootstrapSample]) -> int:
        """Apply bootstrap training samples to a model"""
        if symbol not in self.models:
            return 0
        
        model = self.models[symbol]
        samples_applied = 0
        
        try:
            for sample in bootstrap_samples:
                try:
                    if self.river_available and hasattr(model, 'learn_one'):
                        # River incremental learning
                        features_dict = self._convert_features_to_dict(sample.features)
                        model.learn_one(features_dict, sample.label)
                        samples_applied += 1
                    
                    elif hasattr(model, 'partial_fit'):
                        # Sklearn partial fit
                        X = sample.features.reshape(1, -1)
                        y = [sample.label]
                        
                        # For the first sample, we need to specify all possible classes
                        if samples_applied == 0:
                            model.partial_fit(X, y, classes=[0.0, 0.5, 1.0])
                        else:
                            model.partial_fit(X, y)
                        
                        samples_applied += 1
                
                except Exception as e:
                    self.logger.debug(f"Error applying bootstrap sample for {symbol}: {e}")
                    continue
        
        except Exception as e:
            self.logger.error(f"Error in bootstrap training for {symbol}: {e}")
        
        return samples_applied
    
    def _convert_features_to_dict(self, features: np.ndarray) -> Dict[str, float]:
        """Convert numpy array features to dictionary format for River"""
        return {f'feature_{i}': float(val) for i, val in enumerate(features)}
    
    def _test_bootstrap_model(self, symbol: str, test_samples: List[BootstrapSample]) -> Dict[str, Any]:
        """Test the bootstrap-trained model"""
        if symbol not in self.models or not test_samples:
            return {'success': False, 'reason': 'No model or test samples'}
        
        model = self.models[symbol]
        predictions = []
        confidences = []
        
        try:
            for sample in test_samples:
                try:
                    if self.river_available and hasattr(model, 'predict_one'):
                        # River prediction
                        features_dict = self._convert_features_to_dict(sample.features)
                        prediction = model.predict_one(features_dict)
                        
                        # Convert prediction to decision and confidence
                        if prediction > 0.3:
                            decision = "BUY"
                            confidence = min(prediction, 0.9)
                        elif prediction < -0.3:
                            decision = "SELL"
                            confidence = min(abs(prediction), 0.9)
                        else:
                            decision = "HOLD"
                            confidence = 0.5
                        
                        predictions.append(decision)
                        confidences.append(confidence)
                    
                    elif hasattr(model, 'predict_proba'):
                        # Sklearn probability prediction
                        X = sample.features.reshape(1, -1)
                        proba = model.predict_proba(X)[0]
                        
                        # Find the class with highest probability
                        class_idx = np.argmax(proba)
                        classes = model.classes_
                        predicted_label = classes[class_idx]
                        confidence = proba[class_idx]
                        
                        # Convert label to decision
                        if predicted_label >= 0.75:
                            decision = "BUY"
                        elif predicted_label <= 0.25:
                            decision = "SELL"
                        else:
                            decision = "HOLD"
                        
                        predictions.append(decision)
                        confidences.append(confidence)
                
                except Exception as e:
                    self.logger.debug(f"Error testing sample: {e}")
                    predictions.append("HOLD")
                    confidences.append(0.5)
            
            # Calculate test statistics
            if confidences:
                avg_confidence = np.mean(confidences)
                decision_diversity = len(set(predictions)) / len(predictions)
                non_hold_ratio = sum(1 for p in predictions if p != "HOLD") / len(predictions)
                
                # Consider test successful if:
                # 1. Average confidence > 0.4 (better than random)
                # 2. Some decision diversity (not all HOLD)
                # 3. At least some non-HOLD decisions
                success = (avg_confidence > 0.4 and 
                          decision_diversity > 0.1 and 
                          non_hold_ratio > 0.1)
                
                return {
                    'success': success,
                    'avg_confidence': avg_confidence,
                    'decision_diversity': decision_diversity,
                    'non_hold_ratio': non_hold_ratio,
                    'predictions': predictions,
                    'confidences': confidences
                }
            else:
                return {'success': False, 'reason': 'No valid predictions'}
        
        except Exception as e:
            return {'success': False, 'reason': str(e)}
    
    def _initialize_standard_models(self, active_symbols: List[str]) -> Dict[str, Any]:
        """Fallback to standard initialization without bootstrap"""
        print("[Online Learning] Initializing models with standard method...")
        
        for symbol in active_symbols:
            try:
                if self.river_available:
                    from river import linear_model
                    self.models[symbol] = linear_model.LogisticRegression()
                    print(f"[Online Learning] ✅ Initialized River logistic model for {symbol}")
                else:
                    from sklearn.linear_model import SGDClassifier
                    self.models[symbol] = SGDClassifier(loss='log', learning_rate='adaptive', eta0=0.01)
                    print(f"[Online Learning] ✅ Initialized SGD model for {symbol}")
            except Exception as e:
                print(f"[Online Learning] ❌ Error initializing {symbol}: {e}")
        
        return {
            'bootstrap_enabled': False,
            'total_symbols': len(active_symbols),
            'method': 'standard',
            'models_initialized': len(self.models)
        }
    
    def get_online_prediction_enhanced(self, symbol: str, market_data: Any) -> Tuple[str, float]:
        """
        Enhanced prediction method that works with bootstrap-trained models
        """
        try:
            if symbol not in self.models:
                return "HOLD", 0.5
            
            # Extract features from market data
            features = self._extract_features_from_market_data(market_data)
            if features is None:
                return "HOLD", 0.5
            
            model = self.models[symbol]
            
            # Get prediction based on model type
            if self.river_available and hasattr(model, 'predict_one'):
                # River prediction
                features_dict = self._convert_features_to_dict(features)
                prediction = model.predict_one(features_dict)
                
                # Convert prediction to decision with updated thresholds
                # Lower thresholds to get more diverse predictions after bootstrap training
                if prediction > 0.2:  # Lowered from 0.3
                    decision = "BUY"
                    confidence = min(prediction * 1.2, 0.9)  # Boost confidence slightly
                elif prediction < -0.2:  # Lowered from -0.3
                    decision = "SELL"
                    confidence = min(abs(prediction) * 1.2, 0.9)  # Boost confidence slightly
                else:
                    decision = "HOLD"
                    confidence = 0.5
            
            elif hasattr(model, 'predict_proba'):
                # Sklearn prediction
                X = features.reshape(1, -1)
                
                try:
                    proba = model.predict_proba(X)[0]
                    class_idx = np.argmax(proba)
                    classes = model.classes_
                    predicted_label = classes[class_idx]
                    confidence = proba[class_idx]
                    
                    # Convert label to decision
                    if predicted_label >= 0.6:  # Lowered from 0.75
                        decision = "BUY"
                    elif predicted_label <= 0.4:  # Raised from 0.25
                        decision = "SELL"
                    else:
                        decision = "HOLD"
                
                except Exception as e:
                    # Fallback if predict_proba fails
                    decision = "HOLD"
                    confidence = 0.5
            
            else:
                # No prediction capability
                decision = "HOLD"
                confidence = 0.5
            
            return decision, confidence
        
        except Exception as e:
            self.logger.error(f"Error getting enhanced prediction for {symbol}: {e}")
            return "HOLD", 0.5
    
    def _extract_features_from_market_data(self, market_data: Any) -> Optional[np.ndarray]:
        """Extract features from market data (simplified version)"""
        try:
            # This is a simplified feature extraction
            # In practice, you'd extract comprehensive technical features
            if hasattr(market_data, 'close'):
                # Basic price features
                features = [
                    float(market_data.close),
                    float(getattr(market_data, 'volume', 0)),
                    float(getattr(market_data, 'high', market_data.close)),
                    float(getattr(market_data, 'low', market_data.close)),
                    float(getattr(market_data, 'open', market_data.close))
                ]
                return np.array(features, dtype=np.float64)
            elif isinstance(market_data, dict):
                # Extract from dictionary
                features = [
                    float(market_data.get('close', 0)),
                    float(market_data.get('volume', 0)),
                    float(market_data.get('high', 0)),
                    float(market_data.get('low', 0)),
                    float(market_data.get('open', 0))
                ]
                return np.array(features, dtype=np.float64)
            else:
                # Fallback: create dummy features
                return np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)
        
        except Exception as e:
            self.logger.debug(f"Error extracting features: {e}")
            return np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    
    def get_bootstrap_status(self) -> Dict[str, Any]:
        """Get current bootstrap status and statistics"""
        if not self.bootstrap_history:
            return {'bootstrap_history_available': False}
        
        latest_report = list(self.bootstrap_history.values())[-1]
        
        status = {
            'bootstrap_history_available': True,
            'latest_initialization': latest_report,
            'total_models': len(self.models),
            'bootstrap_config': self.bootstrap_config,
            'models_with_bootstrap': 0
        }
        
        # Count models that were successfully bootstrap-trained
        for symbol_report in latest_report.get('symbol_reports', {}).values():
            if symbol_report.get('bootstrap_samples_used', 0) > 0:
                status['models_with_bootstrap'] += 1
        
        return status

# Integration function to replace existing OnlineLearningManager
def create_enhanced_online_learning_manager(bot_instance=None) -> EnhancedOnlineLearningManager:
    """
    Factory function to create enhanced online learning manager
    This can be used to replace existing OnlineLearningManager instances
    """
    return EnhancedOnlineLearningManager(bot_instance)

# Example usage in existing bot code:
"""
# Replace this line in Bot-Trading_Swing.py:
# self.online_learning = OnlineLearningManager()

# With this:
from online_learning_integration import create_enhanced_online_learning_manager
self.online_learning = create_enhanced_online_learning_manager(self)

# Then use the enhanced initialization:
initialization_report = self.online_learning.initialize_all_online_models_with_bootstrap(list(self.active_symbols))
print(f"Bootstrap initialization report: {initialization_report}")
"""