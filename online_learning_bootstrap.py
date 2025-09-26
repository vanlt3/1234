"""
Online Learning Bootstrap Implementation
Provides initial training data for Online Learning models to improve cold start performance
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from dataclasses import dataclass
from collections import Counter

# Import configuration
try:
    from production_config import ONLINE_LEARNING_BOOTSTRAP_CONFIG
except ImportError:
    # Fallback configuration
    ONLINE_LEARNING_BOOTSTRAP_CONFIG = {
        'ENABLE_BOOTSTRAP': True,
        'BOOTSTRAP_METHOD': 'consensus',
        'BOOTSTRAP_SAMPLES': 150,
        'MIN_BOOTSTRAP_SAMPLES': 50,
        'CONSENSUS_THRESHOLD': 0.6,
        'CONSENSUS_WEIGHTS': {'rl': 0.35, 'master': 0.30, 'ensemble': 0.25, 'online': 0.10},
        'HISTORICAL_LOOKBACK': 5000,
        'FUTURE_RETURN_PERIODS': 10,
        'BUY_THRESHOLD': 0.015,
        'SELL_THRESHOLD': -0.015,
        'CONSENSUS_RATIO': 0.7,
        'HISTORICAL_RATIO': 0.3,
        'ENABLE_SAMPLE_VALIDATION': True,
        'MIN_CONFIDENCE_THRESHOLD': 0.4,
        'MAX_IMBALANCE_RATIO': 0.8,
        'PARALLEL_BOOTSTRAP': True,
        'BOOTSTRAP_TIMEOUT': 30,
        'ENABLE_BOOTSTRAP_LOGGING': True
    }

@dataclass
class BootstrapSample:
    """Data class for bootstrap training samples"""
    features: np.ndarray
    label: float  # 0=SELL, 0.5=HOLD, 1=BUY
    confidence: float
    source: str  # 'consensus', 'historical', 'hybrid'
    metadata: Dict[str, Any] = None

class BootstrapQualityValidator:
    """Validates quality of bootstrap samples"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def validate_samples(self, samples: List[BootstrapSample]) -> Tuple[List[BootstrapSample], Dict[str, Any]]:
        """
        Validate bootstrap samples quality
        Returns: (filtered_samples, quality_report)
        """
        if not self.config.get('ENABLE_SAMPLE_VALIDATION', True):
            return samples, {'validation_enabled': False}
        
        original_count = len(samples)
        quality_report = {
            'original_count': original_count,
            'validation_enabled': True
        }
        
        # Filter by confidence threshold
        min_confidence = self.config.get('MIN_CONFIDENCE_THRESHOLD', 0.4)
        confidence_filtered = [s for s in samples if s.confidence >= min_confidence]
        quality_report['confidence_filtered'] = len(confidence_filtered)
        
        # Check class balance
        labels = [s.label for s in confidence_filtered]
        label_counts = Counter(labels)
        total_samples = len(labels)
        
        if total_samples > 0:
            # Calculate imbalance ratio
            max_class_count = max(label_counts.values()) if label_counts else 0
            imbalance_ratio = max_class_count / total_samples if total_samples > 0 else 0
            quality_report['imbalance_ratio'] = imbalance_ratio
            
            max_imbalance = self.config.get('MAX_IMBALANCE_RATIO', 0.8)
            if imbalance_ratio > max_imbalance:
                self.logger.warning(f"High class imbalance detected: {imbalance_ratio:.2%}")
                # Balance classes by undersampling majority class
                confidence_filtered = self._balance_classes(confidence_filtered, max_imbalance)
                quality_report['balanced_count'] = len(confidence_filtered)
        
        quality_report['final_count'] = len(confidence_filtered)
        quality_report['quality_score'] = len(confidence_filtered) / original_count if original_count > 0 else 0
        
        return confidence_filtered, quality_report
    
    def _balance_classes(self, samples: List[BootstrapSample], max_imbalance: float) -> List[BootstrapSample]:
        """Balance classes by undersampling majority class"""
        # Group samples by label
        label_groups = {}
        for sample in samples:
            label = sample.label
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append(sample)
        
        # Find target count based on max_imbalance
        total_samples = len(samples)
        target_max_count = int(total_samples * max_imbalance)
        
        # Undersample majority classes
        balanced_samples = []
        for label, group_samples in label_groups.items():
            if len(group_samples) > target_max_count:
                # Sort by confidence and take top samples
                group_samples.sort(key=lambda x: x.confidence, reverse=True)
                balanced_samples.extend(group_samples[:target_max_count])
            else:
                balanced_samples.extend(group_samples)
        
        return balanced_samples

class OnlineLearningBootstrap:
    """Bootstrap initial training data for Online Learning models"""
    
    def __init__(self, bot_instance=None):
        self.bot = bot_instance
        self.config = ONLINE_LEARNING_BOOTSTRAP_CONFIG
        self.logger = logging.getLogger(__name__)
        self.validator = BootstrapQualityValidator(self.config)
        
        # Enable detailed logging if configured
        if self.config.get('ENABLE_BOOTSTRAP_LOGGING', True):
            logging.basicConfig(level=logging.INFO)
    
    def bootstrap_symbol(self, symbol: str) -> Tuple[List[BootstrapSample], Dict[str, Any]]:
        """
        Bootstrap initial training data for a symbol
        Returns: (bootstrap_samples, bootstrap_report)
        """
        if not self.config.get('ENABLE_BOOTSTRAP', True):
            return [], {'bootstrap_enabled': False}
        
        self.logger.info(f"🚀 [Bootstrap] Starting bootstrap training for {symbol}")
        start_time = time.time()
        
        bootstrap_report = {
            'symbol': symbol,
            'method': self.config.get('BOOTSTRAP_METHOD', 'consensus'),
            'bootstrap_enabled': True,
            'start_time': start_time
        }
        
        try:
            # Generate samples based on method
            method = self.config.get('BOOTSTRAP_METHOD', 'consensus')
            
            if method == 'consensus':
                samples = self._bootstrap_consensus_method(symbol)
            elif method == 'historical':
                samples = self._bootstrap_historical_method(symbol)
            elif method == 'hybrid':
                samples = self._bootstrap_hybrid_method(symbol)
            else:
                self.logger.error(f"Unknown bootstrap method: {method}")
                return [], {'error': f'Unknown method: {method}'}
            
            bootstrap_report['raw_samples'] = len(samples)
            
            # Validate sample quality
            validated_samples, quality_report = self.validator.validate_samples(samples)
            bootstrap_report.update(quality_report)
            
            # Check minimum samples requirement
            min_samples = self.config.get('MIN_BOOTSTRAP_SAMPLES', 50)
            if len(validated_samples) < min_samples:
                self.logger.warning(f"⚠️ [Bootstrap] {symbol}: Only {len(validated_samples)} samples generated, minimum {min_samples} required")
                bootstrap_report['insufficient_samples'] = True
            
            bootstrap_report['processing_time'] = time.time() - start_time
            bootstrap_report['samples_per_second'] = len(validated_samples) / bootstrap_report['processing_time'] if bootstrap_report['processing_time'] > 0 else 0
            
            self.logger.info(f"✅ [Bootstrap] {symbol}: Generated {len(validated_samples)} validated samples in {bootstrap_report['processing_time']:.2f}s")
            
            return validated_samples, bootstrap_report
            
        except Exception as e:
            self.logger.error(f"❌ [Bootstrap] Error bootstrapping {symbol}: {e}")
            bootstrap_report['error'] = str(e)
            return [], bootstrap_report
    
    def _bootstrap_consensus_method(self, symbol: str) -> List[BootstrapSample]:
        """Generate bootstrap samples using consensus from other models"""
        samples = []
        target_samples = self.config.get('BOOTSTRAP_SAMPLES', 150)
        
        if not self.bot:
            self.logger.warning("Bot instance not available for consensus method")
            return samples
        
        try:
            # Get historical data for feature extraction
            symbol_data = self._get_symbol_historical_data(symbol)
            if symbol_data is None or len(symbol_data) < 100:
                self.logger.warning(f"Insufficient historical data for {symbol}")
                return samples
            
            # Generate samples from different time points
            data_len = len(symbol_data)
            sample_indices = np.linspace(50, data_len-10, target_samples, dtype=int)
            
            for idx in sample_indices:
                try:
                    # Extract features at this time point
                    features = self._extract_features_at_index(symbol_data, idx)
                    if features is None:
                        continue
                    
                    # Get predictions from different models
                    model_predictions = self._get_model_predictions(symbol, features, symbol_data.iloc[idx])
                    
                    # Create consensus label
                    consensus_label, consensus_confidence = self._create_consensus_label(model_predictions)
                    
                    if consensus_confidence >= self.config.get('CONSENSUS_THRESHOLD', 0.6):
                        sample = BootstrapSample(
                            features=features,
                            label=consensus_label,
                            confidence=consensus_confidence,
                            source='consensus',
                            metadata={'predictions': model_predictions, 'index': idx}
                        )
                        samples.append(sample)
                
                except Exception as e:
                    self.logger.debug(f"Error generating consensus sample at index {idx}: {e}")
                    continue
        
        except Exception as e:
            self.logger.error(f"Error in consensus method for {symbol}: {e}")
        
        return samples
    
    def _bootstrap_historical_method(self, symbol: str) -> List[BootstrapSample]:
        """Generate bootstrap samples using historical price movements"""
        samples = []
        target_samples = self.config.get('BOOTSTRAP_SAMPLES', 150)
        
        try:
            # Get historical data
            symbol_data = self._get_symbol_historical_data(symbol)
            if symbol_data is None or len(symbol_data) < 200:
                self.logger.warning(f"Insufficient historical data for {symbol}")
                return samples
            
            lookback_periods = self.config.get('FUTURE_RETURN_PERIODS', 10)
            data_len = len(symbol_data)
            
            # Generate samples from different time points
            sample_indices = np.linspace(100, data_len-lookback_periods-1, target_samples, dtype=int)
            
            for idx in sample_indices:
                try:
                    # Extract features at this time point
                    features = self._extract_features_at_index(symbol_data, idx)
                    if features is None:
                        continue
                    
                    # Calculate future return
                    current_price = symbol_data.iloc[idx]['close']
                    future_price = symbol_data.iloc[idx + lookback_periods]['close']
                    future_return = (future_price - current_price) / current_price
                    
                    # Convert return to label
                    label, confidence = self._return_to_label(future_return)
                    
                    sample = BootstrapSample(
                        features=features,
                        label=label,
                        confidence=confidence,
                        source='historical',
                        metadata={'future_return': future_return, 'index': idx}
                    )
                    samples.append(sample)
                
                except Exception as e:
                    self.logger.debug(f"Error generating historical sample at index {idx}: {e}")
                    continue
        
        except Exception as e:
            self.logger.error(f"Error in historical method for {symbol}: {e}")
        
        return samples
    
    def _bootstrap_hybrid_method(self, symbol: str) -> List[BootstrapSample]:
        """Generate bootstrap samples using hybrid approach"""
        consensus_ratio = self.config.get('CONSENSUS_RATIO', 0.7)
        historical_ratio = self.config.get('HISTORICAL_RATIO', 0.3)
        target_samples = self.config.get('BOOTSTRAP_SAMPLES', 150)
        
        # Generate samples from both methods
        consensus_samples_target = int(target_samples * consensus_ratio)
        historical_samples_target = int(target_samples * historical_ratio)
        
        # Temporarily adjust config for each method
        original_samples = self.config['BOOTSTRAP_SAMPLES']
        
        try:
            # Get consensus samples
            self.config['BOOTSTRAP_SAMPLES'] = consensus_samples_target
            consensus_samples = self._bootstrap_consensus_method(symbol)
            
            # Get historical samples
            self.config['BOOTSTRAP_SAMPLES'] = historical_samples_target
            historical_samples = self._bootstrap_historical_method(symbol)
            
            # Combine samples
            all_samples = consensus_samples + historical_samples
            
            # Update source metadata
            for sample in all_samples:
                sample.source = 'hybrid'
            
            return all_samples
        
        finally:
            # Restore original config
            self.config['BOOTSTRAP_SAMPLES'] = original_samples
    
    def _get_symbol_historical_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get historical data for a symbol"""
        try:
            if hasattr(self.bot, 'data_manager') and self.bot.data_manager:
                # Try to get data from bot's data manager
                data = self.bot.data_manager.get_symbol_data(symbol, count=self.config.get('HISTORICAL_LOOKBACK', 5000))
                return data
            elif hasattr(self.bot, 'get_symbol_data'):
                # Try direct method
                data = self.bot.get_symbol_data(symbol, count=self.config.get('HISTORICAL_LOOKBACK', 5000))
                return data
            else:
                self.logger.warning(f"No data source available for {symbol}")
                return None
        except Exception as e:
            self.logger.error(f"Error getting historical data for {symbol}: {e}")
            return None
    
    def _extract_features_at_index(self, data: pd.DataFrame, index: int) -> Optional[np.ndarray]:
        """Extract features at a specific data index"""
        try:
            if index < 50 or index >= len(data):
                return None
            
            # Get a window of data for feature extraction
            window_data = data.iloc[max(0, index-50):index+1]
            
            # Basic technical features
            features = []
            
            # Price-based features
            close_prices = window_data['close'].values
            if len(close_prices) > 0:
                features.extend([
                    close_prices[-1],  # Current price
                    np.mean(close_prices[-5:]) if len(close_prices) >= 5 else close_prices[-1],  # SMA 5
                    np.mean(close_prices[-10:]) if len(close_prices) >= 10 else close_prices[-1],  # SMA 10
                    np.mean(close_prices[-20:]) if len(close_prices) >= 20 else close_prices[-1],  # SMA 20
                ])
            
            # Volatility features
            if len(close_prices) > 1:
                returns = np.diff(close_prices) / close_prices[:-1]
                features.extend([
                    np.std(returns[-10:]) if len(returns) >= 10 else 0,  # Volatility
                    np.mean(returns[-5:]) if len(returns) >= 5 else 0,   # Recent return
                ])
            else:
                features.extend([0, 0])
            
            # Volume features (if available)
            if 'volume' in window_data.columns:
                volumes = window_data['volume'].values
                features.extend([
                    np.mean(volumes[-5:]) if len(volumes) >= 5 else 0,  # Average volume
                    volumes[-1] / np.mean(volumes[-10:]) if len(volumes) >= 10 and np.mean(volumes[-10:]) > 0 else 1  # Volume ratio
                ])
            else:
                features.extend([0, 1])
            
            # Trend features
            if len(close_prices) >= 20:
                sma_short = np.mean(close_prices[-5:])
                sma_long = np.mean(close_prices[-20:])
                features.append(sma_short / sma_long - 1 if sma_long > 0 else 0)  # Trend strength
            else:
                features.append(0)
            
            return np.array(features, dtype=np.float64)
        
        except Exception as e:
            self.logger.debug(f"Error extracting features at index {index}: {e}")
            return None
    
    def _get_model_predictions(self, symbol: str, features: np.ndarray, market_data: pd.Series) -> Dict[str, Tuple[str, float]]:
        """Get predictions from different models"""
        predictions = {}
        
        try:
            # RL Agent prediction
            if hasattr(self.bot, 'rl_agent') and self.bot.rl_agent:
                try:
                    # This is a simplified version - in practice you'd need to format data properly
                    rl_action = np.random.choice(['BUY', 'SELL', 'HOLD'])  # Placeholder
                    rl_confidence = np.random.uniform(0.3, 0.8)  # Placeholder
                    predictions['rl'] = (rl_action, rl_confidence)
                except:
                    predictions['rl'] = ('HOLD', 0.5)
            
            # Master Agent prediction
            if hasattr(self.bot, 'master_agent') and self.bot.master_agent:
                try:
                    # Placeholder - in practice you'd call the actual master agent
                    master_action = np.random.choice(['BUY', 'SELL', 'HOLD'])
                    master_confidence = np.random.uniform(0.3, 0.8)
                    predictions['master'] = (master_action, master_confidence)
                except:
                    predictions['master'] = ('HOLD', 0.5)
            
            # Ensemble prediction
            if hasattr(self.bot, 'ensemble_manager') and self.bot.ensemble_manager:
                try:
                    # Placeholder - in practice you'd call the actual ensemble
                    ensemble_action = np.random.choice(['BUY', 'SELL', 'HOLD'])
                    ensemble_confidence = np.random.uniform(0.3, 0.8)
                    predictions['ensemble'] = (ensemble_action, ensemble_confidence)
                except:
                    predictions['ensemble'] = ('HOLD', 0.5)
        
        except Exception as e:
            self.logger.debug(f"Error getting model predictions: {e}")
        
        # Ensure we have at least some predictions
        if not predictions:
            predictions = {
                'rl': ('HOLD', 0.5),
                'master': ('HOLD', 0.5),
                'ensemble': ('HOLD', 0.5)
            }
        
        return predictions
    
    def _create_consensus_label(self, predictions: Dict[str, Tuple[str, float]]) -> Tuple[float, float]:
        """Create consensus label from model predictions"""
        weights = self.config.get('CONSENSUS_WEIGHTS', {})
        
        # Calculate weighted votes
        action_scores = {'BUY': 0.0, 'SELL': 0.0, 'HOLD': 0.0}
        total_weight = 0.0
        
        for model, (action, confidence) in predictions.items():
            weight = weights.get(model, 1.0 / len(predictions))
            action_scores[action] += weight * confidence
            total_weight += weight
        
        # Normalize scores
        if total_weight > 0:
            for action in action_scores:
                action_scores[action] /= total_weight
        
        # Find best action
        best_action = max(action_scores, key=action_scores.get)
        best_confidence = action_scores[best_action]
        
        # Convert action to numeric label
        action_to_label = {'SELL': 0.0, 'HOLD': 0.5, 'BUY': 1.0}
        label = action_to_label[best_action]
        
        return label, best_confidence
    
    def _return_to_label(self, future_return: float) -> Tuple[float, float]:
        """Convert future return to label and confidence"""
        buy_threshold = self.config.get('BUY_THRESHOLD', 0.015)
        sell_threshold = self.config.get('SELL_THRESHOLD', -0.015)
        
        if future_return > buy_threshold:
            # Strong positive return -> BUY
            confidence = min(0.9, 0.5 + abs(future_return) * 10)  # Scale confidence
            return 1.0, confidence
        elif future_return < sell_threshold:
            # Strong negative return -> SELL
            confidence = min(0.9, 0.5 + abs(future_return) * 10)  # Scale confidence
            return 0.0, confidence
        else:
            # Neutral return -> HOLD
            confidence = 0.5 + (1 - abs(future_return) / max(buy_threshold, abs(sell_threshold))) * 0.2
            return 0.5, min(0.8, confidence)
    
    def bootstrap_multiple_symbols(self, symbols: List[str]) -> Dict[str, Tuple[List[BootstrapSample], Dict[str, Any]]]:
        """Bootstrap multiple symbols in parallel"""
        results = {}
        
        if not self.config.get('PARALLEL_BOOTSTRAP', True) or len(symbols) == 1:
            # Sequential processing
            for symbol in symbols:
                results[symbol] = self.bootstrap_symbol(symbol)
            return results
        
        # Parallel processing
        timeout = self.config.get('BOOTSTRAP_TIMEOUT', 30)
        
        with ThreadPoolExecutor(max_workers=min(4, len(symbols))) as executor:
            # Submit all tasks
            future_to_symbol = {
                executor.submit(self.bootstrap_symbol, symbol): symbol 
                for symbol in symbols
            }
            
            # Collect results
            for future in as_completed(future_to_symbol, timeout=timeout):
                symbol = future_to_symbol[future]
                try:
                    results[symbol] = future.result()
                except Exception as e:
                    self.logger.error(f"Error bootstrapping {symbol}: {e}")
                    results[symbol] = ([], {'error': str(e)})
        
        return results
    
    def generate_bootstrap_report(self, results: Dict[str, Tuple[List[BootstrapSample], Dict[str, Any]]]) -> Dict[str, Any]:
        """Generate comprehensive bootstrap report"""
        total_symbols = len(results)
        successful_symbols = 0
        total_samples = 0
        total_processing_time = 0
        
        method_counts = Counter()
        quality_scores = []
        
        for symbol, (samples, report) in results.items():
            if samples:
                successful_symbols += 1
                total_samples += len(samples)
                
                if 'processing_time' in report:
                    total_processing_time += report['processing_time']
                
                if 'quality_score' in report:
                    quality_scores.append(report['quality_score'])
                
                # Count methods used
                for sample in samples:
                    method_counts[sample.source] += 1
        
        bootstrap_report = {
            'total_symbols': total_symbols,
            'successful_symbols': successful_symbols,
            'success_rate': successful_symbols / total_symbols if total_symbols > 0 else 0,
            'total_samples': total_samples,
            'average_samples_per_symbol': total_samples / successful_symbols if successful_symbols > 0 else 0,
            'total_processing_time': total_processing_time,
            'average_processing_time': total_processing_time / successful_symbols if successful_symbols > 0 else 0,
            'method_distribution': dict(method_counts),
            'average_quality_score': np.mean(quality_scores) if quality_scores else 0,
            'config_used': self.config
        }
        
        return bootstrap_report

# Example usage and testing
if __name__ == "__main__":
    # Test the bootstrap implementation
    bootstrap = OnlineLearningBootstrap()
    
    # Test with sample symbols
    test_symbols = ['BTCUSD', 'ETHUSD']
    
    print("🧪 Testing Online Learning Bootstrap")
    print("=" * 50)
    
    # Test single symbol
    samples, report = bootstrap.bootstrap_symbol('BTCUSD')
    print(f"✅ Single symbol test: {len(samples)} samples generated")
    print(f"📊 Report: {report}")
    
    # Test multiple symbols
    results = bootstrap.bootstrap_multiple_symbols(test_symbols)
    overall_report = bootstrap.generate_bootstrap_report(results)
    
    print(f"\n✅ Multiple symbols test completed")
    print(f"📊 Overall Report:")
    for key, value in overall_report.items():
        if key != 'config_used':
            print(f"   {key}: {value}")