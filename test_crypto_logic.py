#!/usr/bin/env python3

import datetime
import pytz
from enum import Enum

class SymbolType(Enum):
    """Enum for symbol types"""
    CRYPTO = "crypto"
    FOREX = "forex"
    EQUITY = "equity"
    COMMODITY = "commodity"
    UNKNOWN = "unknown"

CRYPTO_SYMBOLS = {'BTCUSD', 'ETHUSD', 'XRPUSD', 'LTCUSD', 'ADAUSD'}

def get_symbol_type(symbol: str) -> SymbolType:
    """Get the type of a trading symbol"""
    symbol_upper = symbol.upper()
    
    if symbol_upper in CRYPTO_SYMBOLS:
        return SymbolType.CRYPTO
    else:
        return SymbolType.UNKNOWN

def is_crypto_symbol(symbol: str) -> bool:
    """Check if a symbol represents cryptocurrency"""
    return get_symbol_type(symbol) == SymbolType.CRYPTO

def is_weekend() -> bool:
    """Check if current time is weekend"""
    try:
        now = datetime.datetime.now(pytz.UTC)
        weekday = now.weekday()
        return weekday >= 5  # Saturday = 5, Sunday = 6
    except Exception as e:
        print(f"Weekend check error: {e}")
        now = datetime.datetime.now()
        return now.weekday() >= 5

def is_market_open(symbol, current_time=None):
    """Check if the market for a specific symbol is currently open"""
    try:
        # Crypto symbols trade 24/7
        if is_crypto_symbol(symbol):
            return True
        
        # For other symbols, check if it's weekend
        return not is_weekend()
    except Exception as e:
        print(f"Market open check error for {symbol}: {e}")
        return False

if __name__ == "__main__":
    print('=== CRYPTO TESTING ===')
    print(f'BTCUSD is crypto: {is_crypto_symbol("BTCUSD")}')
    print(f'ETHUSD is crypto: {is_crypto_symbol("ETHUSD")}')
    print(f'BTCUSD symbol type: {get_symbol_type("BTCUSD")}')
    print(f'ETHUSD symbol type: {get_symbol_type("ETHUSD")}')

    print('\n=== MARKET STATUS ===')
    print(f'Is weekend: {is_weekend()}')
    print(f'BTCUSD market open: {is_market_open("BTCUSD")}')
    print(f'ETHUSD market open: {is_market_open("ETHUSD")}')

    print('\n=== TIME INFO ===')
    now_utc = datetime.datetime.now(pytz.UTC)
    print(f'Current UTC time: {now_utc}')
    print(f'Weekday: {now_utc.weekday()} (5=Saturday, 6=Sunday)')