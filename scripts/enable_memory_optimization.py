#!/usr/bin/env python3
"""
Script để bật/tắt memory optimization trong config.py
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

def enable_memory_optimization(
    enable_sampling=True,
    sample_fraction=0.1,
    max_products=None,
    max_stores=None,
    max_time_periods=None
):
    """
    Enable memory optimization trong config.py
    """
    config_file = project_root / 'src' / 'config.py'
    
    if not config_file.exists():
        print(f"Error: {config_file} not found!")
        return False
    
    # Read current config
    with open(config_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Create new MEMORY_OPTIMIZATION config
    new_config = f"""MEMORY_OPTIMIZATION = {{
    'enable_sampling': {enable_sampling},
    'sample_fraction': {sample_fraction},
    'max_products': {max_products},
    'max_stores': {max_stores},
    'max_time_periods': {max_time_periods},
    'use_chunking': True,
    'chunk_size': 100000,
}}"""
    
    # Replace existing MEMORY_OPTIMIZATION block
    import re
    pattern = r'MEMORY_OPTIMIZATION\s*=\s*\{[^}]+\}'
    
    if re.search(pattern, content):
        content = re.sub(pattern, new_config, content)
        print("✓ Updated MEMORY_OPTIMIZATION config")
    else:
        # Insert after ACTIVE_DATASET
        content = content.replace(
            "ACTIVE_DATASET = 'freshretail'",
            f"ACTIVE_DATASET = 'freshretail'\n\n{new_config}"
        )
        print("✓ Added MEMORY_OPTIMIZATION config")
    
    # Write back
    with open(config_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("\nMemory optimization settings:")
    print(f"  - Sampling: {enable_sampling}")
    if enable_sampling:
        print(f"  - Sample fraction: {sample_fraction*100:.1f}%")
    if max_products:
        print(f"  - Max products: {max_products}")
    if max_stores:
        print(f"  - Max stores: {max_stores}")
    if max_time_periods:
        print(f"  - Max time periods: {max_time_periods}")
    
    return True

def disable_memory_optimization():
    """Disable memory optimization"""
    return enable_memory_optimization(
        enable_sampling=False,
        sample_fraction=1.0,
        max_products=None,
        max_stores=None,
        max_time_periods=None
    )

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Enable/disable memory optimization')
    parser.add_argument('--enable', action='store_true', help='Enable memory optimization')
    parser.add_argument('--disable', action='store_true', help='Disable memory optimization')
    parser.add_argument('--sample-fraction', type=float, default=0.1, help='Sample fraction (0.0-1.0)')
    parser.add_argument('--max-products', type=int, help='Max number of products')
    parser.add_argument('--max-stores', type=int, help='Max number of stores')
    parser.add_argument('--max-time', type=int, help='Max number of time periods')
    
    args = parser.parse_args()
    
    if args.disable:
        disable_memory_optimization()
    elif args.enable:
        enable_memory_optimization(
            enable_sampling=True,
            sample_fraction=args.sample_fraction,
            max_products=args.max_products,
            max_stores=args.max_stores,
            max_time_periods=args.max_time
        )
    else:
        print("Usage:")
        print("  Enable: python scripts/enable_memory_optimization.py --enable --sample-fraction 0.1")
        print("  Disable: python scripts/enable_memory_optimization.py --disable")
        print("\nExample (limit to 10 products, 2 stores, 24 hours):")
        print("  python scripts/enable_memory_optimization.py --enable --max-products 10 --max-stores 2 --max-time 24")


