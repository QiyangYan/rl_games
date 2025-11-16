#!/usr/bin/env python3
"""
Quick script to verify S3 write access before starting training.
This catches AWS credential or permission issues early.

Usage:
    python verify_s3_access.py s3://bucket-name/path/
"""

import sys
import os

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rl_games.common import s3_utils


def main():
    if len(sys.argv) < 2:
        print("Usage: python verify_s3_access.py s3://bucket-name/path/")
        print("\nExample:")
        print("  python verify_s3_access.py s3://far-research-internal/qiyang/")
        sys.exit(1)
    
    s3_path = sys.argv[1]
    
    print("=" * 60)
    print("S3 Access Verification")
    print("=" * 60)
    print(f"Testing S3 path: {s3_path}")
    print()
    
    if not s3_utils.is_s3_path(s3_path):
        print(f"ERROR: '{s3_path}' is not an S3 path")
        print("S3 paths must start with 's3://'")
        sys.exit(1)
    
    # Verify access
    success = s3_utils.verify_s3_access(s3_path)
    
    print()
    print("=" * 60)
    if success:
        print("✓ SUCCESS: S3 access verified!")
        print()
        print("You can now use this path in your rl_games config:")
        print(f"  train_dir: {s3_path}")
        print()
        print("All checkpoints and summaries will be saved to S3.")
    else:
        print("✗ FAILED: Could not verify S3 access")
        print()
        print("Common issues:")
        print("1. AWS credentials not configured")
        print("   Run: aws configure")
        print()
        print("2. Incorrect bucket name or no access to bucket")
        print("   Check: aws s3 ls " + s3_path)
        print()
        print("3. Missing S3 permissions")
        print("   Required: s3:PutObject, s3:GetObject, s3:DeleteObject")
        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    main()
