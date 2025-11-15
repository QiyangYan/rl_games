#!/bin/bash
# Quick setup script for S3 checkpoint support

echo "=========================================="
echo "RL Games S3 Checkpoint Setup"
echo "=========================================="
echo ""

# Install boto3
echo "Installing boto3..."
pip install boto3

echo ""
echo "Checking AWS credentials..."
if aws sts get-caller-identity &>/dev/null; then
    echo "✓ AWS credentials are configured"
    aws sts get-caller-identity
else
    echo "⚠ AWS credentials not found or not configured"
    echo ""
    echo "Please configure AWS credentials using one of these methods:"
    echo ""
    echo "1. Run: aws configure"
    echo "   (You'll need your AWS Access Key ID and Secret Access Key)"
    echo ""
    echo "2. Set environment variables:"
    echo "   export AWS_ACCESS_KEY_ID=your_access_key"
    echo "   export AWS_SECRET_ACCESS_KEY=your_secret_key"
    echo "   export AWS_DEFAULT_REGION=us-east-1"
    echo ""
    echo "3. If running on EC2/ECS, attach an IAM role with S3 permissions"
fi

echo ""
echo "=========================================="
echo "Testing S3 bucket access..."
echo "=========================================="

S3_PATH="s3://far-research-internal/qiyang/"

if aws s3 ls "$S3_PATH" &>/dev/null; then
    echo "✓ Successfully accessed: $S3_PATH"
    echo ""
    echo "Contents:"
    aws s3 ls "$S3_PATH"
else
    echo "⚠ Could not access: $S3_PATH"
    echo ""
    echo "Please verify:"
    echo "1. The bucket exists"
    echo "2. You have the necessary permissions (s3:ListBucket, s3:GetObject, s3:PutObject)"
    echo "3. The bucket path is correct"
fi

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "To use S3 checkpoints, set train_dir in your config:"
echo "  train_dir: s3://far-research-internal/qiyang/"
echo ""
echo "Run the test script to verify:"
echo "  python test_s3_checkpoints.py"
echo ""
