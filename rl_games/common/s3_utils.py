"""
S3 utilities for saving and loading checkpoints to/from AWS S3.
"""
import os
import tempfile
import torch
from urllib.parse import urlparse


def is_s3_path(path):
    """Check if a path is an S3 path (starts with s3://)."""
    return path.startswith('s3://')


def parse_s3_path(s3_path):
    """Parse S3 path into bucket and key."""
    parsed = urlparse(s3_path)
    bucket = parsed.netloc
    key = parsed.path.lstrip('/')
    return bucket, key


def save_checkpoint_to_s3(state, s3_path):
    """
    Save a PyTorch checkpoint to S3.
    
    Args:
        state: The checkpoint state dictionary
        s3_path: S3 path in format s3://bucket-name/path/to/checkpoint.pth
    """
    try:
        import boto3
    except ImportError:
        raise ImportError("boto3 is required for S3 operations. Install it with: pip install boto3")
    
    # Ensure path ends with .pth
    if not s3_path.endswith('.pth'):
        s3_path = s3_path + '.pth'
    
    bucket, key = parse_s3_path(s3_path)
    
    # Save to temporary file first
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as tmp_file:
        tmp_path = tmp_file.name
        torch.save(state, tmp_path)
    
    try:
        # Upload to S3
        s3_client = boto3.client('s3')
        print(f"=> Uploading checkpoint to '{s3_path}'")
        s3_client.upload_file(tmp_path, bucket, key)
        print(f"=> Successfully saved checkpoint to S3: {s3_path}")
        
        # Verify the file exists in S3
        try:
            response = s3_client.head_object(Bucket=bucket, Key=key)
            file_size = response['ContentLength']
            print(f"=> Verified: File exists in S3 ({file_size} bytes)")
        except Exception as e:
            print(f"=> WARNING: Could not verify file in S3: {e}")
            
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def load_checkpoint_from_s3(s3_path):
    """
    Load a PyTorch checkpoint from S3.
    
    Args:
        s3_path: S3 path in format s3://bucket-name/path/to/checkpoint.pth
        
    Returns:
        The loaded checkpoint state dictionary
    """
    try:
        import boto3
    except ImportError:
        raise ImportError("boto3 is required for S3 operations. Install it with: pip install boto3")
    
    bucket, key = parse_s3_path(s3_path)
    
    # Download to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as tmp_file:
        tmp_path = tmp_file.name
    
    try:
        # Download from S3
        s3_client = boto3.client('s3')
        print(f"=> Downloading checkpoint from '{s3_path}'")
        s3_client.download_file(bucket, key, tmp_path)
        
        # Load checkpoint
        state = torch.load(tmp_path)
        print(f"=> Successfully loaded checkpoint from S3: {s3_path}")
        return state
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def s3_path_join(base, *paths):
    """
    Join S3 paths correctly, preserving the s3:// prefix.
    
    Args:
        base: Base S3 path (e.g., 's3://bucket/path')
        *paths: Path components to join
        
    Returns:
        Properly joined S3 path
    """
    if not is_s3_path(base):
        return os.path.join(base, *paths)
    
    # For S3 paths, manually join with '/' to avoid os.path.join issues
    result = base.rstrip('/')
    for path in paths:
        path = path.strip('/')
        if path:
            result = result + '/' + path
    return result


def get_s3_base_path(s3_path):
    """
    Get the base S3 path without filename for directory operations.
    
    Args:
        s3_path: Full S3 path
        
    Returns:
        Base S3 path
    """
    if not is_s3_path(s3_path):
        return s3_path
    
    # For S3 paths, we just return the path as-is since S3 doesn't have real directories
    return s3_path


def verify_s3_access(s3_path, experiment_name=None):
    """
    Verify S3 write access by creating a small test file.
    Also creates a permanent initialization marker file.
    Call this at the beginning of training to catch configuration issues early.
    
    Args:
        s3_path: S3 path to test (e.g., 's3://bucket/path/')
        experiment_name: Optional experiment name to include in the init file
        
    Returns:
        True if S3 access works, False otherwise
    """
    try:
        import boto3
        from datetime import datetime
    except ImportError:
        print("=> WARNING: boto3 not installed, cannot verify S3 access")
        return False
    
    if not is_s3_path(s3_path):
        print(f"=> Not an S3 path, skipping verification: {s3_path}")
        return True
    
    try:
        bucket, key_prefix = parse_s3_path(s3_path)
        s3_client = boto3.client('s3')
        
        # Create a test file key
        test_key = key_prefix.rstrip('/') + '/.s3_access_test.txt'
        test_content = f"S3 access test at {datetime.now().isoformat()}\n"
        
        print(f"=> Verifying S3 write access to s3://{bucket}/{key_prefix}...")
        
        # Write test file
        s3_client.put_object(
            Bucket=bucket,
            Key=test_key,
            Body=test_content.encode('utf-8')
        )
        
        # Verify by reading back
        response = s3_client.get_object(Bucket=bucket, Key=test_key)
        content = response['Body'].read().decode('utf-8')
        
        # Clean up test file
        s3_client.delete_object(Bucket=bucket, Key=test_key)
        
        if content != test_content:
            print(f"=> ✗ S3 verification failed: content mismatch")
            return False
        
        print(f"=> ✓ S3 write access verified successfully")
        
        # Create a permanent initialization marker file
        timestamp = datetime.now()
        init_key = key_prefix.rstrip('/') + '/' + experiment_name + '/training_initialized.txt'
        init_content = f"""Training Initialization
Timestamp: {timestamp.isoformat()}
Date: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}
"""
        if experiment_name:
            init_content += f"Experiment: {experiment_name}\n"
        
        init_content += f"\nThis file marks the start of training with S3 storage.\n"
        
        s3_client.put_object(
            Bucket=bucket,
            Key=init_key,
            Body=init_content.encode('utf-8')
        )
        
        print(f"=> ✓ Created initialization marker: s3://{bucket}/{init_key}")
        return True
            
    except Exception as e:
        print(f"=> ✗ S3 access verification failed: {e}")
        print(f"=> Check:")
        print(f"   1. AWS credentials are configured (aws configure)")
        print(f"   2. You have access to bucket: {bucket}")
        print(f"   3. You have s3:PutObject, s3:GetObject, s3:DeleteObject permissions")
        return False


def save_config_to_s3(config_dict, s3_path, filename='config.yaml'):
    """
    Save a configuration dictionary to S3 as a YAML file.
    
    Args:
        config_dict: Configuration dictionary to save
        s3_path: S3 path (directory) where to save the config
        filename: Name of the config file (default: 'config.yaml')
    """
    try:
        import boto3
        import yaml
        import tempfile
    except ImportError as e:
        print(f"=> WARNING: Cannot save config to S3: {e}")
        return False
    
    if not is_s3_path(s3_path):
        # Not an S3 path, save locally instead
        local_path = os.path.join(s3_path, filename)
        try:
            with open(local_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False)
            print(f"=> Saved config to: {local_path}")
            return True
        except Exception as e:
            print(f"=> WARNING: Could not save config locally: {e}")
            return False
    
    try:
        bucket, key_prefix = parse_s3_path(s3_path)
        s3_client = boto3.client('s3')
        
        # Construct full S3 key
        config_key = key_prefix.rstrip('/') + '/' + filename
        
        # Write config to temp file first
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp_file:
            yaml.dump(config_dict, tmp_file, default_flow_style=False)
            tmp_path = tmp_file.name
        
        try:
            # Upload to S3
            s3_client.upload_file(tmp_path, bucket, config_key)
            print(f"=> Saved config to S3: s3://{bucket}/{config_key}")
            return True
        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
                
    except Exception as e:
        print(f"=> WARNING: Could not save config to S3: {e}")
        return False


def sync_directory_to_s3(local_dir, s3_path, delete_after_sync=False):
    """
    Sync a local directory to S3, uploading all files.
    
    Args:
        local_dir: Local directory path to sync
        s3_path: S3 destination path (e.g., 's3://bucket/path/to/summaries')
        delete_after_sync: If True, delete local files after successful upload
        
    Returns:
        Number of files uploaded
    """
    try:
        import boto3
    except ImportError:
        raise ImportError("boto3 is required for S3 operations. Install it with: pip install boto3")
    
    if not is_s3_path(s3_path):
        print(f"=> Not an S3 path, skipping sync: {s3_path}")
        return 0
    
    if not os.path.exists(local_dir):
        print(f"=> Local directory does not exist: {local_dir}")
        return 0
    
    bucket, key_prefix = parse_s3_path(s3_path)
    s3_client = boto3.client('s3')
    
    uploaded_count = 0
    files_to_delete = []
    
    # Walk through local directory
    for root, dirs, files in os.walk(local_dir):
        for filename in files:
            local_path = os.path.join(root, filename)
            
            # Calculate relative path from local_dir
            rel_path = os.path.relpath(local_path, local_dir)
            
            # Construct S3 key (use forward slashes)
            s3_key = key_prefix.rstrip('/') + '/' + rel_path.replace(os.sep, '/')
            
            try:
                # Upload file
                s3_client.upload_file(local_path, bucket, s3_key)
                uploaded_count += 1
                
                if delete_after_sync:
                    files_to_delete.append(local_path)
                    
            except Exception as e:
                print(f"=> Error uploading {local_path} to S3: {e}")
    
    # Delete local files if requested
    if delete_after_sync:
        for file_path in files_to_delete:
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"=> Error deleting {file_path}: {e}")
    
    if uploaded_count > 0:
        print(f"=> Synced {uploaded_count} files to {s3_path}")
    
    return uploaded_count
