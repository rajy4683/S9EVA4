"""
S3 Upload Script for ImageNet Subset
Uploads downloaded ImageNet data to AWS S3 bucket.
"""

import os
import argparse
from pathlib import Path
from typing import Optional
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from tqdm import tqdm


class S3Uploader:
    """Upload local ImageNet dataset to S3."""

    def __init__(
        self,
        bucket_name: str,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        region_name: str = 'us-east-1'
    ):
        """
        Initialize S3 uploader.

        Args:
            bucket_name: Name of the S3 bucket
            aws_access_key_id: AWS access key (or set via env/config)
            aws_secret_access_key: AWS secret key (or set via env/config)
            region_name: AWS region name
        """
        self.bucket_name = bucket_name

        # Initialize S3 client
        session_kwargs = {'region_name': region_name}
        if aws_access_key_id and aws_secret_access_key:
            session_kwargs.update({
                'aws_access_key_id': aws_access_key_id,
                'aws_secret_access_key': aws_secret_access_key
            })

        self.s3_client = boto3.client('s3', **session_kwargs)

    def create_bucket_if_not_exists(self) -> bool:
        """
        Create S3 bucket if it doesn't exist.

        Returns:
            True if bucket exists or was created successfully
        """
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            print(f"Bucket '{self.bucket_name}' already exists.")
            return True
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                try:
                    print(f"Creating bucket '{self.bucket_name}'...")
                    self.s3_client.create_bucket(Bucket=self.bucket_name)
                    print(f"Bucket '{self.bucket_name}' created successfully.")
                    return True
                except ClientError as create_error:
                    print(f"Error creating bucket: {create_error}")
                    return False
            else:
                print(f"Error checking bucket: {e}")
                return False

    def upload_file(self, local_path: Path, s3_key: str) -> bool:
        """
        Upload a single file to S3.

        Args:
            local_path: Local file path
            s3_key: S3 object key (path in bucket)

        Returns:
            True if upload successful
        """
        try:
            self.s3_client.upload_file(
                str(local_path),
                self.bucket_name,
                s3_key
            )
            return True
        except FileNotFoundError:
            print(f"File not found: {local_path}")
            return False
        except NoCredentialsError:
            print("AWS credentials not found.")
            return False
        except Exception as e:
            print(f"Error uploading {local_path}: {e}")
            return False

    def upload_directory(
        self,
        local_dir: str,
        s3_prefix: str = "imagenet_subset",
        file_extensions: tuple = ('.jpg', '.jpeg', '.png', '.txt')
    ) -> dict:
        """
        Upload entire directory to S3.

        Args:
            local_dir: Local directory path
            s3_prefix: Prefix for S3 keys (folder in bucket)
            file_extensions: Tuple of file extensions to upload

        Returns:
            Dictionary with upload statistics
        """
        local_path = Path(local_dir)

        if not local_path.exists():
            print(f"Error: Directory {local_dir} does not exist.")
            return {'success': 0, 'failed': 0, 'skipped': 0}

        # Create bucket if needed
        if not self.create_bucket_if_not_exists():
            print("Cannot proceed without valid bucket.")
            return {'success': 0, 'failed': 0, 'skipped': 0}

        # Collect all files to upload
        files_to_upload = []
        for root, dirs, files in os.walk(local_path):
            for file in files:
                if file.lower().endswith(file_extensions):
                    file_path = Path(root) / file
                    # Create relative path for S3 key
                    rel_path = file_path.relative_to(local_path)
                    s3_key = f"{s3_prefix}/{rel_path}".replace("\\", "/")
                    files_to_upload.append((file_path, s3_key))

        print(f"\nFound {len(files_to_upload)} files to upload.")

        # Upload files with progress bar
        stats = {'success': 0, 'failed': 0, 'skipped': 0}

        for local_file, s3_key in tqdm(files_to_upload, desc="Uploading to S3"):
            if self.upload_file(local_file, s3_key):
                stats['success'] += 1
            else:
                stats['failed'] += 1

        print(f"\n{'='*60}")
        print(f"Upload complete!")
        print(f"Successful: {stats['success']}")
        print(f"Failed: {stats['failed']}")
        print(f"Bucket: s3://{self.bucket_name}/{s3_prefix}/")
        print(f"{'='*60}")

        return stats

    def list_bucket_contents(self, prefix: str = "", max_keys: int = 10):
        """
        List contents of S3 bucket.

        Args:
            prefix: Filter by prefix
            max_keys: Maximum number of keys to display
        """
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix,
                MaxKeys=max_keys
            )

            if 'Contents' in response:
                print(f"\nListing up to {max_keys} objects in s3://{self.bucket_name}/{prefix}")
                for obj in response['Contents']:
                    print(f"  - {obj['Key']} ({obj['Size']} bytes)")
            else:
                print(f"No objects found with prefix '{prefix}'")

        except Exception as e:
            print(f"Error listing bucket contents: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Upload ImageNet subset to AWS S3"
    )
    parser.add_argument(
        "--local_dir",
        type=str,
        required=True,
        help="Local directory containing the dataset"
    )
    parser.add_argument(
        "--bucket_name",
        type=str,
        required=True,
        help="S3 bucket name"
    )
    parser.add_argument(
        "--s3_prefix",
        type=str,
        default="imagenet_subset",
        help="S3 prefix (folder) for uploaded files"
    )
    parser.add_argument(
        "--region",
        type=str,
        default="us-east-1",
        help="AWS region (default: us-east-1)"
    )
    parser.add_argument(
        "--aws_access_key_id",
        type=str,
        default=None,
        help="AWS access key ID (optional, can use env vars)"
    )
    parser.add_argument(
        "--aws_secret_access_key",
        type=str,
        default=None,
        help="AWS secret access key (optional, can use env vars)"
    )
    parser.add_argument(
        "--list_after_upload",
        action="store_true",
        help="List bucket contents after upload"
    )

    args = parser.parse_args()

    # Create uploader
    uploader = S3Uploader(
        bucket_name=args.bucket_name,
        aws_access_key_id=args.aws_access_key_id,
        aws_secret_access_key=args.aws_secret_access_key,
        region_name=args.region
    )

    # Upload directory
    stats = uploader.upload_directory(
        local_dir=args.local_dir,
        s3_prefix=args.s3_prefix
    )

    # Optionally list contents
    if args.list_after_upload and stats['success'] > 0:
        uploader.list_bucket_contents(prefix=args.s3_prefix, max_keys=20)


if __name__ == "__main__":
    main()
