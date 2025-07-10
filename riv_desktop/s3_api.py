from fastapi import APIRouter, Query, Request, HTTPException, Body
from fastapi.responses import JSONResponse, StreamingResponse
import boto3
from pathlib import Path
import tempfile
import uuid
import os
import logging
import time
import zlib
import io
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("s3_logger")

# Load .env from current directory
load_dotenv(Path(".env"))

router = APIRouter()

def env_credentials_present():
    return all([
        os.getenv("AWS_ACCESS_KEY_ID"),
        os.getenv("AWS_SECRET_ACCESS_KEY"),
        os.getenv("AWS_DEFAULT_REGION"),
        os.getenv("AWS_S3_BUCKET"),
    ])

def save_credentials_to_env(credentials):
    """Save credentials to .env file"""
    try:
        env_path = Path(".env")
        
        # Read existing .env content
        existing_content = ""
        if env_path.exists():
            with open(env_path, 'r') as f:
                existing_content = f.read()
        
        # Prepare new credentials
        new_lines = [
            f"AWS_ACCESS_KEY_ID={credentials['access_key']}",
            f"AWS_SECRET_ACCESS_KEY={credentials['secret_key']}",
            f"AWS_DEFAULT_REGION={credentials['region']}",
            f"AWS_S3_BUCKET={credentials['bucket']}"
        ]
        
        # Remove existing AWS credentials if present
        lines = existing_content.split('\n')
        filtered_lines = [line for line in lines if not any(
            line.startswith(key) for key in ['AWS_ACCESS_KEY_ID=', 'AWS_SECRET_ACCESS_KEY=', 'AWS_DEFAULT_REGION=', 'AWS_S3_BUCKET=']
        )]
        
        # Add new credentials
        filtered_lines.extend(new_lines)
        
        # Write back to file
        with open(env_path, 'w') as f:
            f.write('\n'.join(line for line in filtered_lines if line.strip()))
            f.write('\n')
        
        logger.info(f"Credentials saved to {env_path}")
        return True
        
    except Exception as e:
        logger.error(f"Could not save to .env file: {str(e)}")
        return False

def calculate_s3_object_crc(s3_client, bucket: str, key: str) -> str:
    """Calculate CRC32 for an S3 object without downloading the entire file."""
    try:
        # For large files, we'll use the ETag as a proxy for CRC
        # For small files, we can download and calculate actual CRC
        response = s3_client.head_object(Bucket=bucket, Key=key)
        
        # Get file size
        file_size = response.get('ContentLength', 0)
        
        if file_size < 10 * 1024 * 1024:  # Less than 10MB, calculate actual CRC
            obj = s3_client.get_object(Bucket=bucket, Key=key)
            content = obj['Body'].read()
            crc = zlib.crc32(content) & 0xFFFFFFFF
            return format(crc, '08x')
        else:
            # For larger files, use ETag + metadata as CRC proxy
            etag = response.get('ETag', '').strip('"')
            last_modified = response.get('LastModified', '').isoformat() if response.get('LastModified') else ''
            
            # Create a composite CRC from metadata
            metadata_str = f"{key}:{etag}:{last_modified}:{file_size}"
            crc = zlib.crc32(metadata_str.encode('utf-8')) & 0xFFFFFFFF
            return format(crc, '08x')
            
    except Exception as e:
        logger.warning(f"Could not calculate CRC for {key}: {str(e)}")
        # Fallback to path-based CRC
        crc = zlib.crc32(key.encode('utf-8')) & 0xFFFFFFFF
        return format(crc, '08x')

# Initialize S3 variables
s3 = None
bucket_name = None

if env_credentials_present():
    print("[S3 INIT] Using credentials from .env")
    try:
        # Configure S3 client with proper timeouts and retries
        from botocore.config import Config
        
        config = Config(
            region_name=os.getenv("AWS_DEFAULT_REGION"),
            retries={
                'max_attempts': 10,
                'mode': 'adaptive'
            },
            max_pool_connections=50,
            connect_timeout=60,
            read_timeout=60
        )
        
        s3 = boto3.client(
            "s3",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            config=config
        )
        bucket_name = os.getenv("AWS_S3_BUCKET")
        print(f"[S3 INIT] Connected to bucket: {bucket_name}")
    except Exception as e:
        print(f"[S3 INIT] Error initializing S3: {e}")
        s3 = None
        bucket_name = None
else:
    print("[S3 INIT] .env credentials not found — will show web form")

@router.get("/api/s3-status")
async def get_s3_status():
    """Check if S3 is configured and accessible"""
    global s3, bucket_name
    
    if not s3 or not bucket_name:
        return {
            "configured": False,
            "needs_credentials": True,
            "message": "S3 credentials not configured"
        }
    
    try:
        # Test connection by checking if the specific bucket exists and is accessible
        s3.head_bucket(Bucket=bucket_name)
        return {
            "configured": True,
            "needs_credentials": False,
            "bucket": bucket_name,
            "message": "S3 configured and accessible"
        }
    except Exception as e:
        return {
            "configured": False,
            "needs_credentials": True,
            "error": str(e),
            "message": "S3 configured but not accessible"
        }

@router.post("/api/set-s3-credentials")
async def set_s3_credentials(request: Request):
    """Set S3 credentials from the frontend form"""
    global s3, bucket_name
    
    try:
        data = await request.json()
        access_key = data.get('accessKey')
        secret_key = data.get('secretKey')
        region = data.get('region')
        bucket = data.get('bucket')
        save_to_env = data.get('saveToEnv', False)
        
        if not all([access_key, secret_key, region, bucket]):
            raise HTTPException(status_code=400, detail="All fields are required")
        
        # Test the credentials with proper timeout configuration
        from botocore.config import Config
        
        config = Config(
            region_name=region,
            retries={
                'max_attempts': 10,
                'mode': 'adaptive'
            },
            max_pool_connections=50,
            connect_timeout=60,
            read_timeout=60
        )
        
        test_s3 = boto3.client(
            's3',
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            config=config
        )
        
        # Test connection
        test_s3.head_bucket(Bucket=bucket)
        
        # If successful, update global variables
        s3 = test_s3
        bucket_name = bucket
        
        response_data = {
            "message": "S3 credentials set successfully",
            "bucket": bucket,
            "region": region
        }
        
        # Save to .env if requested
        if save_to_env:
            credentials = {
                'access_key': access_key,
                'secret_key': secret_key,
                'region': region,
                'bucket': bucket
            }
            if save_credentials_to_env(credentials):
                response_data["saved_to_env"] = True
            else:
                response_data["saved_to_env"] = False
                response_data["env_warning"] = "Could not save to .env file"
        
        return JSONResponse(content=response_data)
        
    except Exception as e:
        logger.error(f"Error setting S3 credentials: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid credentials or bucket: {str(e)}")


@router.get("/api/s3-flat-list")
async def get_s3_flat_list(use_parallel: bool = Query(False, description="Use parallel processing for faster listing")):
    global s3, bucket_name

    if not s3 or not bucket_name:
        return JSONResponse(
            status_code=503,
            content={
                "error": "S3 not configured. Please set credentials first.",
                "needs_credentials": True
            }
        )

    if use_parallel:
        # Use the new fast listing endpoint
        logger.info("Using parallel processing for flat list")
        return await s3_fast_list(use_parallel=True, auto_prefixes=True)
    
    # Original single-threaded implementation
    try:
        paginator = s3.get_paginator("list_objects_v2")
        files = []

        for page in paginator.paginate(Bucket=bucket_name):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if not key.lower().endswith((".dcm", ".e2e", ".fds", ".dicom",".fda")):
                    continue

                files.append({
                    "key": key,
                    "size": obj["Size"],
                    "last_modified": obj["LastModified"].isoformat()
                })

        logger.info(f"S3 flat list loaded with {len(files)} items")
        return files

    except Exception as e:
        logger.error(f"Error loading S3 flat list: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})


def generate_prefixes_from_pattern(pattern: str = "year/month", max_prefixes: int = 50):
    """
    Generate a list of prefixes based on common patterns.
    Useful for organizing large S3 buckets.
    """
    prefixes = []
    
    if pattern == "year/month":
        # Generate prefixes like "2024/01/", "2024/02/", etc.
        current_year = time.localtime().tm_year
        for year in range(current_year - 2, current_year + 1):  # Last 2 years + current
            for month in range(1, 13):
                prefixes.append(f"{year}/{month:02d}/")
                if len(prefixes) >= max_prefixes:
                    return prefixes
                    
    elif pattern == "alphabetical":
        # Generate prefixes like "A/", "B/", etc.
        for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            prefixes.append(f"{letter}/")
            if len(prefixes) >= max_prefixes:
                return prefixes
                
    elif pattern == "numerical":
        # Generate prefixes like "0/", "1/", etc.
        for num in range(10):
            prefixes.append(f"{num}/")
            if len(prefixes) >= max_prefixes:
                return prefixes
    
    return prefixes


@router.get("/api/s3-generate-prefixes")
async def generate_prefixes(
    pattern: str = Query("year/month", description="Pattern for prefix generation"),
    max_prefixes: int = Query(50, description="Maximum number of prefixes to generate")
):
    """
    Generate a list of prefixes based on common patterns.
    """
    try:
        prefixes = generate_prefixes_from_pattern(pattern, max_prefixes)
        return {
            "prefixes": prefixes,
            "pattern": pattern,
            "total_prefixes": len(prefixes)
        }
    except Exception as e:
        logger.error(f"Error generating prefixes: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.get("/api/download_dicom_from_s3")
async def download_dicom_from_s3(path: str = Query(...)):
    global s3, bucket_name

    if not s3 or not bucket_name:
        return JSONResponse(
            status_code=503,
            content={
                "error": "S3 not configured. Please set credentials first.",
                "needs_credentials": True
            }
        )

    try:
        from main import process_dicom_file, process_e2e_file, process_fds_file,process_fda_file, stored_images, load_from_cache
    except ImportError:
        logger.error("Could not import processing functions from main")
        raise HTTPException(status_code=500, detail="Server configuration error")

    # Generate CRC for this file path and metadata
    try:
        # Get S3 object metadata for CRC calculation
        head_response = s3.head_object(Bucket=bucket_name, Key=path)
        file_size = head_response.get('ContentLength', 0)
        last_modified = head_response.get('LastModified', '').isoformat() if head_response.get('LastModified') else ''
        etag = head_response.get('ETag', '').strip('"')
        
        # Calculate CRC based on file metadata (same as save-to-cache)
        metadata_str = f"{path}:{etag}:{last_modified}:{file_size}"
        crc = format(zlib.crc32(metadata_str.encode('utf-8')) & 0xFFFFFFFF, '08x')
        
        logger.info(f"Generated CRC for {path}: {crc}")
        
    except Exception as e:
        logger.warning(f"Could not get S3 metadata for {path}: {str(e)}")
        # Fallback CRC based on path only
        crc = format(zlib.crc32(path.encode('utf-8')) & 0xFFFFFFFF, '08x')

    cache_key = path.replace('/', '_').replace('.', '_')

    # First check disk cache
    try:
        cached_images, metadata = load_from_cache(crc)
        if cached_images:
            logger.info(f"Disk cache hit for {path} (CRC: {crc})")
            # Load into memory cache
            key = str(uuid.uuid4())
            stored_images[key] = cached_images
            stored_images[key]["timestamp"] = time.time()
            stored_images[key]["crc"] = crc
            stored_images[key]["s3_key"] = path
            
            # Handle E2E files specially
            if cached_images.get("file_type") == "e2e":
                # Count total images for E2E files
                total_images = len([
                    k for k in cached_images.keys()
                    if isinstance(cached_images[k], io.BytesIO)
                ])
                
                logger.info(f"E2E file loaded from cache. Total images: {total_images}")
                logger.info(f"E2E data keys: {list(cached_images.keys())}")
                logger.info(f"Left eye data: {cached_images.get('left_eye_data', {})}")
                logger.info(f"Right eye data: {cached_images.get('right_eye_data', {})}")
                
                return JSONResponse(content={
                    "message": "E2E file loaded from disk cache.",
                    "number_of_frames": total_images,
                    "dicom_file_path": key,
                    "cache_source": "disk",
                    "file_type": "e2e",
                    "left_eye_data": cached_images.get("left_eye_data", {"dicom": [], "oct": []}),
                    "right_eye_data": cached_images.get("right_eye_data", {"dicom": [], "oct": []})
                })
            else:
                # Regular files
                return JSONResponse(content={
                    "message": "File loaded from disk cache.",
                    "number_of_frames": len([k for k in cached_images.keys() if isinstance(k, int)]),
                    "dicom_file_path": key,
                    "cache_source": "disk"
                })
    except Exception as e:
        logger.warning(f"Could not check disk cache for {path}: {str(e)}")

    # Return cached version if present (check memory cache)
    for key, value in stored_images.items():
        if isinstance(value, dict) and (
            value.get("s3_key") == path or 
            value.get("crc") == crc
        ):
            logger.info(f"Memory cache hit for {path} (CRC: {crc})")
            
            # Handle E2E files specially
            if value.get("file_type") == "e2e":
                # Count total images for E2E files
                total_images = len([
                    k for k in value.keys()
                    if isinstance(value[k], io.BytesIO)
                ])
                
                logger.info(f"E2E file loaded from memory cache. Total images: {total_images}")
                logger.info(f"E2E data keys: {list(value.keys())}")
                logger.info(f"Left eye data: {value.get('left_eye_data', {})}")
                logger.info(f"Right eye data: {value.get('right_eye_data', {})}")
                
                return JSONResponse(content={
                    "message": "E2E file loaded from memory cache.",
                    "number_of_frames": total_images,
                    "dicom_file_path": key,
                    "cache_source": "memory",
                    "file_type": "e2e",
                    "left_eye_data": value.get("left_eye_data", {"dicom": [], "oct": []}),
                    "right_eye_data": value.get("right_eye_data", {"dicom": [], "oct": []})
                })
            else:
                # Regular files
                return JSONResponse(content={
                    "message": "File loaded from memory cache.",
                    "number_of_frames": len([k for k in value.keys() if isinstance(k, int)]),
                    "dicom_file_path": key,
                    "cache_source": "memory"
                })

    file_extension = os.path.splitext(path)[1].lower()
    key = str(uuid.uuid4())

    logger.info(f"Downloading {path} from S3 into memory")
    try:
        # Use streaming download with chunked reading to avoid timeouts
        obj = s3.get_object(Bucket=bucket_name, Key=path)
        
        # Create temp file first
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp:
            temp_path = tmp.name
            
            # Stream download in chunks to avoid timeout issues
            chunk_size = 1024 * 1024  # 1MB chunks
            total_size = obj.get('ContentLength', 0)
            downloaded_size = 0
            
            logger.info(f"Starting chunked download of {total_size} bytes")
            
            for chunk in obj['Body'].iter_chunks(chunk_size=chunk_size):
                tmp.write(chunk)
                downloaded_size += len(chunk)
                if total_size > 0:
                    progress = (downloaded_size / total_size) * 100
                    if downloaded_size % (10 * 1024 * 1024) == 0:  # Log every 10MB
                        logger.info(f"Download progress: {progress:.1f}% ({downloaded_size}/{total_size} bytes)")
            
            logger.info(f"Download complete: {downloaded_size} bytes")
            
    except Exception as e:
        logger.error(f"Failed to get S3 object: {str(e)}")
        raise HTTPException(status_code=404, detail=f"Failed to download file: {str(e)}")

    logger.info(f"Downloaded and saved to temp file: {temp_path}")

    stored_images[key] = {
        "local_path": temp_path,
        "s3_key": path,
        "timestamp": time.time(),
        "crc": crc
    }
    
    # Clean up old entries to prevent memory leaks
    cleanup_old_memory_entries()

    try:
        # Process the file and ensure S3 path mapping is updated after processing
        result = None
        if file_extension in ['.dcm', '.dicom']:
            result = process_dicom_file(temp_path, key, crc)
        elif file_extension == '.e2e':
            result = process_e2e_file(temp_path, key, crc)
        elif file_extension in ['.fds']:
            result = process_fds_file(temp_path, key, crc)
        elif file_extension in ['.fda']:
            result = process_fda_file(temp_path, key, crc)
        else:
            os.remove(temp_path)
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_extension}")
        
        # ENSURE S3 path mapping is updated after processing completes
        # This guarantees the frontend can access the file immediately
        if path not in stored_images and key in stored_images:
            stored_images[path] = stored_images[key]
            logger.info(f"[POST-PROCESSING] Ensured S3 path '{path}' is mapped to processed key '{key}'")
        
        return result
        
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        logger.error(f"Processing error for {path}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

# Enhanced endpoint to get S3 object CRC
@router.get("/api/s3-object-crc")
async def get_s3_object_crc(path: str = Query(...)):
    """Get CRC checksum for an S3 object."""
    global s3, bucket_name
    
    if not s3 or not bucket_name:
        raise HTTPException(status_code=503, detail="S3 not configured")
    
    try:
        crc = calculate_s3_object_crc(s3, bucket_name, path)
        logger.info(f"Calculated CRC for S3 object {path}: {crc}")
        
        return {
            "path": path,
            "crc": crc,
            "source": "s3_metadata"
        }
        
    except Exception as e:
        logger.error(f"Error calculating CRC for S3 object {path}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error calculating CRC: {str(e)}")


@router.post("/api/s3-parallel-list")
async def s3_parallel_list(
    prefixes: list[str] = Body(..., embed=True, description="List of S3 prefixes to scan")
):
    """
    List S3 files in parallel for the given prefixes.
    This is much faster than listing the entire bucket for large datasets.
    """
    global s3, bucket_name

    if not s3 or not bucket_name:
        return JSONResponse(
            status_code=503,
            content={
                "error": "S3 not configured. Please set credentials first.",
                "needs_credentials": True
            }
        )

    def list_prefix(prefix):
        """List files for a single prefix"""
        files = []
        try:
            paginator = s3.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
                for obj in page.get("Contents", []):
                    key = obj["Key"]
                    if not key.lower().endswith((".dcm", ".e2e", ".fds", ".dicom", ".fda")):
                        continue
                    files.append({
                        "key": key,
                        "size": obj["Size"],
                        "last_modified": obj["LastModified"].isoformat()
                    })
            logger.info(f"Listed {len(files)} files for prefix: {prefix}")
        except Exception as e:
            logger.error(f"Error listing prefix {prefix}: {str(e)}")
        return files

    start_time = time.time()
    results = []
    
    # Use ThreadPoolExecutor for parallel processing
    max_workers = min(10, len(prefixes))  # Limit to 8 threads max
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_prefix = {executor.submit(list_prefix, prefix): prefix for prefix in prefixes}
        
        for future in as_completed(future_to_prefix):
            prefix = future_to_prefix[future]
            try:
                files = future.result()
                results.extend(files)
            except Exception as e:
                logger.error(f"Error processing prefix {prefix}: {str(e)}")

    end_time = time.time()
    duration = end_time - start_time
    
    logger.info(f"Parallel S3 list completed in {duration:.2f}s: {len(results)} items from {len(prefixes)} prefixes")
    
    return {
        "files": results,
        "total_files": len(results),
        "prefixes_scanned": len(prefixes),
        "duration_seconds": duration,
        "files_per_second": len(results) / duration if duration > 0 else 0
    }


@router.get("/api/s3-top-prefixes")
async def s3_top_prefixes(delimiter: str = "/", max_keys: int = 100):
    """
    Get top-level prefixes (folders) from S3 bucket.
    Useful for generating prefix lists for parallel listing.
    """
    global s3, bucket_name
    
    if not s3 or not bucket_name:
        return JSONResponse(
            status_code=503,
            content={
                "error": "S3 not configured. Please set credentials first.",
                "needs_credentials": True
            }
        )

    try:
        paginator = s3.get_paginator("list_objects_v2")
        prefixes = set()
        
        for page in paginator.paginate(
            Bucket=bucket_name, 
            Delimiter=delimiter, 
            PaginationConfig={"MaxItems": max_keys}
        ):
            for prefix in page.get("CommonPrefixes", []):
                prefixes.add(prefix["Prefix"])
        
        prefix_list = list(prefixes)
        logger.info(f"Found {len(prefix_list)} top-level prefixes")
        
        return {
            "prefixes": prefix_list,
            "total_prefixes": len(prefix_list),
            "delimiter": delimiter
        }
        
    except Exception as e:
        logger.error(f"Error getting top prefixes: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.get("/api/s3-fast-list")
async def s3_fast_list(
    use_parallel: bool = Query(True, description="Use parallel processing"),
    max_workers: int = Query(10, description="Maximum number of parallel workers"),
    auto_prefixes: bool = Query(True, description="Auto-detect prefixes if none provided")
):
    """
    Fast S3 listing with automatic prefix detection and parallel processing.
    This is the recommended endpoint for large buckets.
    """
    global s3, bucket_name

    if not s3 or not bucket_name:
        return JSONResponse(
            status_code=503,
            content={
                "error": "S3 not configured. Please set credentials first.",
                "needs_credentials": True
            }
        )

    start_time = time.time()
    
    try:
        if auto_prefixes:
            # Auto-detect prefixes
            logger.info("Auto-detecting top-level prefixes...")
            prefix_response = await s3_top_prefixes()
            if isinstance(prefix_response, dict) and "prefixes" in prefix_response:
                prefixes = prefix_response["prefixes"]
            else:
                # Fallback to single-threaded listing
                logger.warning("Could not auto-detect prefixes, falling back to single-threaded listing")
                return await get_s3_flat_list()
        else:
            # Use root prefix for single-threaded listing
            prefixes = [""]

        if use_parallel and len(prefixes) > 1:
            # Use parallel processing
            logger.info(f"Using parallel processing with {len(prefixes)} prefixes")
            
            def list_prefix(prefix):
                files = []
                try:
                    paginator = s3.get_paginator("list_objects_v2")
                    for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
                        for obj in page.get("Contents", []):
                            key = obj["Key"]
                            if not key.lower().endswith((".dcm", ".e2e", ".fds", ".dicom", ".fda")):
                                continue
                            files.append({
                                "key": key,
                                "size": obj["Size"],
                                "last_modified": obj["LastModified"].isoformat()
                            })
                except Exception as e:
                    logger.error(f"Error listing prefix {prefix}: {str(e)}")
                return files

            results = []
            workers = min(max_workers, len(prefixes))
            
            with ThreadPoolExecutor(max_workers=workers) as executor:
                future_to_prefix = {executor.submit(list_prefix, prefix): prefix for prefix in prefixes}
                
                for future in as_completed(future_to_prefix):
                    prefix = future_to_prefix[future]
                    try:
                        files = future.result()
                        results.extend(files)
                    except Exception as e:
                        logger.error(f"Error processing prefix {prefix}: {str(e)}")
        else:
            # Use single-threaded listing
            logger.info("Using single-threaded listing")
            results = await get_s3_flat_list()
            if isinstance(results, list):
                pass  # Already in correct format
            else:
                # Handle error response
                return results

        end_time = time.time()
        duration = end_time - start_time
        
        logger.info(f"Fast S3 list completed in {duration:.2f}s: {len(results)} items")
        
        return {
            "files": results,
            "total_files": len(results),
            "duration_seconds": duration,
            "files_per_second": len(results) / duration if duration > 0 else 0,
            "method": "parallel" if use_parallel and len(prefixes) > 1 else "single-threaded"
        }
        
    except Exception as e:
        logger.error(f"Error in fast S3 listing: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})


def cleanup_old_memory_entries(max_age_hours: int = 24, max_entries: int = 100):
    """Clean up old memory cache entries to prevent memory leaks."""
    try:
        from main import stored_images
        
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        # Remove old entries
        keys_to_remove = []
        for key, value in stored_images.items():
            if isinstance(value, dict) and "timestamp" in value:
                age = current_time - value["timestamp"]
                if age > max_age_seconds:
                    keys_to_remove.append(key)
        
        # Remove old entries
        for key in keys_to_remove:
            # Clean up temp files
            if "local_path" in stored_images[key]:
                try:
                    temp_path = stored_images[key]["local_path"]
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                except Exception as e:
                    logger.warning(f"Could not remove temp file {temp_path}: {str(e)}")
            del stored_images[key]
        
        # If still too many entries, remove oldest ones
        if len(stored_images) > max_entries:
            sorted_entries = sorted(
                stored_images.items(),
                key=lambda x: x[1].get("timestamp", 0) if isinstance(x[1], dict) else 0
            )
            entries_to_remove = len(stored_images) - max_entries
            
            for i in range(entries_to_remove):
                key, value = sorted_entries[i]
                if "local_path" in value:
                    try:
                        temp_path = value["local_path"]
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                    except Exception as e:
                        logger.warning(f"Could not remove temp file {temp_path}: {str(e)}")
                del stored_images[key]
        
        if keys_to_remove or len(stored_images) > max_entries:
            logger.info(f"Cleaned up {len(keys_to_remove)} old memory entries")
            
    except Exception as e:
        logger.error(f"Error cleaning up memory entries: {str(e)}")