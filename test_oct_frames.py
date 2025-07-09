#!/usr/bin/env python3
"""
Test script for OCT frame extraction functionality
"""

import requests
import json
import time

def test_oct_frame_extraction():
    """Test the enhanced OCT frame extraction functionality"""
    
    base_url = "http://localhost:8000"
    
    print("Testing Enhanced OCT Frame Extraction...")
    print("=" * 50)
    
    # Test 1: Check if server is running
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            print("✅ Server is running")
        else:
            print("❌ Server health check failed")
            return
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        return
    
    # Test 2: Check cache status
    try:
        response = requests.get(f"{base_url}/api/cache-status")
        if response.status_code == 200:
            cache_status = response.json()
            print(f"✅ Cache status: {cache_status['memory_entries']} memory entries, {cache_status['disk_entries']} disk entries")
        else:
            print("❌ Cache status check failed")
    except Exception as e:
        print(f"❌ Cache status error: {e}")
    
    # Test 3: Check DICOM support status
    try:
        response = requests.get(f"{base_url}/api/dicom_support_status")
        if response.status_code == 200:
            support_status = response.json()
            print(f"✅ DICOM support: OpenCV={support_status['opencv_available']}, pylibjpeg={support_status['pylibjpeg_available']}")
        else:
            print("❌ DICOM support status check failed")
    except Exception as e:
        print(f"❌ DICOM support error: {e}")
    
    print("\n" + "=" * 50)
    print("OCT Frame Extraction Test Complete!")
    print("\nTo test with actual E2E files:")
    print("1. Upload an E2E file through the web interface")
    print("2. Check the file tree for 'Original OCT Frames' and 'Flattened OCT Frames' branches")
    print("3. Click on individual frames to load them")
    print("4. Use the frame slider to navigate between OCT frames")
    print("\nBackend endpoints available:")
    print("- GET /api/get_e2e_tree_data?dicom_file_path=<crc>")
    print("- GET /api/get_e2e_oct_frames?dicom_file_path=<crc>&eye=<left|right>")
    print("- GET /api/view_e2e_oct_frame?dicom_file_path=<crc>&eye=<left|right>&frame_number=<n>&frame_type=<original|flattened>")

if __name__ == "__main__":
    test_oct_frame_extraction() 