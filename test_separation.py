#!/usr/bin/env python3
"""
Test script to validate E2E file separation between fundus and OCT images.
This script helps debug the issue where fundus images appear in the original OCT frames.
"""

import requests
import json
import sys
import os

def test_e2e_separation(file_path):
    """Test E2E file separation by uploading and validating the file."""
    
    base_url = "http://localhost:8000"
    
    print(f"=== Testing E2E Separation for: {file_path} ===")
    
    # Step 1: Upload the E2E file
    print("\n1. Uploading E2E file...")
    try:
        with open(file_path, 'rb') as f:
            files = {'file': (os.path.basename(file_path), f, 'application/octet-stream')}
            response = requests.post(f"{base_url}/api/upload", files=files)
            
        if response.status_code != 200:
            print(f"❌ Upload failed: {response.status_code}")
            print(f"Error: {response.text}")
            return False
            
        upload_result = response.json()
        print(f"✅ Upload successful")
        print(f"File key: {upload_result.get('dicom_file_path')}")
        print(f"File type: {upload_result.get('file_type')}")
        
        file_key = upload_result.get('dicom_file_path')
        
    except Exception as e:
        print(f"❌ Upload error: {str(e)}")
        return False
    
    # Step 2: Get enhanced tree data
    print("\n2. Getting enhanced tree data...")
    try:
        response = requests.get(f"{base_url}/api/get_e2e_tree_data", 
                              params={'dicom_file_path': file_key})
        
        if response.status_code != 200:
            print(f"❌ Failed to get tree data: {response.status_code}")
            return False
            
        tree_data = response.json()
        print(f"✅ Tree data retrieved")
        
        # Analyze the tree data
        left_eye = tree_data.get('left_eye', {})
        right_eye = tree_data.get('right_eye', {})
        
        print(f"\nLeft Eye Data:")
        print(f"  Fundus/DICOM: {len(left_eye.get('dicom', []))}")
        print(f"  Original OCT: {len(left_eye.get('original_oct', []))}")
        print(f"  Flattened OCT: {len(left_eye.get('flattened_oct', []))}")
        
        print(f"\nRight Eye Data:")
        print(f"  Fundus/DICOM: {len(right_eye.get('dicom', []))}")
        print(f"  Original OCT: {len(right_eye.get('original_oct', []))}")
        print(f"  Flattened OCT: {len(right_eye.get('flattened_oct', []))}")
        
    except Exception as e:
        print(f"❌ Error getting tree data: {str(e)}")
        return False
    
    # Step 3: Validate separation
    print("\n3. Validating separation...")
    try:
        response = requests.get(f"{base_url}/api/validate_e2e_separation", 
                              params={'dicom_file_path': file_key})
        
        if response.status_code != 200:
            print(f"❌ Validation failed: {response.status_code}")
            return False
            
        validation_result = response.json()
        print(f"✅ Validation completed")
        
        # Check for misclassified images
        misclassified = validation_result.get('misclassified_images', [])
        if misclassified:
            print(f"\n❌ MISCLASSIFIED IMAGES FOUND:")
            for item in misclassified:
                print(f"  - {item['key']}: {item['issue']}")
        else:
            print(f"\n✅ No misclassified images found")
        
        # Show detailed breakdown
        fundus_images = validation_result.get('fundus_images', {})
        oct_original = validation_result.get('oct_original_images', {})
        
        print(f"\nDetailed Breakdown:")
        print(f"  Fundus images: {fundus_images.get('count', 0)}")
        if fundus_images.get('keys'):
            print(f"    Keys: {fundus_images['keys']}")
        
        print(f"  OCT original images: {oct_original.get('count', 0)}")
        if oct_original.get('keys'):
            print(f"    Keys: {oct_original['keys']}")
        
        # Check for fundus images in OCT
        fundus_in_oct = [key for key in oct_original.get('keys', []) if 'fundus' in key]
        if fundus_in_oct:
            print(f"\n❌ FUNDUS IMAGES FOUND IN OCT FRAMES:")
            for key in fundus_in_oct:
                print(f"  - {key}")
        
        # Check for OCT images in fundus
        oct_in_fundus = [key for key in fundus_images.get('keys', []) if 'oct_original' in key]
        if oct_in_fundus:
            print(f"\n❌ OCT IMAGES FOUND IN FUNDUS:")
            for key in oct_in_fundus:
                print(f"  - {key}")
        
        validation_passed = validation_result.get('validation_passed', False)
        print(f"\nOverall Validation: {'✅ PASSED' if validation_passed else '❌ FAILED'}")
        
        return validation_passed
        
    except Exception as e:
        print(f"❌ Validation error: {str(e)}")
        return False

def main():
    if len(sys.argv) != 2:
        print("Usage: python test_separation.py <path_to_e2e_file>")
        print("Example: python test_separation.py ./test.e2e")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        sys.exit(1)
    
    if not file_path.lower().endswith('.e2e'):
        print(f"❌ File is not an E2E file: {file_path}")
        sys.exit(1)
    
    # Test the separation
    success = test_e2e_separation(file_path)
    
    if success:
        print(f"\n🎉 Separation test completed successfully!")
    else:
        print(f"\n⚠️ Separation test found issues. Check the details above.")
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 