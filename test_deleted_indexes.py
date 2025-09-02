#!/usr/bin/env python3
"""
Test script to verify that deleted_indexes from manifest.json are properly loaded
"""
import json
import os
import sys

def test_manifest_loading(test_catalog_path=None):
    """Test if manifest.json with deleted_indexes is properly handled"""
    
    # Test data path (modify this to your actual test data path)
    if test_catalog_path is None:
        if len(sys.argv) > 1:
            test_catalog_path = sys.argv[1]
        else:
            print("Usage: python test_deleted_indexes.py <path_to_catalog_folder>")
            print("\nExample:")
            print("  python test_deleted_indexes.py ./data/mydata")
            return False
    
    if not os.path.exists(test_catalog_path):
        print(f"Error: Path does not exist: {test_catalog_path}")
        return False
    
    manifest_path = os.path.join(test_catalog_path, "manifest.json")
    if not os.path.exists(manifest_path):
        print(f"Error: manifest.json not found at: {manifest_path}")
        return False
    
    print(f"\nReading manifest.json from: {manifest_path}")
    
    try:
        with open(manifest_path, 'r') as mf:
            manifest_lines = mf.readlines()
            
            print(f"Manifest has {len(manifest_lines)} lines")
            
            if len(manifest_lines) >= 5:
                # Line 5 contains catalog info
                catalog_info = json.loads(manifest_lines[4])
                
                if "paths" in catalog_info:
                    print(f"Found catalog paths: {catalog_info['paths']}")
                
                if "deleted_indexes" in catalog_info:
                    deleted_indexes = catalog_info["deleted_indexes"]
                    print(f"\nFound {len(deleted_indexes)} deleted indexes:")
                    print(f"Deleted indexes: {deleted_indexes}")
                    
                    # Check if there are catalog files to verify
                    if "paths" in catalog_info:
                        for catalog_file in catalog_info["paths"]:
                            catalog_path = os.path.join(test_catalog_path, catalog_file)
                            if os.path.exists(catalog_path):
                                print(f"\nChecking catalog file: {catalog_file}")
                                
                                # Count entries and check deleted ones
                                total_entries = 0
                                deleted_entries = 0
                                with open(catalog_path, 'r') as cf:
                                    for line in cf:
                                        entry = json.loads(line)
                                        entry_index = entry.get('_index', None)
                                        total_entries += 1
                                        if entry_index in deleted_indexes:
                                            deleted_entries += 1
                                            print(f"  - Entry {entry_index} is marked as deleted")
                                
                                print(f"  Total entries: {total_entries}, Deleted: {deleted_entries}")
                else:
                    print("No deleted_indexes found in manifest.json")
                    
                return True
                
            else:
                print("Manifest file doesn't have enough lines (expected at least 5)")
                return False
                
    except Exception as e:
        print(f"Error reading manifest: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("=" * 60)
    print("Testing deleted_indexes loading from manifest.json")
    print("=" * 60)
    
    success = test_manifest_loading()
    
    if success:
        print("\n[OK] Test completed successfully")
        print("\nTo fully test the fix:")
        print("1. Run main.py")
        print("2. Load the catalog folder with File -> Open Folder")
        print("3. Check if deleted annotations are properly marked")
        print("4. Verify that deleted annotations are excluded from export")
    else:
        print("\n[FAILED] Test failed")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()