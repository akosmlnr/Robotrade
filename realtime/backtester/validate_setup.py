"""
Backtester Setup Validation Script
Validates that all components are properly configured and dependencies are available
"""

import os
import sys
import logging
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def validate_environment():
    """Validate environment setup"""
    print("🔍 Validating Environment Setup...")
    
    issues = []
    warnings = []
    
    # Check Python version
    if sys.version_info < (3, 7):
        issues.append("Python 3.7+ is required")
    else:
        print(f"✅ Python version: {sys.version}")
    
    # Check required environment variables
    if not os.getenv('POLYGON_API_KEY'):
        issues.append("POLYGON_API_KEY environment variable not set")
    else:
        print("✅ Polygon API key found")
    
    return issues, warnings

def validate_dependencies():
    """Validate required dependencies"""
    print("\n🔍 Validating Dependencies...")
    
    issues = []
    warnings = []
    
    required_packages = [
        'pandas', 'numpy', 'requests', 'tensorflow', 'sklearn'
    ]
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} available")
        except ImportError:
            issues.append(f"{package} not installed")
    
    return issues, warnings

def validate_directory_structure():
    """Validate directory structure and files"""
    print("\n🔍 Validating Directory Structure...")
    
    issues = []
    warnings = []
    
    # Check if we're in the right directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    
    # Check for required directories
    required_dirs = [
        'models',  # Should contain LSTM models
        'storage',  # Data storage
        'data'  # Data fetcher
    ]
    
    for dir_name in required_dirs:
        dir_path = os.path.join(parent_dir, dir_name)
        if os.path.exists(dir_path):
            print(f"✅ {dir_name}/ directory found")
        else:
            warnings.append(f"{dir_name}/ directory not found")
    
    # Check for models directory
    models_dir = os.path.join(parent_dir, 'lstms')
    if os.path.exists(models_dir):
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.keras')]
        if model_files:
            print(f"✅ Found {len(model_files)} model files in lstms/")
        else:
            warnings.append("No .keras model files found in lstms/")
    else:
        warnings.append("lstms/ directory not found")
    
    return issues, warnings

def validate_backtester_files():
    """Validate backtester files"""
    print("\n🔍 Validating Backtester Files...")
    
    issues = []
    warnings = []
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    required_files = [
        'backtester.py',
        'historical_data_fetcher.py',
        'prediction_simulator.py',
        'example_usage.py',
        'README.md'
    ]
    
    for filename in required_files:
        file_path = os.path.join(current_dir, filename)
        if os.path.exists(file_path):
            print(f"✅ {filename} found")
        else:
            issues.append(f"{filename} not found")
    
    return issues, warnings

def validate_imports():
    """Validate that imports work correctly"""
    print("\n🔍 Validating Imports...")
    
    issues = []
    warnings = []
    
    try:
        from historical_data_fetcher import HistoricalDataFetcher
        print("✅ HistoricalDataFetcher import successful")
    except Exception as e:
        issues.append(f"HistoricalDataFetcher import failed: {e}")
    
    try:
        from prediction_simulator import PredictionSimulator
        print("✅ PredictionSimulator import successful")
    except Exception as e:
        issues.append(f"PredictionSimulator import failed: {e}")
    
    try:
        from backtester import Backtester
        print("✅ Backtester import successful")
    except Exception as e:
        issues.append(f"Backtester import failed: {e}")
    
    return issues, warnings

def main():
    """Main validation function"""
    print("🚀 Backtester Setup Validation")
    print("=" * 50)
    
    all_issues = []
    all_warnings = []
    
    # Run all validations
    issues, warnings = validate_environment()
    all_issues.extend(issues)
    all_warnings.extend(warnings)
    
    issues, warnings = validate_dependencies()
    all_issues.extend(issues)
    all_warnings.extend(warnings)
    
    issues, warnings = validate_directory_structure()
    all_issues.extend(issues)
    all_warnings.extend(warnings)
    
    issues, warnings = validate_backtester_files()
    all_issues.extend(issues)
    all_warnings.extend(warnings)
    
    issues, warnings = validate_imports()
    all_issues.extend(issues)
    all_warnings.extend(warnings)
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 VALIDATION SUMMARY")
    print("=" * 50)
    
    if all_issues:
        print(f"❌ {len(all_issues)} Critical Issues Found:")
        for i, issue in enumerate(all_issues, 1):
            print(f"  {i}. {issue}")
    
    if all_warnings:
        print(f"\n⚠️  {len(all_warnings)} Warnings:")
        for i, warning in enumerate(all_warnings, 1):
            print(f"  {i}. {warning}")
    
    if not all_issues and not all_warnings:
        print("🎉 All validations passed! Backtester is ready to use.")
        return True
    elif not all_issues:
        print("\n✅ No critical issues found. Backtester should work with warnings.")
        return True
    else:
        print(f"\n❌ {len(all_issues)} critical issues must be resolved before using the backtester.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
