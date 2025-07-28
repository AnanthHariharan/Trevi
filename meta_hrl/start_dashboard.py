#!/usr/bin/env python3
"""
Quick start script for the Meta-HRL dashboard.
This script starts the FastAPI backend server.
"""

import sys
import os
import subprocess
import time

# Add the parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def start_backend():
    """Start the FastAPI backend server."""
    backend_dir = os.path.join(os.path.dirname(__file__), 'frontend', 'backend')
    
    print("Starting Meta-HRL Dashboard Backend...")
    print(f"Backend directory: {backend_dir}")
    
    # Check if uvicorn is available
    try:
        subprocess.run(['uvicorn', '--version'], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: uvicorn not found. Please install it:")
        print("pip install 'uvicorn[standard]'")
        return False
    
    # Start the backend server
    try:
        os.chdir(backend_dir)
        print("Starting FastAPI server at http://localhost:8000")
        print("API documentation available at http://localhost:8000/docs")
        print("Press Ctrl+C to stop the server")
        
        subprocess.run([
            'uvicorn', 'main:app',
            '--host', '0.0.0.0',
            '--port', '8000',
            '--reload'
        ])
        
    except KeyboardInterrupt:
        print("\nShutting down server...")
        return True
    except Exception as e:
        print(f"Error starting server: {e}")
        return False

def main():
    """Main function."""
    print("=" * 60)
    print("Meta-HRL Dashboard Startup")
    print("=" * 60)
    
    success = start_backend()
    
    if success:
        print("\nServer stopped successfully.")
    else:
        print("\nServer failed to start.")
        print("\nTroubleshooting:")
        print("1. Make sure you have installed the requirements:")
        print("   cd frontend/backend && pip install -r requirements.txt")
        print("2. Check if port 8000 is already in use")
        print("3. Run the demo first: python demo_usage.py")

if __name__ == "__main__":
    main()