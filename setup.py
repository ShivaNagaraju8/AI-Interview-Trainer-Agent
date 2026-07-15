#!/usr/bin/env python3
"""
Quick-start setup script for AI Interview Trainer Agent.
Run: python setup.py
"""
import os
import sys
import subprocess
import shutil


def check_python():
    if sys.version_info < (3, 9):
        print("❌ Python 3.9+ required. Current:", sys.version)
        sys.exit(1)
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}")


def install_deps():
    print("\n📦 Installing dependencies...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "--quiet"],
        capture_output=False
    )
    if result.returncode != 0:
        print("❌ Dependency installation failed. Check requirements.txt.")
        sys.exit(1)
    print("✅ Dependencies installed.")


def setup_env():
    if not os.path.exists(".env"):
        shutil.copy(".env.example", ".env")
        print("\n📝 Created .env from .env.example")
        print("   ⚠️  Edit .env and add your IBM Cloud API Key and Project ID before running.")
    else:
        print("\n✅ .env already exists.")


def create_dirs():
    for d in ["reports", ".chroma_db", "knowledge_base/technical",
              "knowledge_base/hr", "knowledge_base/behavioral", "knowledge_base/company"]:
        os.makedirs(d, exist_ok=True)
    print("✅ Directories ready.")


def verify_env():
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("WATSONX_API_KEY", "")
    proj_id = os.getenv("WATSONX_PROJECT_ID", "")

    if not api_key or api_key == "your_ibm_cloud_api_key_here":
        print("\n⚠️  WATSONX_API_KEY not configured — app will run in Demo Mode.")
        print("   Get your key at: https://cloud.ibm.com/iam/apikeys")
    else:
        print("✅ IBM watsonx.ai API key found.")

    if not proj_id or proj_id == "your_watsonx_project_id_here":
        print("⚠️  WATSONX_PROJECT_ID not configured.")
        print("   Get it at: https://dataplatform.cloud.ibm.com/projects/")
    else:
        print("✅ watsonx.ai Project ID found.")


if __name__ == "__main__":
    print("=" * 60)
    print("  AI Interview Trainer Agent — Setup")
    print("  Powered by IBM watsonx.ai + IBM Granite")
    print("=" * 60)

    check_python()
    create_dirs()
    setup_env()
    install_deps()
    verify_env()

    print("\n" + "=" * 60)
    print("✅ Setup complete!")
    print("\n▶  Run the app:")
    print("   streamlit run app.py")
    print("\n📖 Documentation: README.md")
    print("=" * 60)
