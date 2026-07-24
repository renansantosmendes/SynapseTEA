"""Test OpenAI connection and embedding functionality."""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def test_openai_connection():
    """Test OpenAI connection and embedding generation.

    Returns:
        True if all tests pass, False otherwise.
    """
    print("=" * 70)
    print("OpenAI Connection Test")
    print("=" * 70)
    
    # Check environment variables
    print("\n[1/4] Checking environment variables...")
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    
    if not openai_api_key:
        print("❌ OPENAI_API_KEY not found in environment variables")
        print("   Please add OPENAI_API_KEY to your .env file")
        return False
    
    if openai_api_key.startswith("sk-"):
        print(f"✅ OPENAI_API_KEY found: {openai_api_key[:10]}...")
    else:
        print("⚠️  OPENAI_API_KEY exists but may be invalid (should start with 'sk-')")
    
    # Try importing langchain_openai
    print("\n[2/4] Checking langchain_openai package...")
    try:
        from langchain_openai import OpenAIEmbeddings
        print("✅ langchain_openai package imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import langchain_openai: {e}")
        print("   Install with: pip install langchain-openai")
        return False
    
    # Try initializing OpenAI embeddings
    print("\n[3/4] Initializing OpenAI embeddings client...")
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        print("✅ OpenAI embeddings client initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize embeddings: {e}")
        print("   Check if your API key is valid and has proper permissions")
        return False
    
    # Try generating an embedding
    print("\n[4/4] Testing embedding generation with sample text...")
    try:
        test_text = "This is a test sentence to validate OpenAI embeddings."
        embeddings_result = embeddings.embed_documents([test_text])
        
        if not embeddings_result or len(embeddings_result) == 0:
            print("❌ Embedding result is empty")
            return False
        
        vec = embeddings_result[0]
        
        # Check dimension
        expected_dim = 512  # text-embedding-3-small has 512 dimensions
        if len(vec) != expected_dim:
            print(f"⚠️  Unexpected embedding dimension: {len(vec)} (expected {expected_dim})")
        
        # Check for non-zero values
        zero_count = sum(1 for v in vec if v == 0.0)
        non_zero_count = len(vec) - zero_count
        
        if non_zero_count == 0:
            print("❌ Embedding vector contains only zeros!")
            print("   This indicates the embedding generation failed")
            return False
        
        print(f"✅ Embedding generated successfully")
        print(f"   Dimension: {len(vec)}")
        print(f"   Non-zero values: {non_zero_count}/{len(vec)}")
        print(f"   Sample values: {vec[:5]}")
    
    except Exception as e:
        print(f"❌ Failed to generate embedding: {e}")
        print("   Check your API key and rate limits")
        return False
    
    # All tests passed
    print("\n" + "=" * 70)
    print("✅ All tests passed! OpenAI connection is working correctly.")
    print("=" * 70)
    return True


if __name__ == "__main__":
    success = test_openai_connection()
    sys.exit(0 if success else 1)
