#!/usr/bin/env python3
"""
Demonstration: How to Use RAG Tools in SynapseTEA

This script shows practical examples of using the RAG tools for:
- Document search
- Context retrieval for RAG
- Document information gathering
"""

import json
from data.rag_agent import PineconeRAGTools

def pretty_print(data_str: str, title: str = None):
    """Pretty print JSON or text output."""
    if title:
        print(f"\n{'='*70}")
        print(f"  {title}")
        print(f"{'='*70}\n")
    
    try:
        data = json.loads(data_str)
        print(json.dumps(data, indent=2, ensure_ascii=False))
    except json.JSONDecodeError:
        print(data_str)
    
    print()


def demo_1_basic_search():
    """Demo 1: Basic document search."""
    print("\n" + "="*70)
    print("DEMO 1: Basic Document Search")
    print("="*70)
    
    rag = PineconeRAGTools()
    
    # Example 1: Search for speech therapy assessment
    query = "avaliação fonoaudiológica infantil"
    print(f"\nQuery: '{query}'")
    print("Finding the top 3 most relevant documents...")
    
    results = rag.search_documents(query, top_k=3)
    data = json.loads(results)
    
    print(f"\nFound {data['total_results']} results\n")
    for i, result in enumerate(data['results'], 1):
        print(f"{i}. {result['file']}")
        print(f"   Page {result['page']}, Chunk {result['chunk_idx']}")
        print(f"   Similarity Score: {result['score']:.4f}")
        print(f"   Preview: {result['text'][:100]}...\n")


def demo_2_context_retrieval():
    """Demo 2: Retrieve context for RAG."""
    print("\n" + "="*70)
    print("DEMO 2: Context Retrieval for RAG")
    print("="*70)
    
    rag = PineconeRAGTools()
    
    # Example: Get context about occupational therapy
    query = "terapia ocupacional desenvolvimento motor"
    print(f"\nQuery: '{query}'")
    print("Retrieving relevant context chunks with citations...")
    
    context = rag.retrieve_context(query, top_k=2)
    print("\nFormatted Context:")
    print("-"*70)
    print(context)
    print("-"*70)


def demo_3_document_info():
    """Demo 3: Get document information."""
    print("\n" + "="*70)
    print("DEMO 3: Document Information")
    print("="*70)
    
    rag = PineconeRAGTools()
    
    # Get info about a specific document
    filename = "Avaliação Antonio Damasceno Fonoaudiologia.pdf"
    print(f"\nDocument: {filename}")
    print("Gathering document statistics...")
    
    info = rag.get_document_info(filename)
    data = json.loads(info)
    
    print(f"\nChunk Count: {data['total_chunks']}")
    print(f"Pages: {', '.join(str(p) for p in data['pages'])}")
    print(f"Sample Chunks (first 5):")
    for chunk in data['chunks'][:5]:
        print(f"  - Page {chunk['page']}, Chunk {chunk['chunk_idx']}")


def demo_4_rag_workflow():
    """Demo 4: Complete RAG workflow."""
    print("\n" + "="*70)
    print("DEMO 4: Complete RAG Workflow")
    print("="*70)
    
    rag = PineconeRAGTools()
    
    # User's question
    user_question = "Qual é o diagnóstico de António Damasceno?"
    
    print(f"\nUser Question: '{user_question}'")
    print("\n" + "─"*70)
    print("STEP 1: Search for relevant documents")
    print("─"*70)
    
    # Step 1: Search documents
    search_results = rag.search_documents(user_question, top_k=5)
    search_data = json.loads(search_results)
    
    print(f"Found {len(search_data['results'])} relevant documents:")
    for i, result in enumerate(search_data['results'], 1):
        print(f"  {i}. {result['file']} (score: {result['score']:.3f})")
    
    print("\n" + "─"*70)
    print("STEP 2: Retrieve context for generating response")
    print("─"*70)
    
    # Step 2: Get formatted context
    context = rag.retrieve_context(user_question, top_k=3)
    print("\nContext for RAG:")
    print(context[:500] + "...")  # Show first 500 chars
    
    print("\n" + "─"*70)
    print("STEP 3: Use context with LLM to answer question")
    print("─"*70)
    
    print("\nIn your application, you would now:")
    print("  1. Take the retrieved context")
    print("  2. Format it with the user's question")
    print("  3. Send to OpenAI/LLM for answer generation")
    print("\nExample prompt:")
    prompt_template = f"""Context from documents:
{context[:300]}...

Question: {user_question}

Answer based on the context above:"""
    print(prompt_template)


def demo_5_multi_query():
    """Demo 5: Multiple queries on different topics."""
    print("\n" + "="*70)
    print("DEMO 5: Multi-Topic Search")
    print("="*70)
    
    rag = PineconeRAGTools()
    
    queries = [
        "comportamento social e interação",
        "linguagem receptiva expressiva",
        "habilidades motoras finas",
        "plano de intervenção terapêutica",
    ]
    
    for query in queries:
        print(f"\nSearching: '{query}'")
        results = rag.search_documents(query, top_k=1)
        data = json.loads(results)
        
        if data['results']:
            top_result = data['results'][0]
            print(f"  ✓ Top match: {top_result['file']} (score: {top_result['score']:.3f})")
        else:
            print(f"  ✗ No results found")


def main():
    """Run all demonstrations."""
    print("\n" * 2)
    print("="*70)
    print("SynapseTEA RAG Tools - Practical Demonstrations".center(70))
    print("="*70)
    
    try:
        # Run all demos
        demo_1_basic_search()
        demo_2_context_retrieval()
        demo_3_document_info()
        demo_4_rag_workflow()
        demo_5_multi_query()
        
        print("\n" + "="*70)
        print(f"\n[SUCCESS] All demonstrations completed successfully!")
        print("="*70)
        print("\nNext Steps:")
        print("1. Review the examples above")
        print("2. Read RAG_TOOLS_USAGE.md for detailed documentation")
        print("3. Integrate the tools into your application")
        print("4. Check data/test_agent_simple.py for more examples")
        print()
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        raise


if __name__ == "__main__":
    main()
