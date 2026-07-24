"""Interactive test loop for the RAG Agent.

Run this script to start an interactive session where you can ask questions
to the agent and see it retrieve and analyze documents from Pinecone.
"""

import logging
import sys
from typing import Optional

from rag_agent import create_rag_agent, PineconeRAGTools
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def print_header(text: str) -> None:
    """Print a formatted header."""
    print(f"\n{'=' * 70}")
    print(f"  {text}")
    print(f"{'=' * 70}\n")


def print_divider() -> None:
    """Print a divider line."""
    print("\n" + "-" * 70 + "\n")


def run_agent_loop(
    index_name: str = "synapse-tea-index",
    embedding_model: str = "text-embedding-3-large",
    model: str = "gpt-4o-mini",
) -> None:
    """Run the interactive agent loop.

    Args:
        index_name: Pinecone index name.
        embedding_model: Embedding model to use.
        model: LLM model to use for the agent.
    """
    print_header("SynapseTEA RAG Agent - Interactive Test Loop")

    # Initialize agent
    print("Initializing RAG Agent...")
    try:
        agent = create_rag_agent(index_name, embedding_model, model)
        print("✓ Agent initialized successfully\n")
    except Exception as e:
        print(f"✗ Failed to initialize agent: {e}")
        logger.exception("Agent initialization failed")
        return

    # Initialize RAG tools for direct access
    try:
        rag_tools = PineconeRAGTools(index_name, embedding_model)
        print("✓ RAG tools initialized successfully\n")
    except Exception as e:
        print(f"✗ Failed to initialize RAG tools: {e}")
        logger.exception("RAG tools initialization failed")
        return

    print("Available commands:")
    print("  - Type your question to ask the agent")
    print("  - Type 'search <query>' to search documents directly")
    print("  - Type 'context <query>' to retrieve context directly")
    print("  - Type 'info <filename>' to get document info")
    print("  - Type 'help' for help")
    print("  - Type 'quit' or 'exit' to quit\n")

    # Main loop
    query_count = 0
    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()

            if not user_input:
                continue

            # Check for special commands
            if user_input.lower() in ["quit", "exit", "q"]:
                print("\nGoodbye!")
                break

            if user_input.lower() == "help":
                print_divider()
                print("Commands:")
                print("  search <query> - Search for documents")
                print("  context <query> - Get context from documents")
                print("  info <filename> - Get document information")
                print("  help - Show this help message")
                print("  quit - Exit the program")
                continue

            # Handle direct tool calls
            if user_input.lower().startswith("search "):
                query = user_input[7:].strip()
                if query:
                    print_divider()
                    print(f"Searching for: {query}\n")
                    results = rag_tools.search_documents(query, top_k=5)
                    print(results)
                    print_divider()
                continue

            if user_input.lower().startswith("context "):
                query = user_input[8:].strip()
                if query:
                    print_divider()
                    print(f"Retrieving context for: {query}\n")
                    context = rag_tools.retrieve_context(query, top_k=3)
                    print(context)
                    print_divider()
                continue

            if user_input.lower().startswith("info "):
                filename = user_input[5:].strip()
                if filename:
                    print_divider()
                    print(f"Getting info for: {filename}\n")
                    info = rag_tools.get_document_info(filename)
                    print(info)
                    print_divider()
                continue

            # Regular question to agent
            query_count += 1
            print_divider()
            print(f"Processing your question (Query #{query_count})...\n")

            try:
                # Call the agent if available
                if hasattr(agent, "run"):
                    response = agent.run(user_input)
                    print(f"Agent: {response}")
                else:
                    print("Agent interface not available (no run method).")
                print_divider()

            except Exception as e:
                print(f"Error calling agent: {e}")
                logger.exception("Agent execution failed")
                # Fallback: attempt to answer using the direct tools as a simple heuristic
                try:
                    results = rag_tools.search_documents(user_input, top_k=3)
                    print(results)
                except Exception:
                    pass
                print_divider()

        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Goodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            logger.exception("Unexpected error in main loop")
            continue


def run_demo_queries() -> None:
    """Run a demo with pre-defined queries."""
    print_header("SynapseTEA RAG Agent - Demo Mode")

    demo_queries = [
        "Quais são as avaliações de fonoaudiologia disponíveis?",
        "O paciente Antonio Damasceno foi avaliado em que mês de 2025?",
        "Quais documentos contêm informações sobre terapia ocupacional?",
    ]

    print("Initializing RAG Agent for demo...\n")
    try:
        agent = create_rag_agent()
        print("✓ Agent initialized\n")
    except Exception as e:
        print(f"✗ Failed to initialize agent: {e}")
        return

    for i, query in enumerate(demo_queries, 1):
        print(f"\nDemo Query {i}/{len(demo_queries)}")
        print_divider()
        print(f"Question: {query}\n")

        try:
            response = agent.run(query)
            print(f"Response: {response}")
        except Exception as e:
            print(f"Error: {e}")
            logger.exception(f"Demo query {i} failed")

        print_divider()


def main() -> None:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Test the SynapseTEA RAG Agent",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run demo mode with pre-defined queries",
    )
    parser.add_argument(
        "--index",
        default="synapse-tea-index",
        help="Pinecone index name (default: synapse-tea-index)",
    )
    parser.add_argument(
        "--model",
        default="text-embedding-3-large",
        help="Embedding model (default: text-embedding-3-large)",
    )
    parser.add_argument(
        "--llm",
        default="gpt-4o-mini",
        help="LLM model (default: gpt-4o-mini)",
    )

    args = parser.parse_args()

    if args.demo:
        run_demo_queries()
    else:
        run_agent_loop(args.index, args.model, args.llm)


if __name__ == "__main__":
    main()
