#!/usr/bin/env python3
"""
CLI Chatbot Interface for RAG System
Interactive command-line interface for querying the RAG knowledge base
"""

import os
import sys
import argparse
from rag_chatbot import RAGChatbot, create_chatbot


def print_welcome():
    """Print welcome message"""
    print("\n" + "="*60)
    print("🤖 RAG CHATBOT - Interactive Knowledge Base Assistant")
    print("="*60)
    print("\nCommands:")
    print("  - Ask questions about your documents")
    print("  - Type 'help' for available commands")
    print("  - Type 'clear' to clear conversation history")
    print("  - Type 'history' to view recent conversation")
    print("  - Type 'export' to save conversation to file")
    print("  - Type 'quit' or 'exit' to exit")
    print("="*60 + "\n")


def print_help():
    """Print help message"""
    print("\n📋 Available Commands:")
    print("  help          - Show this help message")
    print("  clear         - Clear conversation history")
    print("  history       - Show recent conversation history")
    print("  export        - Export conversation to JSON file")
    print("  stats         - Show conversation statistics")
    print("  quit/exit     - Exit the chatbot")
    print("  <question>    - Ask a question about your documents\n")


def print_stats(chatbot: RAGChatbot):
    """Print conversation statistics"""
    history = chatbot.get_conversation_history()
    if not history:
        print("📊 No conversation history yet")
        return
    
    total_queries = len(history)
    total_time = sum(m.get("metadata", {}).get("total_time", 0) for m in history)
    avg_time = total_time / total_queries if total_queries > 0 else 0
    successful = sum(1 for m in history if m.get("metadata", {}).get("success", False))
    
    print(f"\n📊 Conversation Statistics:")
    print(f"  Total Queries: {total_queries}")
    print(f"  Successful: {successful}/{total_queries}")
    print(f"  Average Response Time: {avg_time:.2f}s")
    print(f"  Total Time: {total_time:.2f}s\n")


def print_history(chatbot: RAGChatbot, num_items: int = 5):
    """Print recent conversation history"""
    history = chatbot.get_conversation_history()
    if not history:
        print("📜 No conversation history yet\n")
        return
    
    print(f"\n📜 Recent Conversation History (last {min(num_items, len(history))} items):")
    print("-" * 60)
    for i, item in enumerate(history[-num_items:], 1):
        query = item.get("query", "")
        response = item.get("response", "")
        timestamp = item.get("timestamp", "")
        metadata = item.get("metadata", {})
        total_time = metadata.get("total_time", 0)
        
        print(f"\n[{i}] {timestamp[:19]}")
        print(f"Q: {query[:80]}{'...' if len(query) > 80 else ''}")
        print(f"A: {response[:80]}{'...' if len(response) > 80 else ''}")
        print(f"   ⏱️ {total_time:.2f}s | 📦 {metadata.get('num_chunks', 0)} chunks")
    print("-" * 60 + "\n")


def interactive_chat(chatbot: RAGChatbot, verbose: bool = False):
    """Run interactive chat loop"""
    print_welcome()
    
    while True:
        try:
            # Get user input
            user_input = input("❓ You: ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye! Thanks for using RAG Chatbot!")
                break
            
            elif user_input.lower() == 'help':
                print_help()
                continue
            
            elif user_input.lower() == 'clear':
                chatbot.clear_context()
                print("✅ Conversation history cleared\n")
                continue
            
            elif user_input.lower() == 'history':
                print_history(chatbot)
                continue
            
            elif user_input.lower() == 'stats':
                print_stats(chatbot)
                continue
            
            elif user_input.lower() == 'export':
                filename = f"conversation_{chatbot.context.messages[0]['timestamp'][:10] if chatbot.context.messages else 'export'}.json"
                chatbot.export_conversation(filename)
                print(f"✅ Conversation exported to {filename}\n")
                continue
            
            # Process query
            print("\n🤔 Thinking...")
            response, metadata = chatbot.query(user_input, verbose=verbose)
            
            # Display response
            print(f"\n🤖 Assistant: {response}\n")
            
            if verbose:
                print(f"   📊 Stats: {metadata['total_time']:.2f}s total | "
                      f"{metadata['retrieval_time']:.2f}s retrieval | "
                      f"{metadata['llm_time']:.2f}s LLM | "
                      f"{metadata['num_chunks']} chunks\n")
        
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye! Thanks for using RAG Chatbot!")
            break
        except EOFError:
            print("\n\n👋 Goodbye! Thanks for using RAG Chatbot!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")
            continue


def main():
    parser = argparse.ArgumentParser(description="RAG Chatbot CLI Interface")
    parser.add_argument(
        "--index-dir",
        required=True,
        help="Path to the FAISS index directory"
    )
    parser.add_argument(
        "--model",
        default="llama3.2:1b",
        help="Chat model to use (default: llama3.2:1b)"
    )
    parser.add_argument(
        "--no-context",
        action="store_true",
        help="Disable conversation context"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed performance metrics"
    )
    args = parser.parse_args()
    
    # Validate index directory
    if not os.path.exists(args.index_dir):
        print(f"❌ Index directory not found: {args.index_dir}")
        return 1
    
    # Check for index files
    index_file = os.path.join(args.index_dir, "index.faiss")
    meta_file = os.path.join(args.index_dir, "meta.json")
    
    if not os.path.exists(index_file) or not os.path.exists(meta_file):
        print(f"❌ Index files not found in: {args.index_dir}")
        print(f"   Expected: {index_file} and {meta_file}")
        return 1
    
    # Create chatbot
    try:
        chatbot = create_chatbot(
            index_dir=args.index_dir,
            chat_model=args.model,
            enable_context=not args.no_context
        )
        print(f"✅ Chatbot initialized with model: {args.model}")
        print(f"✅ Index directory: {args.index_dir}")
        if not args.no_context:
            print(f"✅ Conversation context: Enabled")
        else:
            print(f"ℹ️  Conversation context: Disabled")
    except Exception as e:
        print(f"❌ Failed to initialize chatbot: {e}")
        return 1
    
    # Start interactive chat
    try:
        interactive_chat(chatbot, verbose=args.verbose)
    except Exception as e:
        print(f"❌ Chatbot error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
