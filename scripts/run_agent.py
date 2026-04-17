#-*- coding: utf-8 -*-
"""
@author: Md Rezwanul Haque
"""

"""
Agent Inference Script for SLM-RL-Agents

This script provides multiple ways to interact with a trained SLM agent. It supports
three modes of operation: interactive chat (for testing and exploration), batch inference
(for processing multiple prompts efficiently), and server mode (for production deployment
via REST API).

Usage:
    # Interactive chat mode
    python scripts/run_agent.py --model_path "./outputs/ppo/final" --mode interactive
    
    # Batch inference mode
    python scripts/run_agent.py --model_path "./outputs/ppo/final" --mode batch \
        --input_file prompts.txt --output_file responses.txt
    
    # Server mode (REST API)
    python scripts/run_agent.py --model_path "./outputs/ppo/final" --mode server \
        --host 0.0.0.0 --port 8000

Each mode serves different use cases. Interactive mode lets you explore the model's
capabilities through conversation. Batch mode is efficient for processing datasets.
Server mode enables integration with other applications through HTTP.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)


def run_interactive(model_path: str, **kwargs):
    """Run the agent in interactive chat mode. This mode is perfect for exploring the
    model's capabilities, testing specific prompts, and understanding how the model
    responds to different inputs."""
    
    from src.agent import SLMAgent
    
    logger.info(f"Loading agent from {model_path}")
    agent = SLMAgent.from_pretrained(
        model_path,
        load_in_4bit=kwargs.get("load_in_4bit", False),
    )
    
    print("\n" + "=" * 60)
    print("SLM-RL-Agents Interactive Mode")
    print("=" * 60)
    print("Type your message and press Enter to chat.")
    print("Commands: /quit (exit), /clear (reset conversation), /info (model info)")
    print("=" * 60 + "\n")
    
    # Start conversation mode for multi-turn interactions
    agent.start_conversation()
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.lower() == "/quit":
                print("Goodbye!")
                break
            elif user_input.lower() == "/clear":
                agent.clear_conversation()
                print("[Conversation cleared]")
                continue
            elif user_input.lower() == "/info":
                info = agent.get_model_info()
                print(f"\nModel Info:")
                for key, value in info.items():
                    print(f"  {key}: {value}")
                print()
                continue
            
            # Generate response
            response = agent.generate(
                user_input,
                max_new_tokens=kwargs.get("max_new_tokens", 256),
                temperature=kwargs.get("temperature", 0.7),
            )
            
            print(f"Assistant: {response}\n")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            logger.error(f"Error: {e}")
            continue


def run_batch(model_path: str, input_file: str, output_file: str, **kwargs):
    """Run the agent in batch inference mode. This mode efficiently processes a file
    of prompts and writes the responses to an output file. It's useful for evaluating
    the model on datasets or generating outputs for analysis."""
    
    from src.agent import SLMAgent
    
    logger.info(f"Loading agent from {model_path}")
    agent = SLMAgent.from_pretrained(
        model_path,
        load_in_4bit=kwargs.get("load_in_4bit", False),
    )
    
    # Load prompts
    logger.info(f"Loading prompts from {input_file}")
    prompts = []
    
    if input_file.endswith(".json"):
        with open(input_file) as f:
            data = json.load(f)
            if isinstance(data, list):
                prompts = [item.get("prompt", item) if isinstance(item, dict) else item for item in data]
            else:
                prompts = [data.get("prompt", str(data))]
    elif input_file.endswith(".jsonl"):
        with open(input_file) as f:
            for line in f:
                item = json.loads(line)
                prompts.append(item.get("prompt", str(item)))
    else:
        # Plain text file, one prompt per line
        with open(input_file) as f:
            prompts = [line.strip() for line in f if line.strip()]
    
    logger.info(f"Processing {len(prompts)} prompts...")
    
    # Generate responses
    responses = agent.generate_batch(
        prompts,
        batch_size=kwargs.get("batch_size", 8),
        max_new_tokens=kwargs.get("max_new_tokens", 256),
        temperature=kwargs.get("temperature", 0.7),
    )
    
    # Save results
    logger.info(f"Saving responses to {output_file}")
    
    if output_file.endswith(".json"):
        results = [{"prompt": p, "response": r} for p, r in zip(prompts, responses)]
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
    elif output_file.endswith(".jsonl"):
        with open(output_file, "w") as f:
            for p, r in zip(prompts, responses):
                f.write(json.dumps({"prompt": p, "response": r}) + "\n")
    else:
        # Plain text file
        with open(output_file, "w") as f:
            for p, r in zip(prompts, responses):
                f.write(f"Prompt: {p}\n")
                f.write(f"Response: {r}\n")
                f.write("-" * 40 + "\n")
    
    logger.info(f"Batch inference complete! Processed {len(prompts)} prompts.")


def run_server(model_path: str, host: str = "0.0.0.0", port: int = 8000, **kwargs):
    """Run the agent as a REST API server. This mode deploys the model as a web service
    that other applications can query over HTTP. It's the recommended way to use the
    model in production environments."""
    
    try:
        from src.agent import start_server
        
        logger.info(f"Starting server on {host}:{port}")
        logger.info("Endpoints:")
        logger.info(f"  POST http://{host}:{port}/generate - Generate text")
        logger.info(f"  POST http://{host}:{port}/chat - Multi-turn chat")
        logger.info(f"  GET  http://{host}:{port}/health - Health check")
        
        start_server(
            model_path,
            host=host,
            port=port,
        )
    except ImportError:
        logger.error("FastAPI not installed. Install with: pip install fastapi uvicorn")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Run SLM-RL-Agents for inference")
    
    # Required arguments
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained model")
    parser.add_argument("--mode", type=str, default="interactive",
                        choices=["interactive", "batch", "server"],
                        help="Inference mode")
    
    # Batch mode arguments
    parser.add_argument("--input_file", type=str, default=None,
                        help="Input file for batch mode")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Output file for batch mode")
    
    # Server mode arguments
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Host for server mode")
    parser.add_argument("--port", type=int, default=8000,
                        help="Port for server mode")
    
    # Generation arguments
    parser.add_argument("--max_new_tokens", type=int, default=256,
                        help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for batch mode")
    parser.add_argument("--load_in_4bit", action="store_true",
                        help="Load model in 4-bit quantization")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.mode == "batch":
        if not args.input_file or not args.output_file:
            parser.error("Batch mode requires --input_file and --output_file")
    
    # Run the selected mode
    kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "batch_size": args.batch_size,
        "load_in_4bit": args.load_in_4bit,
    }
    
    if args.mode == "interactive":
        run_interactive(args.model_path, **kwargs)
    elif args.mode == "batch":
        run_batch(args.model_path, args.input_file, args.output_file, **kwargs)
    elif args.mode == "server":
        run_server(args.model_path, args.host, args.port, **kwargs)


if __name__ == "__main__":
    main()
