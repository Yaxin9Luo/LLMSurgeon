from typing import List, Optional
import inspect

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


NEUTRAL_PROMPTS = [
    "Continue the passage:",
    "Write the next paragraph:",
    "The following text discusses:",
    "Consider the following passage:",
    "In this article, we explore:",
    "Here is a paragraph about:",
    "An overview of the topic:",
    "A short description:",
    "The statement is:",
    "This section covers:",
    "We now present:",
    "Background:",
    "Introduction:",
    "Main text:",
    "Details:",
    "Key ideas:",
    "Context:",
    "Notes:",
    "Title:",
    "Abstract:",
    "Overview:",
    "Observation:",
    "Conclusion:",
    "The text continues:",
    "Opening lines:",
    "Body paragraph:",
    "Further explanation:",
    "Elaboration:",
    "Clarification:",
    "Summary:",
    "Rationale:",
    "Motivation:",
    "Discussion:",
    " ",
    "\n",
    "\n\n",
] * 10  

INSTRUCTIONAL_PROMPTS = [
    "Explain the following concept in detail:",
    "Provide step-by-step instructions for the task:",
    "Describe how to solve the following problem:",
    "Give a detailed walkthrough of the process:",
    "Outline the main steps required to complete this:",
    "Explain the rationale behind the following approach:",
    "Provide a beginner-friendly explanation of:",
    "Describe the correct procedure for:",
    "Write a short tutorial on:",
    "Give a checklist of steps for:",
    "Explain how to evaluate the following:",
    "Provide a high-level step-by-step guide for:",
    "Describe the algorithm used to:",
    "Explain how to implement the following method:",
    "Provide a set of instructions for reproducing:",
    "Describe how to verify the result of:",
    "Explain how to debug the following issue:",
    "Write an explanation suitable for a new learner about:",
    "Provide a structured guide for understanding:",
    "Explain the setup process for:",
    "Describe how to configure the following system:",
    "Explain how to prepare the necessary inputs for:",
    "Provide a sequence of actions to perform:",
    "Describe how to correctly interpret the output of:",
    "Explain how to troubleshoot common errors in:",
    "Write an instructional passage about:",
    "Explain how to adapt this procedure for a new setting:",
    "Provide a numbered list of steps for:",
    "Describe the workflow for completing:",
    "Explain the decision process involved in:",
    "Provide usage instructions for the following:",
    "Explain how to systematically approach:",
    "Describe how to check assumptions when dealing with:",
    "Write a concise how-to for:",
    "Explain how to organize the tasks involved in:",
    "Provide a detailed guideline on how to handle:",
] * 10  # ~360 prompts


EXPOSITORY_PROMPTS = [
    "Discuss the following topic in an expository manner:",
    "Provide background information about:",
    "Describe the broader context surrounding:",
    "Give an objective overview of:",
    "Explain the historical development of:",
    "Describe the key characteristics of:",
    "Provide an analytical summary of:",
    "Explain the main factors that influence:",
    "Describe the structure and components of:",
    "Give a neutral explanation of:",
    "Provide an overview suitable for a general audience of:",
    "Explain the relationship between the following ideas:",
    "Describe how this concept fits into a larger framework:",
    "Explain the main challenges associated with:",
    "Describe the typical use cases for:",
    "Provide a clear exposition of:",
    "Explain the fundamental principles behind:",
    "Describe the current state of knowledge about:",
    "Summarize the main arguments related to:",
    "Explain the theoretical basis for:",
    "Describe the main dimensions or aspects of:",
    "Provide a detailed description of how this works:",
    "Explain how different perspectives approach:",
    "Describe the main categories involved in:",
    "Explain the key variables that affect:",
    "Provide a balanced explanation of the pros and cons of:",
    "Describe the main stages or phases of:",
    "Explain the core ideas that define:",
    "Describe the typical patterns observed in:",
    "Explain the main terminology related to:",
    "Provide an expository explanation of the concept of:",
    "Describe the implications of the following for practice:",
    "Explain how this phenomenon is usually studied:",
    "Describe how this idea has evolved over time:",
    "Explain the reasoning that supports the following view:",
    "Provide a concept-focused explanation of:",
] * 10  # ~360 prompts


CONVERSATIONAL_PROMPTS = [
    "Let's talk about the following topic:",
    "What do you think about this idea:",
    "Can we have a conversation about:",
    "I'm curious how you would describe:",
    "Could you help me understand the following:",
    "Let's walk through this together:",
    "How would you explain this to a friend:",
    "What would you say if someone asked you about:",
    "Let's chat informally about:",
    "Can you give me your take on:",
    "How would you describe this in simple terms:",
    "If we were discussing this over coffee, how would you explain:",
    "What stands out to you about:",
    "How would you break this down for someone new to:",
    "What kind of questions would you ask about:",
    "Can you respond as if you're answering a casual question about:",
    "How might a conversation about this topic start:",
    "What are some key points you'd mention when talking about:",
    "If someone was confused about this, how would you clarify:",
    "Let's go step by step and talk through:",
    "What would a back-and-forth discussion about this focus on:",
    "How would you respond if a friend said they don't understand:",
    "What are the most important things to mention in a conversation about:",
    "How would you summarize this aloud in a few sentences:",
    "If you were explaining this live, what would you say about:",
    "What would you highlight first when talking about:",
    "How could you gently introduce someone to:",
    "What questions might people commonly have about:",
    "How could you clear up common misunderstandings about:",
    "How would you phrase this explanation in a relaxed tone about:",
    "What would you emphasize in a conversational explanation of:",
    "If someone interrupted you to ask 'why?', how would you answer about:",
    "How might you reassure someone who is confused about:",
    "What simple example would you use when talking about:",
    "How would you start a friendly explanation of:",
    "What follow-up points would naturally come up when discussing:",
] * 10  # ~360 prompts


CODING_PROMPTS = [
    "Write Python code to implement the following function:",
    "Provide a minimal working example in Python that:",
    "Write a Python script that accomplishes the following task:",
    "Implement a Python class that:",
    "Write a short Python program to demonstrate:",
    "Provide a Python function with docstring that:",
    "Write Python code that reads input, processes it, and:",
    "Show a Python example using standard library functions to:",
    "Write Python code that handles errors correctly for:",
    "Implement a Python solution using lists and dictionaries to:",
    "Write a Python function that takes parameters and returns:",
    "Provide Python code that simulates the following behavior:",
    "Write a Python example that demonstrates file I/O for:",
    "Implement a simple command-line Python program that:",
    "Write Python code to parse and process structured data for:",
    "Provide a Python implementation of the described algorithm:",
    "Write a Python function that logs intermediate steps while:",
    "Show Python code that makes the logic clear for:",
    "Write Python code using list comprehensions to:",
    "Provide a Python example using classes and methods to:",
    "Write Python code that uses recursion to:",
    "Implement a Python function with type hints that:",
    "Write a Python example demonstrating unit tests for:",
    "Provide Python code that uses context managers to:",
    "Write a Python snippet that constructs and manipulates:",
    "Implement a Python function that validates inputs for:",
    "Write Python code that efficiently processes a large list to:",
    "Provide a Python example that uses itertools to:",
    "Write Python code that uses dictionaries to map and transform:",
    "Implement a Python program that prints intermediate debug output while:",
    "Write Python code that uses generators to:",
    "Provide a Python example that shows how to structure a small module for:",
    "Write Python code that uses dataclasses to represent:",
    "Implement a Python function that measures performance while:",
    "Write Python code that uses numpy to:",
    "Provide a Python example that demonstrates the algorithm for:",
] * 10  # ~360 prompts


CATEGORY_STYLE_PROMPTS = {
    # CommonCrawl (raw-ish web; include light HTML/boilerplate cues)
    "commoncrawl": [
        "<html><head><title>Article</title></head><body><article><h1>",
        "<div class='content'><p>",
        "By Staff Writer — Updated:",
        "Breaking: ",
        "Related posts:",
        "Privacy Policy — ",
        "Contact us:",
    ],

    # C4 (cleaned web articles; no HTML, news/blog tone, headings)
    "c4": [
        "Title:",
        "Subtitle:",
        "Introduction",
        "Overview",
        "Key takeaways:",
        "Further reading:",
        "Author’s note:",
    ],

    # GitHub (code/files/readmes across languages)
    "github": [
        "```python\n# Filename: utils.py\n\"\"\"Module description:\"\"\"\n",
        "```javascript\n// file: app.js\n// Description:\n",
        "```cpp\n// utils.hpp\n//",
        "```go\n// Package docs:\n",
        "README.md\n# Project Title\n",
        "LICENSE\nMIT License\n",
        "def solve():\n    \"\"\"",
    ],

    # Wikipedia (markup, sections, neutral tone)
    "wikipedia": [
        "== Introduction ==",
        "{{Short description|}}",
        "== History ==",
        "== See also ==",
        "== References ==\n* ",
        "=== Background ===",
        "Infobox:",
    ],

    # Books (long-form narrative/exposition)
    "books": [
        "Chapter 1\n",
        "Prologue\n",
        "It was a",
        "The morning of",
        "He said,",
        "She remembered",
        "The following chapter explores",
    ],

    # arXiv (LaTeX/math/paper structure)
    "arxiv": [
        "\\documentclass{article}\n\\usepackage{amsmath}\n\\title{",
        "\\begin{abstract}\n",
        "\\section{Introduction}\n",
        "We prove the following theorem.",
        "Definition.",
        "Lemma.",
        "Proof.",
    ],

    # StackExchange (Q/A format, tags, accepted-answer vibe)
    "stackexchange": [
        "Title: How do I\n\nQuestion:\n",
        "Q:\nA:",
        "Problem statement:\n",
        "Accepted answer:\n",
        "Steps to reproduce:",
        "Tags: [python] [arrays]",
        "Comment:",
    ],
}


def load_hf_model(model_name: str, revision: Optional[str] = None):
    # Try loading tokenizer; fall back to slow tokenizer if fast conversion fails
    # (e.g., LLM360/Amber models use LLaMA tokenizer that can have conversion issues)
    try:
        tok = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="left",
            revision=revision,
        )
    except ValueError as e:
        if "Converting from SentencePiece" in str(e) or "slow->fast" in str(e):
            print(f"Fast tokenizer conversion failed, falling back to slow tokenizer: {e}")
            tok = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                padding_side="left",
                revision=revision,
                use_fast=False,
            )
        else:
            raise

    # Ensure a valid pad token for batch padding with causal LMs (e.g., LLaMA)
    if tok.pad_token is None:
        if tok.eos_token is not None:
            tok.pad_token = tok.eos_token
        else:
            tok.add_special_tokens({"pad_token": "<|pad|>"})
            tok.pad_token = "<|pad|>"
    tok.padding_side = "left"
    tok.truncation_side = "left"
    
    # Try loading on a single GPU first, fallback to device_map="auto" if OOM
    try:
        mdl = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="cuda:0",
            trust_remote_code=True,
            revision=revision,
        )
    except torch.cuda.OutOfMemoryError:
        print("OOM on single GPU, using device_map='auto'")
        mdl = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            revision=revision,
        )
    
    # If we added tokens, resize embeddings
    if hasattr(mdl, "get_input_embeddings") and hasattr(mdl, "resize_token_embeddings"):
        vocab_size = mdl.get_input_embeddings().weight.shape[0]
        if tok.vocab_size != vocab_size:
            mdl.resize_token_embeddings(len(tok))
    # Propagate pad_token_id to model config
    try:
        mdl.config.pad_token_id = tok.pad_token_id
    except Exception:
        pass
    try:
        mdl.generation_config.pad_token_id = tok.pad_token_id
    except Exception:
        pass
    if tok.eos_token_id is not None:
        try:
            mdl.generation_config.eos_token_id = tok.eos_token_id
        except Exception:
            try:
                mdl.config.eos_token_id = tok.eos_token_id
            except Exception:
                pass
    mdl.eval()
    return mdl, tok


def generate_texts(
    model_name: str,
    prompts: Optional[List[str]] = None,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_p: float = 0.9,
    batch_size: int = 4,
    revision: Optional[str] = None,
    seed: Optional[int] = None,
) -> List[str]:
    prompts = prompts or NEUTRAL_PROMPTS
    if not prompts:
        raise ValueError("No prompts provided for generation")
    model, tok = load_hf_model(model_name, revision=revision)

    out_texts: List[str] = []
    
    # Determine the device strategy
    if hasattr(model, 'hf_device_map') and model.hf_device_map:
        # Model is distributed across devices, need to move inputs to first device
        candidate_devices = []
        for dev in model.hf_device_map.values():
            if isinstance(dev, torch.device):
                if dev.type != "meta":
                    candidate_devices.append(dev)
            elif isinstance(dev, str):
                if dev.startswith("cuda") or dev.startswith("cpu"):
                    candidate_devices.append(torch.device(dev))
            elif isinstance(dev, int):
                candidate_devices.append(torch.device(f"cuda:{dev}"))
        if not candidate_devices:
            candidate_devices.append(torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"))
        # Prefer CUDA if available in the map
        def _device_key(dev: torch.device) -> tuple:
            if dev.type == "cuda":
                # Prefer lower index CUDA devices first
                return (0, dev.index or 0)
            if dev.type == "cpu":
                return (1, 0)
            return (2, 0)

        candidate_devices.sort(key=_device_key)
        first_device = candidate_devices[0]
        use_device_map = True
    else:
        # Model is on a single device
        first_device = next(model.parameters()).device
        use_device_map = False
    
    # Optional torch.Generator for reproducible sampling across devices
    gen_obj = None
    if seed is not None:
        try:
            if first_device.type == "cuda":
                gen_obj = torch.Generator(device=first_device)
            else:
                gen_obj = torch.Generator()
            gen_obj.manual_seed(int(seed))
        except Exception:
            gen_obj = None

    # Detect whether this transformers version supports `generator` kwarg
    supports_generator = False
    try:
        sig = inspect.signature(model.generate)
        supports_generator = "generator" in sig.parameters
    except Exception:
        supports_generator = False

    for i in tqdm(range(0, len(prompts), batch_size), total=(len(prompts)+batch_size-1)//batch_size, desc="Generating"):
        batch = prompts[i : i + batch_size]
        enc = tok(batch, return_tensors="pt", padding=True)
        # Remove token_type_ids if present, as some models (like OLMo) don't use them
        if "token_type_ids" in enc:
            enc.pop("token_type_ids")
        
        # Move inputs to the appropriate device
        if use_device_map:
            # For distributed models, move to the primary device (Accelerate handles sharding)
            enc = enc.to(first_device)
        else:
            # For single-device models, move to model's device
            enc = enc.to(first_device)
            
        with torch.no_grad():
            # If `generator` kwarg is not supported, set and restore global RNG around generate
            use_local_rng = (seed is not None) and (not supports_generator)
            if use_local_rng:
                batch_seed = int(seed) + i
                cpu_state = torch.get_rng_state()
                cuda_states = None
                if torch.cuda.is_available():
                    try:
                        cuda_states = torch.cuda.get_rng_state_all()
                    except Exception:
                        cuda_states = None
                torch.manual_seed(batch_seed)
                if torch.cuda.is_available():
                    try:
                        torch.cuda.manual_seed_all(batch_seed)
                    except Exception:
                        pass
            try:
                gen_kwargs = dict(
                    enc,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=(getattr(model.config, "pad_token_id", None) or getattr(tok, "pad_token_id", None)),
                )
                if supports_generator and gen_obj is not None:
                    gen_kwargs["generator"] = gen_obj
                gen = model.generate(**gen_kwargs)
            except Exception as e:
                # Known issue: some OLMo/hf_olmo revisions error on None past_key_values.
                # Also handle older transformers that do not support `generator` kwarg.
                print(
                    f"Generation error: {e}\n"
                    "Retrying with use_cache=False and without generator if unsupported. "
                    "Consider pinning --hf_revision to a known-good checkpoint."
                )
                gen_kwargs = dict(
                    enc,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=(getattr(model.config, "pad_token_id", None) or getattr(tok, "pad_token_id", None)),
                    use_cache=False,
                )
                # Only pass generator if supported
                if supports_generator and gen_obj is not None:
                    gen_kwargs["generator"] = gen_obj
                try:
                    gen = model.generate(**gen_kwargs)
                except Exception as e2:
                    # Final fallback: remove generator entirely
                    if "generator" in gen_kwargs:
                        gen_kwargs.pop("generator", None)
                    gen = model.generate(**gen_kwargs)
            finally:
                if use_local_rng:
                    try:
                        torch.set_rng_state(cpu_state)
                        if cuda_states is not None:
                            # Restore per-device CUDA RNG states if available
                            try:
                                torch.cuda.set_rng_state_all(cuda_states)
                            except Exception:
                                # Best-effort: restore on current device only
                                try:
                                    torch.cuda.set_rng_state(cuda_states[0])
                                except Exception:
                                    pass
                    except Exception:
                        pass
        texts = tok.batch_decode(gen, skip_special_tokens=True)
        # Strip the original prompt to keep only generations after the prompt
        for p, full in zip(batch, texts):
            if full.startswith(p):
                out_texts.append(full[len(p) :].strip())
            else:
                out_texts.append(full)
    return out_texts
