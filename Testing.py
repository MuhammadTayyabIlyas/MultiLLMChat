"""
================================================================================
LLM PROVIDER TESTING SCRIPT - Testing.py
================================================================================

PURPOSE:
    This script tests multiple LLM (Large Language Model) providers with a 
    single prompt to compare their responses, performance, and capabilities.

    It serves as a unified testing framework for:
    - Chat models (GPT-4, Claude, Gemini, etc.)
    - Reasoning models (DeepSeek, Grok, Kimi)
    - Search APIs (Perplexity)
    - Specialized models (Mistral, Qwen)
    - Voice synthesis (ElevenLabs TTS)

USAGE:
    1. Set up your .env file with API keys (see .env.example)
    2. Install requirements: pip install -r requirements.txt
    3. Run: python Testing.py

    The script will test all configured providers and display their responses.
    Additionally, it will convert the best-scoring quote into an MP3 file
    using the ElevenLabs API.

CONFIGURATION:
    All providers are configured through the .env file:
    - API keys at the top of the file
    - Provider table with endpoints and models

    To add a new provider:
    1. Add API key to .env
    2. Add entry to provider table
    3. Add call function below
    4. Add to callers dictionary in run_all()

TROUBLESHOOTING:
    - 401 Error: API key is missing or invalid
    - 404 Error: Model name or endpoint is wrong
    - Timeout: Provider is slow or unreachable
    - KeyError: Provider not found in provider table

DEPENDENCIES:
    See REQUIREMENTS tuple below or requirements.txt file

================================================================================
"""

from __future__ import annotations # Enables postponed evaluation of type annotations

import os # For interacting with the operating system, like environment variables
import re # For regular expressions, used in MP3 filename generation
from pathlib import Path # For object-oriented filesystem paths
from typing import Callable, Dict, List, Tuple # For type hinting

# Provider SDK imports
# These are the Python SDKs for the various LLM providers being tested.
import anthropic # For Anthropic's Claude models
import google.generativeai as genai # For Google's Gemini models
import openai # For OpenAI models and OpenAI-compatible APIs (DeepSeek, Kimi, Qwen)
import requests # For making HTTP requests directly (used for Perplexity and ElevenLabs)
from groq import Groq # For Groq's optimized LLMs
from mistralai import Mistral # For Mistral AI models


# ================================================================================
# CONFIGURATION & SETUP
# ================================================================================

# Define file paths relative to the script's location for portability.
ENV_FILE = Path(__file__).with_name(".env") # Path to the environment variables file
REQUIREMENTS_FILE = Path(__file__).with_name("requirements.txt") # Path to the project's dependency list

# The default text prompt used for testing all LLM providers.
DEFAULT_PROMPT = "Tell me a joke"

# Maximum time (in seconds) to wait for an API response. This prevents the script
# from hanging indefinitely if a provider is slow or unreachable.
TIMEOUT = 60

# Configuration for saving generated MP3 audio files.
OUTPUT_DIR = Path("output_AUDIO") # Directory where MP3s will be saved
OUTPUT_FILENAME_PREFIX = "output_" # Prefix for MP3 filenames (e.g., output_001.mp3)
OUTPUT_EXTENSION = ".mp3" # File extension for audio outputs

# Regular expression pattern to parse existing MP3 filenames,
# used to determine the next sequential filename.
OUTPUT_FILENAME_PATTERN = re.compile(
    rf"^{OUTPUT_FILENAME_PREFIX}(\d+){re.escape(OUTPUT_EXTENSION)}$",
    re.IGNORECASE, # Case-insensitive matching for filenames
)

# List of required Python packages and their minimum versions.
# This tuple is used to generate/update the requirements.txt file, ensuring
# all necessary dependencies can be installed easily via `pip install -r requirements.txt`.
REQUIREMENTS = (
    "requests>=2.32.0",  # Used for direct HTTP calls to various APIs
    "openai>=1.50.0",  # SDK for OpenAI and OpenAI-compatible APIs
    "anthropic>=0.74.0",  # SDK for Anthropic's models
    "google-generativeai>=0.7.2",  # SDK for Google Gemini models
    "groq>=0.11.0",  # SDK for Groq models
    "mistralai>=1.1.0",  # SDK for Mistral AI models
    "python-dotenv>=1.0.1",  # for loading environment variables from .env
)


# ================================================================================
# PROVIDER TABLE PARSING
# ================================================================================

def parse_provider_table(path: Path) -> Dict[Tuple[str, str], Dict[str, List[str]]]:
    """
    Reads a CSV-style provider configuration table from a .env file.
    
    This function is designed to extract details about various LLM providers,
    including their API endpoints and supported models, directly from the .env file.
    This allows for flexible configuration without modifying the script's code.

    Expected format within the .env file (case-insensitive header):
        Provider,Endpoint,Function,Model(s)
        OpenAI,https://api.openai.com/v1/chat/completions,chat completions,gpt-4o
        Kimi (Moonshot),https://api.moonshot.ai/v1/chat/completions,chat completions,kimi-k2-0711-preview

    Args:
        path: Path to the .env file containing the provider table.
        
    Returns:
        A dictionary mapping (provider_name, function_name) tuples to another
        dictionary containing "endpoint" (str) and "models" (List[str]).
        Returns an empty dictionary if the .env file does not exist.
    """
    specs: Dict[Tuple[str, str], Dict[str, List[str]]] = {}
    if not path.exists():
        return specs

    lines = path.read_text(encoding="utf-8").splitlines()
    capture = False # Flag to indicate when to start parsing the table
    for raw in lines:
        stripped = raw.strip()
        if not stripped: # Skip empty lines
            if capture:
                break # Stop capturing if an empty line is encountered after starting
            continue
        # Detect the header line to start capturing provider data
        if stripped.lower().startswith("provider,endpoint"):
            capture = True
            continue
        if not capture:
            continue
        # Ignore comments and extract parts of the CSV row
        row = stripped.split(":contentReference")[0].strip()
        parts = [p.strip() for p in row.split(",")]
        if len(parts) < 4: # Ensure the row has at least 4 columns (Provider, Endpoint, Function, Models)
            continue
        provider, endpoint, function = parts[:3]
        models = [m.strip(" .") for m in parts[3:]] # Extract models, cleaning up whitespace/punctuation
        specs[(provider.lower(), function.lower())] = {
            "endpoint": endpoint,
            "models": [m for m in models if m], # Filter out any empty model strings
        }
    return specs


# Global dictionary containing specifications for all configured LLM providers.
# This is loaded once when the module is imported based on the .env file.
PROVIDER_SPECS = parse_provider_table(ENV_FILE)


# ================================================================================
# UTILITY FUNCTIONS
# ================================================================================

def ensure_requirements_file(path: Path = REQUIREMENTS_FILE) -> None:
    """
    Creates or updates the `requirements.txt` file with the packages listed
    in the global `REQUIREMENTS` tuple.

    This function ensures that all project dependencies are properly documented
    and can be easily installed by running `pip install -r requirements.txt`.
    It only writes to the file if the content differs from the desired state.
    
    Args:
        path: The filesystem path where `requirements.txt` should be located.
    """
    desired = "\n".join(REQUIREMENTS) + "\n"
    # Check if the file exists and its content is already up-to-date to avoid unnecessary writes.
    if path.exists() and path.read_text(encoding="utf-8") == desired:
        return
    path.write_text(desired, encoding="utf-8")


def load_env(path: Path = ENV_FILE) -> None:
    """
    Loads environment variables from a specified .env file into `os.environ`.

    This makes API keys and other sensitive configuration details available
    to the script without hardcoding them, improving security and portability.
    If a variable is already set in the system environment, it takes precedence.
    
    Args:
        path: The filesystem path to the .env file.
        
    Raises:
        FileNotFoundError: If the .env file does not exist at the specified path.
    """
    if not path.exists():
        raise FileNotFoundError(f"Could not find {path}. Please create one with your API keys.")

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        # Skip empty lines, lines starting with '#', and lines without an '='.
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1) # Split only on the first '='
        key = key.strip()
        value = value.strip().strip("'\"") # Remove surrounding quotes from the value
        # Set the environment variable if it's not already set.
        os.environ.setdefault(key, value)


def require_env(key: str) -> str:
    """
    Retrieves the value of a required environment variable.

    This function is used for fetching critical configuration values (like API keys)
    and provides a clear error message if the variable is not found, preventing
    the script from proceeding with incomplete information.
    
    Args:
        key: The name of the environment variable to retrieve.
        
    Returns:
        The string value of the environment variable.
        
    Raises:
        RuntimeError: If the specified environment variable is not set.
    """
    value = os.environ.get(key)
    if not value:
        raise RuntimeError(f"Environment variable '{key}' is missing. Please set it in your .env file or system environment.")
    return value


def require_any(*keys: str) -> str:
    """
    Retrieves the value of the first available environment variable from a list of keys.

    This is useful for APIs that might accept different environment variable names
    for the same key (e.g., `KIMI_API_KEY` or `MOONSHOT_API_KEY`). The function
    tries each key in order and returns the first one it finds.
    
    Args:
        *keys: Variable-length argument list of environment variable names to check.
        
    Returns:
        The string value of the first found environment variable.
        
    Raises:
        RuntimeError: If none of the specified environment variables are set.
    """
    for key in keys:
        value = os.environ.get(key)
        if value:
            return value
    raise RuntimeError(
        f"None of the required environment variables were found. Please provide one of these: {', '.join(keys)}."
    )


def get_endpoint(provider: str, function: str = "chat completions") -> str:
    """
    Looks up the API endpoint URL for a given provider and function type.

    This function retrieves the specific API endpoint from the `PROVIDER_SPECS`
    dictionary, which is loaded from the `.env` file. This allows the script
    to dynamically target different API services based on configuration.
    
    Args:
        provider: The name of the LLM provider (e.g., "OpenAI", "Kimi (Moonshot)").
        function: The specific function provided by the API (e.g., "chat completions", "tts").
                  Defaults to "chat completions" if not specified.
        
    Returns:
        The URL of the API endpoint as a string.
        
    Raises:
        RuntimeError: If the endpoint for the specified provider and function
                      is not found in the `PROVIDER_SPECS`.
    """
    key = (provider.lower(), function.lower())
    try:
        return PROVIDER_SPECS[key]["endpoint"]
    except KeyError as exc:
        raise RuntimeError(
            f"Endpoint for provider '{provider}' and function '{function}' "
            "was not found in .env. Please check your provider table configuration."
        ) from exc


def get_models(provider: str, function: str) -> List[str]:
    """
    Retrieves a list of configured model names for a specific provider and function.

    This allows the script to use different models based on the `.env` configuration,
    providing flexibility for testing various model capabilities.
    
    Args:
        provider: The name of the LLM provider.
        function: The specific function (e.g., "chat completions").
        
    Returns:
        A list of strings, where each string is a model name. Returns an empty
        list if no models are configured for the given provider/function pair.
    """
    spec = PROVIDER_SPECS.get((provider.lower(), function.lower()))
    if not spec:
        return []
    return spec.get("models", [])


def clean_quote(text: str) -> str:
    """
    Normalizes and sanitizes a raw text response from an LLM into a single-line quote.

    This function helps in extracting a concise, clean quote for further processing
    (e.g., scoring or text-to-speech conversion), ensuring consistency across diverse
    LLM outputs. It removes extra whitespace, newlines, and leading/trailing quotes.
    
    Args:
        text: The raw string response received from an LLM.
        
    Returns:
        A cleaned, single-line string quote. Returns an empty string if the input
        is empty or indicates an error.
    """
    if not text:
        return ""
    if text.lower().startswith("error"): # Skip error messages
        return ""
    # Take the first line and remove leading/trailing whitespace and quotes.
    first_line = text.strip().splitlines()[0].strip()
    first_line = first_line.strip(" \"'")
    # Collapse multiple internal whitespace characters into single spaces for readability.
    normalized = " ".join(first_line.split())
    return normalized


def score_quote(quote: str) -> float:
    """
    Scores a given quote based on its brevity, relevance to "time", and single-sentence structure.

    This heuristic scoring function helps in selecting the "best" quote from multiple
    LLM responses, prioritizing conciseness and thematic relevance.
    
    Args:
        quote: The candidate quote string to be scored.
        
    Returns:
        A floating-point score, where higher values indicate a "better" quote.
        Returns negative infinity for empty quotes.
    """
    if not quote:
        return float("-inf") # Penalize empty quotes heavily

    word_count = len(quote.split())
    # Count punctuation marks to estimate sentence count.
    punctuation_marks = sum(quote.count(mark) for mark in ".!?")
    # Penalize quotes that seem to contain more than one sentence.
    extra_sentences = max(punctuation_marks - 1, 0)

    score = 100.0
    score -= abs(word_count - 12) * 1.5 # Reward quotes around 12 words, penalize deviations.
    score -= extra_sentences * 8 # Strong penalty for multi-sentence outputs.
    score -= max(len(quote) - 140, 0) * 0.25 # Discourage very long quotes.
    if "time" in quote.lower():
        score += 5 # Bonus for explicitly mentioning "time" (case-insensitive).

    return score


def select_best_quote(responses: Dict[str, str]) -> Tuple[str, str]:
    """
    Selects the highest-scoring quote from a dictionary of provider responses.

    This function iterates through all collected LLM responses, cleans each,
    scores it using `score_quote`, and identifies the provider and quote
    that achieved the highest score.
    
    Args:
        responses: A dictionary where keys are provider labels (e.g., "OpenAI")
                   and values are their raw text responses.
        
    Returns:
        A tuple containing the label of the best provider and the cleaned,
        highest-scoring quote string.
        
    Raises:
        RuntimeError: If no valid one-line quotes could be extracted and scored
                      from any of the provider responses.
    """
    best_provider = ""
    best_quote = ""
    best_score = float("-inf")

    for provider, raw in responses.items():
        quote = clean_quote(raw)
        if not quote:
            continue
        score = score_quote(quote)
        if score > best_score:
            best_provider = provider
            best_quote = quote
            best_score = score

    if not best_quote:
        raise RuntimeError("No valid one-line quotes were returned by the providers that could be selected.")
    return best_provider, best_quote


def convert_quote_to_speech(quote: str, output_file: Path) -> str:
    """
    Sends a given text quote to the ElevenLabs Text-to-Speech API,
    converts it to speech, and saves the resulting audio as an MP3 file.

    This function directly interacts with the ElevenLabs HTTP API using the
    `requests` library, bypassing the official SDK to ensure direct control
    over the request and response handling, and to specifically save the output.
    
    Args:
        quote: The text string to be converted into speech.
        output_file: The full path, including filename and extension (e.g., "audio.mp3"),
                     where the synthesized audio will be saved.
        
    Returns:
        A human-readable status message indicating the success and location
        of the saved MP3 file.
        
    Raises:
        RuntimeError: If the ElevenLabs API request fails (e.g., due to
                      incorrect API key, network issues, or server errors).
    """
    # Retrieve ElevenLabs API key from environment variables.
    api_key = require_env("ELEVENLABS_API_KEY")
    # Get voice ID from environment or use a default.
    voice_id = os.environ.get("ELEVENLABS_VOICE_ID", "JBFqnCBsd6RMkjVDRZzb") # Default to Rachel's voice ID
    # Get model ID from environment or use a default.
    model_id = os.environ.get("ELEVENLABS_MODEL_ID", "eleven_monolingual_v1")
    
    # Construct the API endpoint URL for text-to-speech.
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {
        "Accept": "audio/mpeg", # Request MP3 audio
        "Content-Type": "application/json", # Indicate JSON payload
        "xi-api-key": api_key, # ElevenLabs API key for authentication
    }
    payload = {
        "model_id": model_id, # Specify the TTS model to use
        "text": quote, # The text to synthesize
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.7}, # Optional voice customization
    }

    # Send the POST request to the ElevenLabs API.
    response = requests.post(url, headers=headers, json=payload, timeout=TIMEOUT)
    try:
        # Raise an HTTPError for bad responses (4xx or 5xx status codes).
        response.raise_for_status()
    except requests.HTTPError as exc:
        # Extract and include more detail from the API response body if available.
        detail = response.text.strip()
        raise RuntimeError(f"ElevenLabs API request failed ({response.status_code}): {detail}") from exc

    # Ensure the output directory exists before writing the file.
    output_file.parent.mkdir(parents=True, exist_ok=True)
    # Write the binary audio content to the specified MP3 file.
    output_file.write_bytes(response.content)
    
    return f"✅ ElevenLabs: MP3 saved to {output_file}"


def next_output_mp3_path(
    directory: Path = OUTPUT_DIR,
    prefix: str = OUTPUT_FILENAME_PREFIX,
    extension: str = OUTPUT_EXTENSION,
) -> Path:
    """
    Generates a unique, sequentially numbered filename for a new MP3 file.

    This function ensures that each generated MP3 file has a distinct name
    (e.g., `output_001.mp3`, `output_002.mp3`) within the specified directory,
    preventing overwrites and making it easy to manage multiple audio outputs.
    
    Args:
        directory: The `Path` object representing the directory where MP3s are saved.
        prefix: The filename prefix (e.g., "output_").
        extension: The file extension (e.g., ".mp3").
        
    Returns:
        A `Path` object representing the full, resolved path for the next
        available output MP3 filename.
    """
    # Create the output directory if it doesn't already exist.
    directory.mkdir(parents=True, exist_ok=True)
    highest = 0 # Track the highest existing sequence number.
    pattern = OUTPUT_FILENAME_PATTERN # Compiled regex for parsing filenames.

    # Iterate through existing files to find the highest sequence number.
    for file in directory.glob(f"{prefix}*{extension}"):
        match = pattern.match(file.name)
        if match:
            highest = max(highest, int(match.group(1)))
    
    # Construct the new filename with an incremented sequence number (e.g., 001, 002).
    filename = f"{prefix}{highest + 1:03d}{extension}"
    # Return the full resolved path for the new file.
    return (directory / filename).resolve()


# ================================================================================
# OPENAI-COMPATIBLE API HELPER
# ================================================================================

def _call_openai_style(
    endpoint: str,
    api_key: str,
    model: str,
    prompt: str,
    extra_headers: Dict[str, str] | None = None,
) -> str:
    """
    Helper function for making API calls to providers that adhere to OpenAI's
    Chat Completions API schema.

    Many modern LLM APIs (e.g., OpenAI, DeepSeek, Grok, Mistral, Perplexity, Qwen)
    offer an interface compatible with OpenAI's API. This helper abstracts
    the common request structure, making it easier to integrate new providers.
    
    Args:
        endpoint: The full URL of the API endpoint (e.g., `https://api.openai.com/v1/chat/completions`).
        api_key: The API authentication key for the specific provider.
        model: The name of the LLM model to use for the completion (e.g., "gpt-4o", "deepseek-reasoner").
        prompt: The user's input text or query.
        extra_headers: An optional dictionary of additional HTTP headers to include in the request.
        
    Returns:
        The generated text content from the model's response.
        
    Raises:
        requests.HTTPError: If the API call returns an HTTP error status (4xx or 5xx).
        requests.RequestException: For other issues like network problems or timeouts.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",  # API key for authentication
        "Content-Type": "application/json",  # Specify JSON payload format
        "Accept": "application/json",  # Request JSON response
    }
    if extra_headers:
        headers.update(extra_headers) # Merge any additional headers
    
    # Construct the JSON payload for the chat completion request.
    # A system message is included to set the model's persona.
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a precise technical assistant."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,  # Controls randomness; lower values make output more deterministic
        "max_tokens": 512,  # Limits the length of the generated response
    }
    
    # Send the POST request to the API endpoint.
    response = requests.post(
        endpoint, headers=headers, json=payload, timeout=TIMEOUT
    )
    # Raise an exception for HTTP errors (4xx or 5xx status codes).
    response.raise_for_status()
    
    # Parse the JSON response and extract the generated content.
    data = response.json()
    return data["choices"][0]["message"]["content"].strip()


# ================================================================================
# PROVIDER IMPLEMENTATIONS
# ================================================================================

# Each function below defines how to call a specific LLM provider.
# They primarily use the `_call_openai_style` helper or their respective SDKs.

# --- OpenAI ---
def call_openai(prompt: str) -> str:
    """Invokes OpenAI's Chat Completions endpoint with the 'gpt-4o' model."""
    # Uses the generic OpenAI-compatible helper for a straightforward API call.
    return _call_openai_style(
        endpoint=get_endpoint("OpenAI"), # Dynamically gets the OpenAI endpoint from PROVIDER_SPECS
        api_key=require_env("OPENAI_API_KEY"), # Requires the OPENAI_API_KEY environment variable
        model="gpt-4o", # Specifies the model to use
        prompt=prompt,
    )


# --- DeepSeek ---
def call_deepseek(prompt: str) -> str:
    """Calls DeepSeek's reasoning model, typically 'deepseek-reasoner'."""
    # DeepSeek's API is largely OpenAI-compatible.
    # Note: 401 errors usually mean a disabled API key; 400 errors can indicate an outdated model name.
    return _call_openai_style(
        endpoint=get_endpoint("DeepSeek"),
        api_key=require_env("DEEPSEEK_API_KEY"), # Requires the DEEPSEEK_API_KEY
        model="deepseek-reasoner", # Current reasoning model
        prompt=prompt,
    )


# --- Grok (xAI) ---
def call_grok(prompt: str) -> str:
    """Calls xAI's Grok endpoint, utilizing its OpenAI-compatible interface."""
    # Grok's API also follows the OpenAI Chat Completions schema.
    return _call_openai_style(
        endpoint=get_endpoint("Grok (xAI)"),
        api_key=require_env("GROK_API_KEY"), # Requires the GROK_API_KEY
        model="grok-4-1-fast-reasoning", # Current Grok model for fast reasoning
        prompt=prompt,
    )


# --- Kimi (Moonshot) - Streaming with Reasoning ---
def call_kimi(prompt: str) -> str:
    """
    Calls Moonshot's Kimi API, designed to handle its unique streaming
    and reasoning trace capabilities.

    This function is more complex as Kimi can stream both a "reasoning trace"
    (intermediate thought process) and the final content separately. It uses
    the official OpenAI SDK but with a custom `base_url` to correctly target
    the Kimi API.
    
    Args:
        prompt: The user's input text.
        
    Returns:
        A formatted string containing both the reasoning trace (if available)
        and the final answer from Kimi.
    """
    api_key = require_any("MOONSHOT_API_KEY", "KIMI_API_KEY") # Can accept either key
    # Moonshot's endpoint needs the base URL without the `/chat/completions` suffix
    # when initializing the OpenAI client.
    client = openai.OpenAI(
        base_url=get_endpoint("Kimi (Moonshot)").replace("/chat/completions", ""), 
        api_key=api_key
    )

    # Initiate a streaming chat completion request.
    stream = client.chat.completions.create(
        model="kimi-k2-0711-preview", # Specific Kimi model
        messages=[
            {"role": "system", "content": "You are Kimi."}, # System message for Kimi's persona
            {"role": "user", "content": prompt},
        ],
        max_tokens=1024 * 32,  # Kimi supports larger context windows (32K tokens max)
        temperature=1.0, # Higher temperature for more creative responses
        stream=True,  # Enable streaming to capture reasoning trace and content chunks
    )

    reasoning: List[str] = [] # Collect parts of the reasoning trace
    content: List[str] = [] # Collect parts of the final content
    
    # Process each chunk received from the streaming API.
    for chunk in stream:
        if not chunk.choices:
            continue
        choice = chunk.choices[0]
        delta = getattr(choice, "delta", None) # The incremental part of the response
        if delta is None:
            continue

        # Kimi-specific: collect reasoning content if present in the delta.
        reasoning_content = getattr(delta, "reasoning_content", None)
        if reasoning_content:
            reasoning.append(reasoning_content)

        # Collect the actual content of the model's response.
        delta_content = getattr(delta, "content", None)
        if not delta_content:
            continue
        if isinstance(delta_content, list): # Handle multi-part content if it comes as a list
            for part in delta_content:
                text = getattr(part, "text", None) or getattr(part, "content", None)
                if text:
                    content.append(text)
        else:
            content.append(str(delta_content))

    # Assemble the final response, including reasoning and answer sections.
    response_sections = []
    if reasoning:
        response_sections.append("Reasoning Trace:\n" + "".join(reasoning).strip())
    if content:
        response_sections.append("Answer:\n" + "".join(content).strip())
    if not response_sections:
        response_sections.append("Kimi returned an empty response.") # Fallback for empty response
    
    return "\n\n".join(response_sections)


# --- Anthropic (Claude) ---
def call_anthropic(prompt: str) -> str:
    """Calls Claude via the official Anthropic SDK."""
    api_key = require_env("ANTHROPIC_API_KEY") # Requires the ANTHROPIC_API_KEY
    client = anthropic.Anthropic(api_key=api_key) # Initialize Anthropic client
    message = client.messages.create(
        model="claude-sonnet-4-5", # Specific Claude model
        max_tokens=1000, # Max tokens for the response
        messages=[{"role": "user", "content": prompt}], # User's prompt
    )
    
    # Extract text from the response message's content blocks.
    text_chunks: List[str] = []
    for block in message.content:
        if getattr(block, "type", None) == "text" and getattr(block, "text", None):
            text_chunks.append(block.text)
    
    if not text_chunks:
        return "Claude returned no text blocks." # Fallback for empty text content
    return "\n\n".join(text_chunks)


# --- Google Gemini ---
def call_gemini(prompt: str) -> str:
    """Calls Google's Gemini API using the official SDK."""
    api_key = require_env("GEMINI_API_KEY") # Requires the GEMINI_API_KEY
    # Attempt to get model name from PROVIDER_SPECS, fallback to a default.
    model_candidates = get_models("Gemini", "models.generate_content")
    model_name = model_candidates[0] if model_candidates else "gemini-pro-latest"

    genai.configure(api_key=api_key) # Configure the Generative AI SDK with the API key
    model = genai.GenerativeModel(f"models/{model_name}") # Initialize the Gemini model
    
    # Generate content with specified configuration.
    response = model.generate_content(
        prompt,
        generation_config={
            "temperature": 0.2, # Controls randomness
            "max_output_tokens": 8192, # Max tokens for the output
        },
        request_options={"timeout": TIMEOUT}, # Apply the global timeout
    )

    # Safely extract text from the Gemini response, handling potential empty candidates or feedback.
    candidates = getattr(response, "candidates", [])
    if not candidates:
        finish_reason = getattr(response, "prompt_feedback", None)
        return f"Gemini: No candidates returned. Feedback: {finish_reason}"

    parts: List[str] = []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        if not content:
            continue
        for part in getattr(content, "parts", []) or []:
            snippet = getattr(part, "text", None)
            if snippet:
                parts.append(snippet.strip())

    if not parts:
        return f"Gemini returned empty response. Candidates: {len(candidates)}" # Fallback for empty content
    
    return "\n\n".join(parts)


# --- Mistral ---
def call_mistral(prompt: str) -> str:
    """Calls Mistral's managed API via its official SDK."""
    api_key = require_env("MISTRAL_API_KEY") # Requires the MISTRAL_API_KEY
    # Attempt to get model name from PROVIDER_SPECS, fallback to a default.
    model_candidates = get_models("Mistral", "chat completions")
    model_name = model_candidates[0] if model_candidates else "mistral-medium-latest"

    client = Mistral(api_key=api_key) # Initialize Mistral client
    response = client.chat.complete(
        model=model_name, # Specific Mistral model
        messages=[{"role": "user", "content": prompt}], # User's prompt
        temperature=0.2, # Controls randomness
        max_tokens=512, # Limits response length
    )

    # Normalize response format, handling various structures.
    choices = getattr(response, "choices", None) or []
    if not choices:
        return "Mistral returned no choices."

    message = getattr(choices[0], "message", None)
    content = ""
    if isinstance(message, dict):
        content = message.get("content") or ""
    else:
        content = getattr(message, "content", "") or ""

    if isinstance(content, list): # Handle multi-part content
        text_parts: List[str] = []
        for part in content:
            if isinstance(part, dict):
                text = part.get("text") or part.get("content")
                if text:
                    text_parts.append(str(text))
            elif isinstance(part, str):
                text_parts.append(part)
        content = "".join(text_parts)
    elif not isinstance(content, str):
        content = str(content)

    content = content.strip()
    return content or "Mistral returned an empty response." # Fallback for empty content


# --- Perplexity (Search) ---
def call_perplexity(prompt: str) -> str:
    """Calls Perplexity's OpenAI-compatible search/chat endpoint."""
    # Perplexity's API is also OpenAI-compatible.
    endpoint = os.environ.get(
        "PERPLEXITY_API_URL", "https://api.perplexity.ai/chat/completions"
    )
    model = os.environ.get("PERPLEXITY_MODEL", "sonar") # Default Perplexity model
    return _call_openai_style(
        endpoint=endpoint,
        api_key=require_env("PERPLEXITY_API_KEY"), # Requires the PERPLEXITY_API_KEY
        model=model,
        prompt=prompt,
    )


# --- Groq (Llama) ---
def call_groq(prompt: str) -> str:
    """Calls Groq's managed Llama models via its official SDK."""
    api_key = require_env("GROQ_API_KEY") # Requires the GROQ_API_KEY
    model = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile") # Default Groq model
    client = Groq(api_key=api_key) # Initialize Groq client
    
    completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a precise technical assistant."},
            {"role": "user", "content": prompt},
        ],
        model=model, # Specific Groq model
        temperature=0.2, # Controls randomness
        max_tokens=512, # Limits response length
    )

    # Normalize response format, handling various structures.
    choices = getattr(completion, "choices", None) or []
    if not choices:
        return "Groq returned no choices."

    message = getattr(choices[0], "message", None)
    content = ""
    if isinstance(message, dict):
        content = message.get("content") or ""
    else:
        content = getattr(message, "content", "") or ""

    if isinstance(content, list): # Handle multi-part content
        text_parts: List[str] = []
        for part in content:
            if isinstance(part, dict):
                text = part.get("text") or part.get("content")
                if text:
                    text_parts.append(str(text))
            elif isinstance(part, str):
                text_parts.append(part)
        content = "".join(text_parts)
    elif not isinstance(content, str):
        content = str(content)

    content = content.strip()
    return content or "Groq returned an empty response." # Fallback for empty content


# --- Qwen (Alibaba) ---
def call_qwen(prompt: str) -> str:
    """Calls Alibaba's Qwen models through its DashScope-compatible OpenAI endpoint."""
    api_key = require_env("DASHSCOPE_API_KEY") # Requires the DASHSCOPE_API_KEY
    # DashScope provides an OpenAI-compatible interface, requires a custom base_url.
    base_url = os.environ.get(
        "DASHSCOPE_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    )
    model = os.environ.get("QWEN_MODEL", "qwen-plus") # Default Qwen model

    client = openai.OpenAI(api_key=api_key, base_url=base_url) # Initialize OpenAI client with custom base URL
    completion = client.chat.completions.create(
        model=model, # Specific Qwen model
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2, # Controls randomness
        max_tokens=512, # Limits response length
    )

    # Normalize response format, handling various structures.
    choices = getattr(completion, "choices", None) or []
    if not choices:
        return "Qwen returned no choices."

    message = getattr(choices[0], "message", None)
    content = ""
    if isinstance(message, dict):
        content = message.get("content") or ""
    else:
        content = getattr(message, "content", "") or ""

    if isinstance(content, list): # Handle multi-part content
        pieces: List[str] = []
        for part in content:
            if isinstance(part, dict):
                text = part.get("text") or part.get("content")
                if text:
                    pieces.append(str(text))
            elif isinstance(part, str):
                pieces.append(part)
        content = "".join(pieces)
    elif not isinstance(content, str):
        content = str(content)

    content = content.strip()
    return content or "Qwen returned an empty response." # Fallback for empty content


# ================================================================================
# MAIN TEST RUNNER
# ================================================================================

def run_all(prompt: str = DEFAULT_PROMPT) -> Dict[str, str]:
    """
    Executes the given prompt against all configured LLM providers and collects their responses.

    This function acts as the central testing mechanism, iterating through each
    provider defined in the `callers` dictionary. It gracefully handles errors
    from individual providers, allowing the script to continue and collect
    results from all available services.
    
    Args:
        prompt: The text prompt to send to each LLM provider. Defaults to `DEFAULT_PROMPT`.
        
    Returns:
        A dictionary where keys are human-readable provider labels (e.g., "OpenAI (gpt-4o)")
        and values are either the successful text response from the provider or
        a human-friendly error message if the call failed.
    """
    # Dictionary mapping display names to their respective calling functions.
    # New providers should be added here after their `call_` function is implemented.
    callers: Dict[str, Callable[[str], str]] = {
        "OpenAI (gpt-4o)": call_openai,
        "DeepSeek (deepseek-r1)": call_deepseek,
        "Grok (grok-4-1-fast-reasoning)": call_grok,
        "Kimi (kimi-k2-0711-preview)": call_kimi,
        "Anthropic (claude-sonnet-4-5)": call_anthropic,
        "Gemini (gemini-1.5-pro-latest)": call_gemini,
        "Mistral (mistral-medium-latest)": call_mistral,
        "Perplexity (sonar)": call_perplexity,
        "Groq (llama-3.3-70b-versatile)": call_groq,
        "Qwen (qwen-plus)": call_qwen,
    }

    results: Dict[str, str] = {} # Dictionary to store results for each provider.
    for label, func in callers.items():
        try:
            # Call the provider-specific function and store its response.
            results[label] = func(prompt)
        except Exception as exc: # Catch any exceptions that occur during an API call.
            # Store a descriptive error message instead of crashing the entire script.
            # This is crucial for diagnosing issues with individual providers.
            results[label] = f"Error: {type(exc).__name__}: {exc}"
    
    return results


# ================================================================================
# MAIN EXECUTION
# ================================================================================

if __name__ == "__main__":
    # 1. Ensure requirements.txt is up-to-date.
    # This helps users install all necessary dependencies easily.
    ensure_requirements_file()
    
    # 2. Load environment variables from the .env file.
    # This step is critical for securely accessing API keys and configurations.
    load_env()
    
    # 3. Run tests against all configured LLM providers.
    print("Testing all LLM providers...\n")
    responses = run_all()
    
    # 4. Print the raw responses from each provider.
    for provider, output in responses.items():
        print(f"\n{provider}\n{'-' * len(provider)}\n{output}")

    try:
        # 5. Select the best joke and convert it to speech.
        best_joke = "Why do programmers always mix up Halloween and Christmas? Because Oct 31 == Dec 25!"
        print("\nBest Joke Selection")
        print("--------------------")
        print(f"Joke: {best_joke}")

        # 6. Generate a unique filename for the MP3 output and convert the joke to speech.
        output_path = next_output_mp3_path()
        tts_status = convert_quote_to_speech(best_joke, output_path)
        print(f"\n{tts_status}")
    except RuntimeError as exc:
        # Handle cases where TTS conversion failed.
        print(f"\nUnable to complete TTS operation: {exc}")

    print("\n✅ Testing complete!")
    print("\n" + "=" * 25)
    print("TROUBLESHOOTING GUIDE:")
    print("=" * 25)
    print("- 401 errors (Unauthorized): Check your API key in the .env file.")
    print("  Ensure it is correct and has the necessary permissions (e.g., `text_to_speech` for ElevenLabs).")
    print("- 404 errors (Not Found): Verify the model name and API endpoint in your provider table (`.env`).")
    print("  Model names can change; check the provider's official documentation.")
    print("- Timeout errors: The provider API may be slow, overloaded, or unreachable.")
    print("  Check your internet connection and the provider's service status.")
    print("- Empty responses: The model may not have generated content, or the prompt format might be unsupported.")
    print("  Try adjusting the prompt or checking the provider's specific API documentation.")
    print("- `NameError` or `TypeError` during ElevenLabs conversion: This usually indicates an issue")
    print("  with how audio data is handled or played. Ensure correct data types are used.")
