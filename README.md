# AI Chatbot with Memory

Authors: Dhiraj Surve and Swaroop Ingavale

A Python-based conversational AI system with a memory mechanism that retrieves relevant context from past interactions.

## Overview

This project implements a chatbot that can remember and use previous conversations to inform its responses. It uses embeddings and semantic similarity to retrieve the most relevant past messages for each new user input, creating a more coherent and context-aware conversation.

## Features

- **Semantic Memory**: Uses sentence embeddings to encode and retrieve relevant past interactions
- **Relevance-Based Retrieval**: Employs cosine similarity to find the most contextually appropriate memories
- **Memory Management**: Includes mechanisms for trimming and summarizing memory to prevent overflow
- **Integration with LLM APIs**: Works with Groq's API to access powerful language models
- **Interactive Command Line Interface**: Simple text-based interface for conversations

## How It Works

1. **Memory Storage**: Each message (both user and assistant) is stored along with its vector embedding
2. **Retrieval Mechanism**: When a user sends a new message, the system finds similar past messages using cosine similarity
3. **Context Construction**: Relevant memories are combined with the current input to create context-aware prompts
4. **Response Generation**: The enhanced prompt is sent to a large language model via Groq's API
5. **Memory Update**: Both the user input and AI response are added to the memory buffer

## Technical Components

### Dependencies

- `sentence_transformers`: For creating semantic embeddings of text
- `sklearn`: For computing cosine similarity between embeddings
- `groq`: API client for connecting to Groq's language models
- `numpy`: For numerical operations on embeddings

### Key Functions

- `add_to_memory()`: Stores messages with their embeddings
- `retrieve_relevant_memory()`: Finds the most similar past messages
- `construct_prompt()`: Builds a context-aware prompt with relevant memories
- `trim_memory()`: Keeps memory size manageable
- `summarize_memory()`: Creates condensed representations of memory content
- `display_memory()`: Visualizes the current memory buffer

### Models Used

- **Embedding Model**: `all-MiniLM-L6-v2` (via SentenceTransformers)
- **Conversation Models**:
  - `deepseek-r1-distill-llama-70b`: Primary response generation
  - `meta-llama/llama-4-scout-17b-16e-instruct`: Used for memory summarization

## Usage

1. Ensure you have the required packages installed
2. Set up your Groq API credentials
3. Run the script:
   ```
   python main.py
   ```
4. Interact with the chatbot via the command line
5. Type 'exit' to end the conversation

## Memory Management Strategies

The system implements two approaches to memory management:

1. **Trimming**: Removes the oldest memories when the buffer exceeds a specified size
2. **Summarization**: Condenses multiple memories into a single summary memory when needed

## Example Interaction Flow

1. User sends a message
2. System encodes the message and retrieves relevant past interactions
3. A prompt is constructed using these relevant memories and the current input
4. The prompt is sent to the language model for response generation
5. Both the user input and AI response are added to memory
6. The memory is trimmed if necessary

## Future Improvements

- Implement more sophisticated memory organization (e.g., episodic vs. semantic memory)
- Add support for multimodal inputs and memories
- Develop more advanced memory management techniques
- Create a web-based or GUI interface
- Add personalization features based on user history

## License

MIT License

Copyright (c) 2025 Dhiraj Surve and Swaroop Ingavale

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
