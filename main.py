from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq
import numpy as np

# Initialize the Groq client and embedding model
client = Groq()
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Memory buffer with embeddings
memory = []

def add_to_memory(role, content):
    """
    Add a message to memory along with its embedding.
    """
    embedding = embedding_model.encode(content, convert_to_numpy=True)
    memory.append({"role": role, "content": content, "embedding": embedding})

def retrieve_relevant_memory(user_input, top_k=5):
    """
    Retrieve the top-k most relevant messages from memory based on cosine similarity.
    """
    if not memory:
        return []

    # Compute the embedding of the user input
    user_embedding = embedding_model.encode(user_input, convert_to_numpy=True)

    # Calculate similarities
    similarities = [cosine_similarity([user_embedding], [m["embedding"]])[0][0] for m in memory]

    # Sort memory by similarity and return the top-k messages
    relevant_messages = sorted(zip(similarities, memory), key=lambda x: x[0], reverse=True)
    return [m[1] for m in relevant_messages[:top_k]]

def construct_prompt(memory, user_input, max_tokens=500):
    """
    Construct the prompt by combining relevant memory and the current user input.
    """
    relevant_memory = retrieve_relevant_memory(user_input)

    # Combine relevant memory into the prompt
    prompt = ""
    token_count = 0
    for message in relevant_memory:
        message_text = f'{message["role"]}: {message["content"]}\n'
        token_count += len(message_text.split())
        if token_count > max_tokens:
            break
        prompt += message_text

    # Add the user input at the end
    prompt += f'user: {user_input}\n'
    return prompt

def trim_memory(max_size=50):
    """
    Trim the memory to keep it within the specified max size.
    """
    if len(memory) > max_size:
        memory.pop(0)  # Remove the oldest entry

def summarize_memory():
    """
    Summarize the memory buffer to free up space.
    """
    if not memory:
        return

    long_term_memory = " ".join([m["content"] for m in memory])
    summary = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {"role": "system", "content": "Summarize the following text for key points."},
            {"role": "user", "content": long_term_memory},
        ],
        max_tokens= 4096,
    )
    memory.clear()
    memory.append({"role": "system", "content": summary.choices[0].text})

def display_memory():
    """
    Visualize the memory buffer.
    """
    for i, m in enumerate(memory):
        print(f"Memory {i+1}: {m['content']} (Role: {m['role']})")

# Start the chatbot loop
print("Welcome to the AI Chatbot with Memory! Type 'exit' to end the conversation.")
while True:
    # Get user input
    user_input = input("You: ")

    # Exit condition
    if user_input.lower() == "exit":
        print("Goodbye!")
        break

    # Construct prompt with relevant memory
    prompt = construct_prompt(memory, user_input)
    print("\n prompt given:\n")

    # Get response from the model
    completion = client.chat.completions.create(
        model="deepseek-r1-distill-llama-70b",
        messages=[{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt}],
        temperature=0.6,
        max_tokens=4096,  # Limit response length
        top_p=0.95,
        stream=True,
        stop=None,
    )

    # Display the assistant's response
    print("\n\nAI: ", end="")
    response = ""
    for chunk in completion:
        response_part = chunk.choices[0].delta.content or ""
        print(response_part, end="")
        response += response_part
    print()  # Add a new line after the chatbot response

    # Update memory buffer with the user input and AI response
    add_to_memory("user", user_input)
    add_to_memory("assistant", response)

    # Optionally trim memory to keep it manageable
    trim_memory()
