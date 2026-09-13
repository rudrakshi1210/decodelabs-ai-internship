# Project 1: Deterministic Rule-Based Chatbot for DecodeLabs

# 1. Knowledge Base (Dictionary with 5+ predefined intents)
responses = {
    "hello": "Hello! Welcome to DecodeLabs AI support. How can I assist you?",
    "hi": "Hi there! What can I do for you today?",
    "what is ai": "Artificial Intelligence is the simulation of human intelligence processes by machines.",
    "who are you": "I am a deterministic, rule-based chatbot designed for Project 1.",
    "how does this work": "I match your inputs against pre-configured rules stored in a Python dictionary.",
    "help": "You can greet me, ask 'what is ai', ask 'who are you', or type 'exit' to quit."
}

print("=" * 50)
print(" DecodeLabs AI Rule Engine Initialized ")
print("=" * 50)
print("Bot: Ask a question, or type 'exit' to end the chat.\n")

# 2. Continuous Loop
while True:
    # 3. Input & Sanitization (.lower() and .strip())
    raw_input = input("You: ")
    clean_input = raw_input.lower().strip()

    # Ignore empty inputs
    if not clean_input:
        continue

    # 4. Exit Strategy
    if clean_input in ["exit", "quit", "bye"]:
        print("Bot: Goodbye! Shutting down system.")
        break

    # 5. Intent Lookup with Fallback Response
    reply = responses.get(clean_input, "Bot: I'm sorry, I don't recognize that rule. Type 'help' for options.")
    print(reply + "\n")