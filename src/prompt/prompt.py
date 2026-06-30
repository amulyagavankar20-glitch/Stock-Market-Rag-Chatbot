# src/prompt.py
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

def prompt_template():
    # Define chat template
    chat_template = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful stock market agent."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{query}")
    ])

    # Load chat history from file
    chat_history = []
    try:
        with open("chat_history.txt", "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Simple format: prefix with "User:" or "AI:"
                if line.startswith("User:"):
                    chat_history.append(HumanMessage(content=line.replace("User:", "").strip()))
                elif line.startswith("AI:"):
                    chat_history.append(AIMessage(content=line.replace("AI:", "").strip()))
                else:
                    # fallback: treat as human message
                    chat_history.append(HumanMessage(content=line))
    except FileNotFoundError:
        print("[Prompt] No chat_history.txt found, starting fresh.")

    print("[Prompt] Loaded chat history:", chat_history)

    # # Create prompt with empty query for now
    # prompt = chat_template.invoke({"chat_history": chat_history, "query": ""})
    # print("[Prompt] Final prompt:", prompt)
