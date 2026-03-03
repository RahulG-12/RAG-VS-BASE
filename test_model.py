from langchain_ollama import OllamaLLM

print("Loading model...")
llm = OllamaLLM(model="phi3:mini")

print("Sending prompt...")
response = llm.invoke("Say hello")

print("Response:", response)