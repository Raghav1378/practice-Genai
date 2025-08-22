# from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint

# from dotenv import load_dotenv


# load_dotenv()
# llm=HuggingFaceEndpoint(
#     repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",task="text-generation"
# )
# model=ChatHuggingFace(llm=llm)


# answer=model.invoke("What is the color of parrot")
# print(answer.content)






import time
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFacePipeline
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import os
from dotenv import load_dotenv

load_dotenv()  # for HUGGINGFACEHUB_API_TOKEN if using API

def call_hf_api_with_retry(question, repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0", task="text-generation", max_retries=5):
    """
    Calls Hugging Face API endpoint with retries on failure (503 etc.)
    """
    llm = HuggingFaceEndpoint(repo_id=repo_id, task=task)
    model = ChatHuggingFace(llm=llm)
    
    for i in range(max_retries):
        try:
            answer = model.invoke(question)
            return answer.content
        except Exception as e:
            wait = 2 ** i
            print(f"[API] Request failed ({e}). Retrying in {wait}s...")
            time.sleep(wait)
    return "[API] Failed after multiple retries."

def call_hf_local_with_retry(question, model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0", max_retries=3):
    """
    Calls a local Hugging Face model with retries (mostly for GPU memory issues)
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=100, temperature=0.7)
        llm = HuggingFacePipeline(pipeline=pipe)
        chat_model = ChatHuggingFace(llm=llm)
        
        for i in range(max_retries):
            try:
                answer = chat_model.invoke(question)
                return answer.content
            except Exception as e:
                wait = 2 ** i
                print(f"[LOCAL] Request failed ({e}). Retrying in {wait}s...")
                time.sleep(wait)
        return "[LOCAL] Failed after multiple retries."
    except Exception as e:
        return f"[LOCAL] Initialization failed: {e}"

# ===========================
# USAGE EXAMPLES
# ===========================

question1 = "What is the capital of India?"
question2 = "Why is the sky blue?"
question3 = "Why do ChatGPT models take time to load?"

# Using API (requires HUGGINGFACEHUB_API_TOKEN)
# print(call_hf_api_with_retry(question1))
# print(call_hf_api_with_retry(question2))

# Using Local Model
# Make sure you have enough RAM/GPU for this
print(call_hf_local_with_retry(question3))
