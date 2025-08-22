from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from transformers import pipeline

pipe = pipeline(
    task="text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    temperature=0.7,
    max_new_tokens=100
)


llm =HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs=dict( temperature=0.7,
    max_new_tokens=100)
)

# Create a LangChain chat model
model = ChatHuggingFace(llm=llm)

# Run a query
answer = model.invoke("What is the capital of India?")
answer2=model.invoke("Why the color of the sky is blue")
answer3=model.invoke("why the chatgpt models take some time to load")
print(answer3.content)

# print(answer3.content)


# Install required packages first (run in your terminal)
# pip install langchain transformers torch sentencepiece

# from langchain_community.llms import HuggingFacePipeline
# from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

# model_name = "TheBloke/vicuna-7B-1.1-HF"  
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="auto")

# pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_length=512)


# llm = HuggingFacePipeline(pipeline=pipe)


# output = llm("Explain LangChain in simple words.")
# print(output)
