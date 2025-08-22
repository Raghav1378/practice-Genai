from langchain_huggingface import ChatHuggingFace,HuggingFacePipeline
from langchain_core.prompts import PromptTemplate


llm=HuggingFacePipeline.from_model_id(model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",task="text-generation",pipeline_kwargs=dict(temperature=0.5,max_new_tokens=100))
model=ChatHuggingFace(llm=llm)


answer=model.invoke("What is the average weight of an adult giant panda?"
)

print(answer.content)