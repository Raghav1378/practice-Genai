from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import streamlit as st 

load_dotenv()

model=ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# Basic code for my chatgpt
# question=st.text_input("Please enter your input here")

# if question:
#     answer=model.invoke(question)
#     st.write(answer.content)



## this is the code for static prompt where we get fixed prompt taht you give to model every time 
# Basic code for my summarizer
# st.header("Research tool")

# user_input=st.text_input("Please enter your prompt")

# if st.button("Summarize"): 
#     result=model.invoke(user_input)
#     st.write(result.content)









# Dynmaic Prompt ->itsmore of like that pick up the tone and length and all and it is not same it depend upon the style of user and the words in it it adapts based upon the variables  
st.header("Research Tool")

user_input=st.text_input("Please enter your input")

if st.button("Summarize"): 
    result=model.invoke(user_input)
    st.write(result.content)
    