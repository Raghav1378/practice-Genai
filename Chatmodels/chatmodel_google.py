# from langchain_google_genai import ChatGoogleGenerativeAI
# from dotenv import load_dotenv
# load_dotenv()


# model=ChatGoogleGenerativeAI(model='gemini-2.5-flash')

# output=model.invoke("ok so wha is the capital of the india")

# print(output.content)

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model=ChatGoogleGenerativeAI(model="gemini-2.5-flash")

def chat_bot():
    print("Ask anything you want to ask from here ")
    while True: 
        user=input("You:")
        if user.lower()=="exit":
            break
        answer=model.invoke(user)
        print("Gemini:",answer.content)
    
    
