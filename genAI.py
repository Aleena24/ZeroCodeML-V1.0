from langchain_core.prompts import ChatPromptTemplate 
from langchain_core.output_parsers import StrOutputParser 
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv 
import re

load_dotenv()

def getDesciption(model_name, metrics):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", 
             "You are a highly knowledgeable Python and machine learning expert. "
             "Your task is to provide a brief and factual description and performance analysis of a given machine learning or deep learning model. "
             "Base your response strictly on the provided model name and KPIs. Do not speculate or hallucinate. "
             "Ensure the response is concise and easy to interpret."),
            ("user", 
             "Model Name: {model_name}\nKPIs: {kpis}\n\n"
             "Provide a brief description of the model and a concise analysis of its performance based only on the given KPIs.")
        ]
    )

    groqApi = ChatGroq(
        model="gemma2-9b-it",
        temperature=1,
        api_key=os.getenv("GROQ_API_KEY")  # safe and avoids deployment issues
    )

    outputparser = StrOutputParser()
    chainSec = prompt | groqApi | outputparser
        
    response = chainSec.invoke({"model_name": model_name, "kpis": metrics})

    # Format markdown-style bold to HTML and add line breaks after periods
    response = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", response)
    response = re.sub(r'(?<!\d)\.(\s|$)', r'.<br>\1', response)

    return response

# Only for testing locally
if __name__ == "__main__":
    model_name = "linear_regression"
    metrics = {
        'R2': 0.7867,
        'MSE': 0.2459,
        'RMSE': 0.4959,
        'MAE': 0.2999,
        'MAPE': 4.72
    }

    response = getDesciption(model_name, metrics)
    print(response)

    # Optional: format for templates
    response = re.sub(r"\*\*(.*?)\*\*", r'}}<strong>{{\1}}</strong>{{', response)
    print(response)
