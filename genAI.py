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


    groqApi = ChatGroq(model="gemma2-9b-it",temperature=1)
    outputparser = StrOutputParser()
    chainSec = prompt|groqApi|outputparser
        
    response = chainSec.invoke({"model_name": model_name, "kpis": metrics})
    response = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", response)
    response = re.sub(r'(?<!\d)\.(\s|$)', r'.<br>\1', response)
    # print(response)
    # response = re.sub(r'(</strong>) (\*)', r'\1<br> \2', response)
    return response

model_name = "linear_regression"
metrics = {'R2': 0.7867095141283653, 'MSE': 0.24593774104165655, 'RMSE': 0.49592110364619146, 'MAE': 0.2998984561204504, 'MAPE': 4.720263932348494}
response = getDesciption(model_name, metrics)
print(response)
response = re.sub(r"\*\*(.*?)\*\*", r'}}<strong>{{\1}}</strong>{{', response)
print(response)