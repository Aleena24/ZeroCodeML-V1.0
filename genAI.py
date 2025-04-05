import os
import re
from langchain_core.prompts import ChatPromptTemplate 
from langchain_core.output_parsers import StrOutputParser 
from langchain_groq import ChatGroq

# ✅ Directly setting the environment variable (for testing only)
os.environ["GROQ_API_KEY"] = "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"  # Replace with real key

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

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("Missing GROQ_API_KEY")

    groqApi = ChatGroq(
        model="gemma2-9b-it",
        temperature=1,
        api_key=api_key
    )

    outputparser = StrOutputParser()
    chainSec = prompt | groqApi | outputparser
        
    response = chainSec.invoke({"model_name": model_name, "kpis": metrics})

    # Optional: Markdown formatting cleanup
    response = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", response)
    response = re.sub(r'(?<!\d)\.(\s|$)', r'.<br>\1', response)

    return response

# Test it out
if __name__ == "__main__":
    model_name = "random_forest"
    metrics = {
        'Accuracy': 0.94,
        'Precision': 0.92,
        'Recall': 0.93,
        'F1-Score': 0.925
    }

    print(getDesciption(model_name, metrics))
