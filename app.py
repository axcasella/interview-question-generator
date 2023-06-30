import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import AgentType
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.chains.question_answering import load_qa_chain
import pickle
import os

load_dotenv()

# Main page contents
def main():
    llm = OpenAI(max_tokens=1024)
    prompt = PromptTemplate(
        input_variables=["career", "subject"],
        template = "Generate 10 multiple choice interview questions, each with 4 choices, to test a {career} candidate's knowledge of {subject}"
    )

    chain = LLMChain(llm=llm, prompt=prompt)

    query_career = st.text_input("Career:")
    query_subject = st.text_input("Subject:")
    if query_career and query_subject:
        st.write(chain.run({
            "career": query_career,
            "subject": query_subject
        }))


if __name__ == "__main__":
    main()