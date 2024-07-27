from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
import textwrap
import pandas as pd
import numpy as np

import streamlit as st
import os

from typing import Dict, Generator





#prompt, resume etc job summary: {job_summary} directory
docdir='rag_docs/'
text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=0)





#functions
def langchain_gemini_invoke(prompt,job_sumarry,passage):
    prompt_template = PromptTemplate.from_template("""
    Your are a digital twin of a job applicant, Kong Ren Hwai. KNOWLEDGE below are how Kong Ren Hwai answer interview question.
    You will answer job interviewer's PROMPT using Kong Ren Hwai's perspective. Such as:
    Question: What is your name?
    Answer: My name is Kong Ren Hwai, I am looking for job role in Business Analyst, Data Analyst or Investment Analyst.
    The JOB DESCRIPTION is given below, and you will reply PROMPT below with YOUR KNOWLEDGE below;
    IF the JOB DESCRIPTION and YOUR KNOWLEDGE are not related to PROMPT, you can ignore JOB DESCRIPTION and YOUR KNOWLEDGE during answering.
    PROMPT: {prompt}
    JOB DESCRIPTION: {job_summary}
    KNOWLEDGE: {passage}
    """ 
    )
    llm = ChatGoogleGenerativeAI(model='gemini-1.0-pro')
    chain = prompt_template | llm
   
    stream=chain.invoke(
        {
            'prompt':prompt,
            'job_summary':job_summary,
            'passage':passage,
            }
            )
    for chunk in stream.content:
        yield chunk


def update_job_summary(job_description):
    llm = ChatGoogleGenerativeAI(model='gemini-1.0-pro')

    prompt_template=""""
    Briefly describe below job description into job position name, company name, job responsibilities and job requirements.
    This is job description: {job_description}
    """
    prompt_template = PromptTemplate.from_template(prompt_template)
    chain =(prompt_template|llm)#|StrOutputParser()
    job_summary=chain.invoke({"job_description":job_description})
    text_file=open(docdir+'job_summary.txt','w')
    text_file.write(job_summary.content)
    text_file.close()
    return job_summary.content    

def update_knowledge():
    table=pd.DataFrame(columns=['document', 'content','embedding','relevant score'])
    i=0
    for doc in os.listdir(docdir):
        docsplit=TextLoader(docdir+doc,encoding='utf8').load_and_split(text_splitter)
        for chunk in docsplit:
            embedding = genai.embed_content(model='models/text-embedding-004',content=chunk.page_content,task_type="retrieval_query")
            table.loc[i]=[chunk.metadata['source'],chunk.page_content,embedding['embedding'],0]
            i=i+1
    return table

def retrieve_knowledge(query, table):
  query_embedding = genai.embed_content(model='models/text-embedding-004',
                                        content=query,
                                        task_type="retrieval_query")
  table['relevant score'] = np.dot(np.stack(table['embedding']), query_embedding["embedding"])
  return table['content'].iloc[np.argmax(table['relevant score'])] # Return text from index with max value





#streamlit layout
tab1, tab2 = st.tabs(["Chat", "RAG Search Result"])

with tab1:
    st.title("Job Candidate Agent: Ren Hwai")
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.table=update_knowledge()

#side bar
with st.sidebar:
    job_summary=''
    job_description=st.text_area(
    "Before begin the interview, please provide the job description to generate summary",
    label_visibility="visible",
    height=250
    )
    if job_description!='':
        job_summary= update_job_summary(job_description)
    if job_summary!='':
        st.write(job_summary)

#Main chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
if prompt := st.chat_input("How could I help you?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        passage=retrieve_knowledge(prompt, st.session_state.table)
        response=st.write_stream(langchain_gemini_invoke(prompt,job_summary,passage))
    st.session_state.messages.append(
        {"role": "assistant", "content": response})





with tab2:
    st.table(st.session_state.table[['content','relevant score']].sort_values('relevant score',ascending=False).head(3))