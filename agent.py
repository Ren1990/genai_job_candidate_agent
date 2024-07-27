import google.generativeai as genai
import os
import textwrap
import pandas as pd
import numpy as np
import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader






#prompt, resume etc job summary: {job_summary} directory
docdir='rag_docs/'
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)





#functions
def make_prompt(prompt, job_summary, passage):
  escaped = passage.replace("'", "").replace('"', "").replace("\n", " ")
  prompt = textwrap.dedent("""
  You are a helpful and informative bot that answers questions using text from the reference passage included below.
  Your are a digital twin of a job applicant, Kong Ren Hwai. KNOWLEDGE below are how Kong Ren Hwai answer interview question.
  You will answer job interviewer's PROMPT using Kong Ren Hwai's perspective. Such as:
  Question: What is your name?
  Answer: My name is Kong Ren Hwai, I am looking for job role in Business Analyst, Data Analyst or Investment Analyst.
  The JOB DESCRIPTION is given below, and you will reply PROMPT below with YOUR KNOWLEDGE below;
  IF the JOB DESCRIPTION and YOUR KNOWLEDGE are not related to PROMPT, you can ignore JOB DESCRIPTION and YOUR KNOWLEDGE during answering.
  PROMPT: {prompt}
  JOB DESCRIPTION: {job_summary}
  KNOWLEDGE: {passage}
  """).format(prompt=prompt, job_summary=job_summary,passage=escaped)

  return prompt

def gemini_chat(full_prompt):
    model = genai.GenerativeModel('gemini-1.0-pro')
    answer = model.generate_content(full_prompt)
    for chunk in answer.text:
        yield chunk


def update_job_summary(job_description):
    prompt = textwrap.dedent("""
    Briefly describe below job description into job position name, company name, job responsibilities and job requirements.
    This is job description: {job_description}
    """).format(job_description=job_description)
    job_model = genai.GenerativeModel('gemini-1.0-pro')
    job_summary= job_model.generate_content(prompt)
    return job_summary.text    

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
  relevant_knowledge=table.loc[(table['relevant score']>0.5)]['content']
  text_list=[]
  i=1
  for t in relevant_knowledge.apply(lambda x: x.replace("\ufeff", "")) and i<=3:
    print(t)
    text_list.append("KNOWLEDGE "+str(i)+": "+t+" ")
    i=i+1

  return "".join(text_list)





#streamlit layout
tab1, tab2 = st.tabs(["Chatbot", "Relevant Knowledge"])

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
        st.write(passage)
        response=st.write_stream(gemini_chat(make_prompt(prompt, job_summary, passage)))
    st.session_state.messages.append(
        {"role": "assistant", "content": response})





with tab2:
    st.table(st.session_state.table[['content','relevant score']].sort_values('relevant score',ascending=False).head(3))