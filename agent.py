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
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
rank_threshold=0.5
ranking=2





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
  relevant_knowledge=table.loc[(table['relevant score']>rank_threshold)].sort_values('relevant score',ascending=False).head(ranking)
  text_list=[]
  i=1
  for t in relevant_knowledge['content'].apply(lambda x: x.replace("\ufeff", "")):
    text_list.append("KNOWLEDGE "+str(i)+": "+t+" ")
    i=i+1

  return "".join(text_list)





#streamlit layout
st.set_page_config(page_title="Job Interviewee AI Agent", page_icon="🏠", layout="wide") 
margin_r,body,margin_l = st.columns([0.4, 3, 0.4])

with body:
    st.header("Job Interviewee AI Agent",divider='rainbow')
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "Good day! I am Digital Twin of Ren Hwai. I am here for job interview."}]
        st.session_state.table=update_knowledge()

    col1, col2, col3 = st.columns([1.3 ,0.2, 1])

    with col1:
        st.title("About Myself...")
        st.image("assets/image1.png", width=360)
        st.write('Name: Kong Ren Hwai')

 
        
    with col3:
        st.title("Your Job Description")
        job_summary=''
        job_description=st.text_area(
        "You can paste a job description here.",
        label_visibility="visible",
        height=250
        )
        if job_description!='':
            job_summary= update_job_summary(job_description)
   
#Main chat
    st.subheader("Start interview.",divider='rainbow') #,divider='rainbow'
    if job_summary!='':
        st.write(job_summary)   

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
    if prompt := st.chat_input("Let's start!"):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        

        with st.chat_message("assistant"):
            passage=retrieve_knowledge(prompt, st.session_state.table)
            response=st.write_stream(gemini_chat(make_prompt(prompt, job_summary, passage)))
       