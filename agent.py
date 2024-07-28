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
Your are a digital twin of a job applicant, Kong Ren Hwai. KNOWLEDGE below are how Kong Ren Hwai answer interview question. KNOWLEDGE 1 is the most relevant knowledge to the PROMPT, follow by KNOWLEDGE 2, and then KNOWLEDGE 3. You can skip the words 'KNOWLEDGE 1', 'KNOWLEDGE 2' and 'KNOWLEDGE 3' in your reply.
You will answer job interviewer's PROMPT. OPENING is job opening information provided by hiring manager. KNOWLEDGE is provided by Kong Ren Hwai. 
You are encouraged to use KNOWLEDGE to answer PROMPT specific to the JOB DESCRIPTION.
IF the JOB DESCRIPTION and KNOWLEDGE are not related to PROMPT, you can ignore JOB DESCRIPTION and KNOWLEDGE during answering.
Answer in perspective of Kong Ren Hwai, for example:
Question: Tell me about yourself.
Answer: My name is Kong Ren Hwai, and I am seeking a role as a Business Analyst, Data Analyst, or Investment Analyst. Previously, I worked as a successful Business Analyst and Engineer at Micron. During a recent career break, I focused on enhancing my Python skills in data science and expanding my finance knowledge by studying for the CFA. I'm excited to be here and to have the opportunity to discuss how my background and skills align with this role.
PROMPT: {prompt}
JOB DESCRIPTION: {job_summary}
KNOWLEDGE: {passage}
""").format(prompt=prompt, job_summary=job_summary,passage=escaped)

  return prompt

def gemini_chat(full_prompt):
    #model = genai.GenerativeModel('gemini-1.0-pro')
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
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

    col1, col2, col3 = st.columns([1.3 ,0.2, 1])

    with col1:
        st.subheader("About Myself")
        st.image("assets/image1.jpg", width=360)
        st.write('Hi! This is me, Ren Hwai, chilling in Iceland. Happy family trip during my career break!')
        st.write("My [Linkedin](https://www.linkedin.com/in/renhwai-kong/) Profile")
        st.write("Visit my [Github](https://github.com/Ren1990?tab=repositories) projects")
        st.write("Take a look on [Tableau](https://public.tableau.com/app/profile/kyloren.kong/viz/Demo_2024InvestmentPortfolio/DBPortfolio) viz")        
        st.write('After working in top US semicond company for 8 years as Sr. Business Analyst and Process Development Engineer, I took a long break to sharpen my Python skill in data science & analysis, and study for CFA (Chartered Finance Analyst) to look for new industry exposure and work opportunity.')
                 
    with col3:
        st.subheader("Job Description")
        job_summary=''
        job_description=st.text_area(
        "You can paste a job description here.",
        label_visibility="visible",
        height=400
        )
        if job_description!='':
            job_summary= update_job_summary(job_description)
   
#Main chat
    st.subheader("Begin interview here...",divider='rainbow')
    if job_summary!='':
        st.write(job_summary)

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.table=update_knowledge()

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Good day, nice to meet you!"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    passage=retrieve_knowledge(prompt, st.session_state.table)

    with st.chat_message("assistant"):
        response=st.write_stream(gemini_chat(make_prompt(prompt, job_summary, passage)))
        st.session_state.messages.append(
            {"role": "assistant", "content": response})
       