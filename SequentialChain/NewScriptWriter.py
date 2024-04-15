import os
from dotenv import load_dotenv

load_dotenv()

import streamlit as st
from langchain_community.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory

os.environ['OPENAI_API_KEY'] = os.getenv('API_key')

#app framework
st.title("ScriptGen 2.0")
prompt = st.text_input("Type a topic you need a YT video title and a script of :")
st.button("Go")

#prompt templates
title_template = PromptTemplate(
    input_variables = ['topic'],
    template = "Write a suitable topic for a Youtube video related to this {topic}  "
)

script_template = PromptTemplate(
    input_variables = ['title'],
    template = "Write a script for this Title : {title}. "
)


#llms
llm = OpenAI(temperature=0.9)
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='title')
script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True, output_key='script')
sequential_chain = SequentialChain(chains=[title_chain,script_chain], input_varibles = ['topic'], output_varibles = ['title','script'], verbose=True)

#response
if prompt:
    response = sequential_chain({'topic' :prompt})
    st.write(response['title']) 
    st.write(response['script']) 