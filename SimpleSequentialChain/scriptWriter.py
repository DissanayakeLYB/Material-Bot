import os
from dotenv import load_dotenv

load_dotenv()

import streamlit as st
from langchain_community.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain

os.environ['OPENAI_API_KEY'] = os.getenv('API_key')

#app framework
st.title("ScriptGen")
prompt = st.text_input("Type a topic you need a YT video script of : ")
st.button("Go")

#prompt templates
title_template = PromptTemplate(
    input_variables = ['topic'],
    template = "Write a suitable topic for a Youtube video related to this {topic}  "
)

ref_template = PromptTemplate(
    input_variables = ['title'],
    template = "Write a script for this Title : {title}. "
)


#llms
llm = OpenAI(temperature=0.9)
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True)
reference_chain = LLMChain(llm=llm, prompt=ref_template, verbose=True)
sequential_chain = SimpleSequentialChain(chains=[title_chain,reference_chain], verbose=True)

#response
if prompt:
    response = sequential_chain.run(prompt)
    st.write(response) 