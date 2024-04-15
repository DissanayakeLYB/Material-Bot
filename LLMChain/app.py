import os
from dotenv import load_dotenv

load_dotenv()

import streamlit as st
from langchain_community.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper

os.environ['OPENAI_API_KEY'] = os.getenv('API_key')


# app framework
st.title("MatGPT")
prompt = st.text_input("Ask from a Materials Science & Engineering expert : ")



# prompt templates
respond_template = PromptTemplate(
    input_variables = ['topic', 'wikipedia_research'],
    template = "Help me as you are an expert in only Materials Science and Engineering. Write about {topic} in a single paragraph. Use the details on this wikipedia reaserches as well : {wikipedia_research}.If any subject which is not directly or indirectly connected to materials science and engineering field is given as the {topic}, just mention to 'Ask something related to Materials Science and Engineering.' "
)

# memory
memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')


# llms
llm = OpenAI(temperature=0.2)
respond_chain = LLMChain(llm=llm, prompt=respond_template, verbose=True, memory=memory)

wiki = WikipediaAPIWrapper()

submit = st.button("Submit", key = "Enter")


# response
if submit:
    wiki_research = wiki.run(prompt)
    response = respond_chain.run(topic=prompt , wikipedia_research =wiki_research)

    st.write(response) 

    with st.expander('Message History') :
        st.info(memory.buffer)

    with st.expander('Wikipedia Research') :
        st.info(wiki_research)