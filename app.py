import streamlit as sc
import os
from keys import OPEN_AI_API
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SimpleSequentialChain,SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from callback import MyCustomSyncHandler, MyCustomAsyncHandler

#os.environ['OPEN_AI_API']=OPEN_AI_API

print("--------------------------------------------------------------------------------")
sc.title("🦜🔗 Youtube Content Generator")
prompt=sc.text_input("Topic...")

# prompt template 
title_template=PromptTemplate(
    input_variables=["topic"],
    template="Write a Youtube video title about {topic}"
)
script_template=PromptTemplate(
    input_variables=["title"],
    template="write a youtube video script based on {title} "
)

# memory
title_memory=ConversationBufferMemory(input_key='topic',memory_key="chat history")
script_memory=ConversationBufferMemory(input_key='title',memory_key="chat history")

# LLM model initialize
llm_model=OpenAI(openai_api_key=OPEN_AI_API,
                 temperature=0.9,
                 streaming=True,
                 callbacks=[MyCustomSyncHandler(), MyCustomAsyncHandler()]
                 )
# Title chain 
title_chain=LLMChain(llm=llm_model,prompt=title_template,output_key="title",memory=title_memory,verbose=True)
# script chain
script_chain=LLMChain(llm=llm_model,prompt=script_template,output_key="script",memory=script_memory,verbose=True)



# for output
if prompt:
    # output=sequential_chain({"topic":prompt})
    title_=title_chain.run(prompt)

    script_out=script_chain.run(title=title_)

    sc.write("Title: "+title_)
    sc.write("Script: "+script_out)
    with sc.expander('Title History'):
        sc.info(title_memory.buffer)
    with sc.expander('Script History'):
        sc.info(script_memory.buffer)
