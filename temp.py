"""
Core Patent Concept Extractor
Main AI agent class for patent seed keyword extraction system
"""

import json
import datetime
import os
import logging
from typing import Any, Dict, List, Literal, Optional, cast

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

from typing import Dict, List, TypedDict, Annotated, Optional
# from langchain_community.llms import Ollama
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import Tool
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field
from langchain.chains import LLMChain
from langchain_tavily import TavilySearch
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
import requests
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Dict

class NormalizationOutput(BaseModel):
    """Output model for normalization: extract problem and technical points from input"""
    problem: str = Field(
        description="The main problem or challenge described in the input idea."
    )
    technical: str = Field(
        description="The core technical solution, method, or approach described in the input idea."
    )

parser = PydanticOutputParser(pydantic_object=NormalizationOutput)
prompt = PromptTemplate(
    template="""<OBJECTIVE_AND_PERSONA>
You are a patent analyst specializing in technical idea normalization. Your task is to read the provided input and extract two main points:
1. The main problem or challenge explicitly described.
2. The core technical solution, method, or approach explicitly proposed.
Your extraction must preserve maximum fidelity to the input, capturing all explicit technical details, constraints, and context without omission.
</OBJECTIVE_AND_PERSONA>

<INSTRUCTIONS>
1. Carefully read the input idea in full.
2. For the "problem":
   - Identify only what is explicitly described as the problem, challenge, limitation, or requirement.
   - Include all explicit technical constraints, requirements, and context.
   - If no problem is mentioned, output exactly: "Not mentioned."
3. For the "technical":
   - Identify only the explicitly proposed technical solution, method, or approach.
   - Include all explicit technical details, mechanisms, components, steps, and context.
   - If no technical solution is mentioned, output exactly: "Not mentioned."
4. Use direct quotations from the input whenever possible. If necessary for readability, you may minimally reassemble fragmented text, but do not infer or add unstated information.
5. If multiple distinct explicit details are present, include all of them clearly.
6. Do not generalize, summarize, or explain beyond what is explicitly written in the input.
</INSTRUCTIONS>

<CONTEXT>
Input idea:
{input}
</CONTEXT>

<OUTPUT_FORMAT>
{format_instructions}
</OUTPUT_FORMAT>

<RECAP>
Extract and return only the JSON output with exactly two fields: "problem" and "technical". Each field must include every explicit detail from the input. If a field is not mentioned, use "Not mentioned." Do not add any extra explanation or text outside the JSON.
</RECAP>
""",
    input_variables=["input"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

llm = ChatOpenAI(
    model_name="qwen/qwen-2.5-72b-instruct:free",
    temperature=0.7,
    openai_api_key="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",
    base_url="https://openrouter.ai/api/v1"
)
input_text = """**Idea title**: Smart Irrigation System with IoT Sensors

**User scenario**: A farmer managing a large agricultural field needs to optimize water usage 
while ensuring crops receive adequate moisture. The farmer wants to monitor soil conditions 
remotely and automatically adjust irrigation based on real-time data from multiple field locations.

**User problem**: Traditional irrigation systems either over-water or under-water crops because 
they operate on fixed schedules without considering actual soil moisture, weather conditions, 
or crop-specific needs. This leads to water waste, increased costs, and potentially reduced 
crop yields."""


prompt = PromptTemplate.from_template(input_text)
chain = LLMChain(llm=llm, prompt=prompt)
result = chain.run({})
print(result)