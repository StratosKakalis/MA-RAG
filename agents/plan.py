from marag_system.src.utils import GraphState
from marag_system.src.prompt_template import planing_system_message, planing_human_message, planing_input_variables
from marag_system.src.utils import PlanFormat

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import os

from langchain_core.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate

load_dotenv()

def plan_agent(state: GraphState):
    API_KEY = os.getenv("OPENROUTER_API_KEY")
    original_question = state["original_question"]
    all_mem = []
    for past_exp in state["past_exp"]:
        memory = ""
        plan = ', '.join(past_exp["plan"])
        memory += f"Plan: [{plan}]\n"
        memory += f"Status: {past_exp['plan_summary']['output']} Score: {past_exp['plan_summary']['score']}\n"
        all_mem.append(memory)
    memory = ""
    if len(all_mem) == 0:
        memory = "empty"
    else:
        for id in range(len(all_mem)):
            memory += f"Trial {id}:\n{all_mem[id]}\n"
    
    messages = [
        SystemMessagePromptTemplate.from_template(planing_system_message),
        HumanMessagePromptTemplate.from_template(planing_human_message),
    ]
    prompt = ChatPromptTemplate(input_variables=planing_input_variables, messages=messages)
    provider_preferences = {
        "order": [
            "Phala", 
            "DeepInfra", 
            "Novita", 
            "SiliconFlow", 
            "DeepSeek"
        ],
        "allow_fallbacks": True
    }
    llm = ChatOpenAI(model=os.getenv("AGENT_MODEL_NAME"), temperature=0.5, api_key=API_KEY, base_url=os.environ["OPENROUTER_API_BASE"], max_tokens=8192,
        extra_body={
            # "repetition_penalty": 1.1,
            "provider": provider_preferences
        })
    structured_llm = llm.with_structured_output(PlanFormat)
    chain = prompt | structured_llm
    fprompt = prompt.format(
        question = original_question,
        memory = memory
    )
    output = chain.invoke({
        "question": original_question,
        "memory": memory
    })
    return {"plan": output.step}
