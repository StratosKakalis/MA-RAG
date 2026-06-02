import os

from marag_system.agents.step_definer import task_define
from marag_system.agents.rag import build_rag_agent
from marag_system.src.utils import GraphState, PlanExecState, RagState, StepTaskFormat, StepTaskState, PlanFormat, PlanSummaryFormat, PlanSummaryState, QAAnswerFormat, QAAnswerState
from marag_system.src.prompt_template import extract_system_messgage, extract_human_message, extract_input_variables
from marag_system.src.prompt_template import qa_human_message, qa_input_variables, qa_system_message
from marag_system.src.prompt_template import aggregate_human_message, aggregate_input_variables, aggregate_system_message
from langchain_core.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import MessagesState, StateGraph, START, END

from dotenv import load_dotenv

load_dotenv()

def build_plan_executor():
    API_KEY = os.environ["OPENROUTER_API_KEY"]

    rag_agent = build_rag_agent()

    def single_task_execute(state: PlanExecState):
        cur_task = state["step_question"][-1]
        query = cur_task["task"]
        step_original_docs = []
        if cur_task["type"] == "aggregate":
            messages = [
                SystemMessagePromptTemplate.from_template(aggregate_system_message),
                HumanMessagePromptTemplate.from_template(aggregate_human_message),
            ]
            prompt = ChatPromptTemplate(input_variables=aggregate_input_variables, messages=messages)
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
            llm = ChatOpenAI(model=os.getenv("AGENT_MODEL_NAME"), temperature=0.5, openai_api_key=API_KEY, base_url=os.environ["OPENROUTER_API_BASE"], max_retries=2, max_tokens=8192, 
                extra_body={
                    # "repetition_penalty": 1.1,
                    "provider": provider_preferences
                })
            structured_llm = llm.with_structured_output(QAAnswerFormat)
            chain = prompt | structured_llm
            full_prompt = prompt.format(
                question=query,
            )
            response = chain.invoke({"question": query})
            response = QAAnswerState(**response.model_dump())
            step_doc_ids = []
            step_notes = []
        else:
            response = rag_agent.invoke({
                "question": query
            })
            step_doc_ids = [response["doc_ids"]]
            step_notes = [response["notes"]]
            step_original_docs = [response["documents"]]
            response = response["final_raw_answer"]

    
        return {"step_output": [response], "step_docs_ids": step_doc_ids, "step_notes": step_notes, "step_original_docs": step_original_docs}
    
    def task_definer_out(state: PlanExecState):
        if state["stop"] == True:
            return END
        else:
            return "single_task_execute"
        
    graph_builder = StateGraph(PlanExecState)
    
    graph_builder.add_node("task_definer", task_define)
    graph_builder.add_node("single_task_execute", single_task_execute)
    graph_builder.add_edge(START, "task_definer")
    graph_builder.add_edge("single_task_execute", "task_definer")
    graph_builder.add_conditional_edges("task_definer", task_definer_out)
    graph = graph_builder.compile()
    return graph

