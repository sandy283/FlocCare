# SQLite compatibility fix for ChromaDB on cloud platforms
import sys
try:
    import pysqlite3
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import streamlit as st
import operator
from typing import Annotated, Literal, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.graph.message import add_messages
import ollama
from uuid import uuid4
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
import os
import requests

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from rag_agent import advanced_rag_query


# --- Backend selector and LLM abstraction ---
with st.sidebar:
    st.header("Settings")
    backend = st.selectbox(
        "Choose LLM Backend:",
        ("Hugging Face (Cloud)", "Ollama (Local)"),
        help="Choose which backend to use for LLM responses. Hugging Face works everywhere; Ollama only works locally."
    )
    hf_token_input = st.text_input("Hugging Face Token (paste here for cloud backend)", type="password")
    regulation = st.selectbox(
        "Select Regulation:",
        ["FDA", "EMA", "HSA"],
        help="Choose the regulatory body for compliance checking"
    )
   
    st.markdown("---")
    st.markdown("Example Claims")
    examples = [
        "This drug guarantees 100% effectiveness in curing diabetes.",
        "Clinical studies show this knee surgery has a 95% success rate.",
        "Our pain relief cream is the most advanced in the world!",
        "FDA-approved for treatment of mild to moderate pain.",
        "This supplement will prevent heart attacks."
    ]
   
    for ex in examples:
        if st.button(ex[:50] + "...", key=ex):
            st.session_state.messages.append({"role": "user", "content": ex})
           
            state = {
                "messages": [HumanMessage(content=ex)],
                "regulation": regulation,
                "compliant": None,
                "explanation": None,
                "ask_human": None,
            }
           
            thread_id = uuid4().hex
            config = {"configurable": {"thread_id": thread_id}}
           
            st.session_state.current_thread_id = thread_id
            st.session_state.current_config = config
           
            with st.spinner("Checking compliance..."):
                try:
                    result = st.session_state.graph.invoke(state, config=config)
                   
                    compliant = result.get("compliant", "unknown")
                   
                    # Add section separator for initial compliance result
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": '<div style="background-color: #FFFF00; color: black; padding: 8px; border-radius: 5px; font-weight: bold;"> INITIAL COMPLIANCE ASSESSMENT</div>'
                    })
                    
                    if compliant == "yes":
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"COMPLIANT with {regulation} regulations"
                        })
                        st.session_state.awaiting_other_regulation_check = True
                        st.session_state.current_claim = ex
                    else:
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"NON-COMPLIANT with {regulation} regulations"
                        })
                        st.session_state.awaiting_other_regulation_check = True
                        st.session_state.current_claim = ex
                       
                except Exception as e:
                    st.error(f"Error processing claim: {str(e)}")
           
            st.rerun()


# --- LLM call abstraction ---
def call_llm(prompt, backend, model_name="gemma3:4b"):
    if backend == "Hugging Face (Cloud)":
        HF_API_URL = st.secrets.get("HF_API_URL") or os.environ.get("HF_API_URL")
        HF_TOKEN = hf_token_input or st.secrets.get("HF_TOKEN") or os.environ.get("HF_TOKEN")
        if not HF_API_URL or not HF_TOKEN:
            return "[Hugging Face API credentials not set. Please set HF_API_URL and HF_TOKEN in Streamlit secrets, environment variables, or paste in the sidebar.]"
        headers = {"Authorization": f"Bearer {HF_TOKEN}"}
        try:
            response = requests.post(
                HF_API_URL,
                headers=headers,
                json={"inputs": prompt}
            )
            response.raise_for_status()
            result = response.json()
            if isinstance(result, list) and "generated_text" in result[0]:
                return result[0]["generated_text"]
            elif "generated_text" in result:
                return result["generated_text"]
            elif "error" in result:
                return f"[Hugging Face API error: {result['error']}]"
            else:
                return str(result)
        except Exception as e:
            return f"[Hugging Face API error: {e}]"
    elif backend == "Ollama (Local)":
        try:
            import ollama
            response = ollama.chat(
                model=model_name,
                messages=[{"role": "user", "content": prompt}]
            )
            return response["message"]["content"]
        except Exception as e:
            return f"[Ollama error: {e}]"
    else:
        return "[Invalid backend selected.]"


# --- Replace ask_ollama with call_llm ---
def ask_llm(question: str) -> str:
    return call_llm(question, backend)


def extract_things_to_find(claim, regulation_results, explanations):
    prompt = f"""
You are a regulatory research assistant. Based on the medical claim and compliance analysis, identify the specific things that need to be searched in regulatory documents.

Medical Claim: {claim}
Compliance Results: {regulation_results}
Key Issues from Analysis: {explanations[:500]}...

Extract and list the specific regulatory elements that need to be researched:
- Specific regulations or sections to look up
- Key terms or concepts to search for
- Types of evidence needed
- Regulatory precedents to find

Provide a concise summary (max 200 words) of what exactly needs to be searched in the regulatory database.

Focus on actionable search terms and specific regulatory areas, not general explanations.
"""
    response = ask_llm(prompt)
    return response.strip()


# --- Pydantic Output Schema ---
class ComplianceResult(BaseModel):
    compliant: Literal["yes", "no"]
    regulation: Literal["FDA", "EMA", "HSA"]
    explanation: Optional[str] = None


# --- State Definition ---
class ComplianceState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    compliant: Optional[Literal["yes", "no"]]
    regulation: Optional[Literal["FDA", "EMA", "HSA"]]
    explanation: Optional[str]
    ask_human: Optional[str]


def classifier_node(state: ComplianceState) -> dict:
    user_msg = next((m.content for m in state["messages"] if isinstance(m, HumanMessage)), "")
    regulation = state.get('regulation', 'FDA')
    reg_contexts = {
        'FDA': 'Food and Drug Administration (USA) pharmaceutical and medical device regulations',
        'EMA': 'European Medicines Agency pharmaceutical and medical product regulations',
        'HSA': 'Health Sciences Authority (Singapore) pharmaceutical and medical device regulations'
    }
    prompt = (
        f"You are a pharmaceutical regulatory compliance expert. Classify the following medical claim as compliant or non-compliant with "
        f"{reg_contexts[regulation]}. "
        "Focus on issues like unsubstantiated health claims, lack of clinical evidence, misleading statements, and regulatory approval requirements. "
        "Respond with ONLY 'yes' if compliant, 'no' if non-compliant.\n\n"
        f"Medical Claim: {user_msg}"
    )
    result = ask_llm(prompt).strip().lower()
    if "yes" in result:
        return {"compliant": "yes"}
    elif "no" in result:
        return {"compliant": "no"}
    else:
        return {"compliant": "no"}


def ask_human_node(state: ComplianceState) -> dict:
    answer: str = interrupt("Claim is non-compliant. Would you like an explanation? (yes/no)")
    return {"ask_human": answer.strip().lower()}


def reasoning_node(state: ComplianceState) -> dict:
    user_msg = next((m.content for m in state["messages"] if isinstance(m, HumanMessage)), "")
    decision = state.get("compliant", "no")
    regulation = state.get("regulation", "FDA")
    reg_contexts = {
        'FDA': 'Food and Drug Administration (USA) pharmaceutical and medical device regulations',
        'EMA': 'European Medicines Agency pharmaceutical and medical product regulations', 
        'HSA': 'Health Sciences Authority (Singapore) pharmaceutical and medical device regulations'
    }
    prompt = (
        f"You are a pharmaceutical regulatory compliance expert. Explain why the following medical claim was classified as '{decision}' under {reg_contexts[regulation]}. "
        "Focus on specific regulatory violations such as: unsubstantiated health claims, lack of clinical evidence, "
        "misleading advertising, required disclaimers, approval requirements, or prohibited language. "
        "Be specific about the regulatory principles that apply. Keep response concise but informative.\n\n"
        f"Medical Claim: {user_msg}\n"
        f"Classification: {decision.upper()}"
    )
    explanation = ask_llm(prompt)
    return {"explanation": explanation}


# --- Build Graph ---
def build_graph():
    graph_builder = StateGraph(ComplianceState)
    graph_builder.add_node("classifier", classifier_node)
    graph_builder.add_node("human_input", ask_human_node)
    graph_builder.add_node("reasoning", reasoning_node)
   
    graph_builder.add_edge(START, "classifier")
   
    graph_builder.add_conditional_edges(
        "classifier",
        lambda state: "human_input" if state.get("compliant") == "no" else END,
        {"human_input": "human_input", END: END}
    )
   
    graph_builder.add_conditional_edges(
        "human_input",
        lambda state: "reasoning" if state.get("ask_human") == "yes" else END,
        {"reasoning": "reasoning", END: END}
    )
   
    graph_builder.add_edge("reasoning", END)
   
    memory = MemorySaver()
    return graph_builder.compile(checkpointer=memory)


st.set_page_config(page_title="Medical Compliance Checker", page_icon="", layout="wide")


st.markdown("""
<style>
.stChat {
    background-color: #f0f2f6;
    border-radius: 10px;
    padding: 10px;
}
.user-message {
    background-color: #007AFF;
    color: white;
    padding: 10px 15px;
    border-radius: 18px;
    margin: 5px 0;
    max-width: 70%;
    margin-left: auto;
    text-align: right;
}
.bot-message {
    background-color: #E9E9EB;
    color: black;
    padding: 10px 15px;
    border-radius: 18px;
    margin: 5px 0;
    max-width: 70%;
}
.compliance-badge {
    display: inline-block;
    padding: 5px 15px;
    border-radius: 20px;
    font-weight: bold;
    margin: 10px 0;
}
.compliant {
    background-color: #4CAF50;
    color: white;
}
.non-compliant {
    background-color: #f44336;
    color: white;
}
</style>
""", unsafe_allow_html=True)


if "messages" not in st.session_state:
    st.session_state.messages = []
if "graph" not in st.session_state:
    st.session_state.graph = build_graph()
if "awaiting_explanation_response" not in st.session_state:
    st.session_state.awaiting_explanation_response = False
if "awaiting_other_regulation_check" not in st.session_state:
    st.session_state.awaiting_other_regulation_check = False
if "current_thread_id" not in st.session_state:
    st.session_state.current_thread_id = None
if "current_config" not in st.session_state:
    st.session_state.current_config = None
if "current_claim" not in st.session_state:
    st.session_state.current_claim = None
if "all_regulation_results" not in st.session_state:
    st.session_state.all_regulation_results = {}
if "non_compliant_regulations" not in st.session_state:
    st.session_state.non_compliant_regulations = []
if "awaiting_rag_response" not in st.session_state:
    st.session_state.awaiting_rag_response = False
if "current_explanations" not in st.session_state:
    st.session_state.current_explanations = []


st.title("Medical Compliance Checker")
st.markdown("Check if medical claims comply with regulatory standards (FDA/EMA/HSA)")


chat_container = st.container()


with chat_container:
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f'<div class="user-message">{message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="bot-message">{message["content"]}</div>', unsafe_allow_html=True)


if st.session_state.awaiting_other_regulation_check:
    st.markdown("Would you like to check this claim with other regulations?")
    col1, col2, col3 = st.columns([1, 1, 5])
    with col1:
        if st.button("Check Other Regulations", type="primary"):
            st.session_state.awaiting_other_regulation_check = False
            
            with st.spinner("Checking compliance across all regulations..."):
                # Clear previous results
                st.session_state.all_regulation_results = {}
                st.session_state.non_compliant_regulations = []
                
                # Check with all regulations
                all_regs = ["FDA", "EMA", "HSA"]
                
                for reg in all_regs:
                    state = {
                        "messages": [HumanMessage(content=st.session_state.current_claim)],
                        "regulation": reg,
                        "compliant": None,
                        "explanation": None,
                        "ask_human": None,
                    }
                   
                    result = st.session_state.graph.invoke(state, config=st.session_state.current_config)
                    compliant = result.get("compliant", "unknown")
                    st.session_state.all_regulation_results[reg] = compliant
                    
                    if compliant == "no":
                        st.session_state.non_compliant_regulations.append(reg)
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"{compliant.upper()} COMPLIANT with {reg} regulations"
                    })
               
                # Add section separator for multi-regulation results
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": '<div style="background-color: #FFFF00; color: black; padding: 8px; border-radius: 5px; font-weight: bold;">MULTI-REGULATION COMPLIANCE RESULTS</div>'
                })
                
                if st.session_state.non_compliant_regulations:
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"Regulation Check Complete! \n\n{len(st.session_state.non_compliant_regulations)} regulation(s) found non-compliant. Would you like detailed explanations for all violations?"
                    })
                    st.session_state.awaiting_explanation_response = True
                else:
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": "All regulations passed! This claim appears to be compliant across all regulatory frameworks."
                    })
            st.rerun()
   
    with col2:
        if st.button("Skip Other Regulations"):
            st.session_state.awaiting_other_regulation_check = False
            if any("NON-COMPLIANT" in msg["content"] for msg in st.session_state.messages if msg["role"] == "assistant"):
                st.session_state.non_compliant_regulations = [regulation]  # Current regulation only
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"Would you like an explanation for the non-compliant {regulation} result?"
                })
                st.session_state.awaiting_explanation_response = True
            st.rerun()


if st.session_state.awaiting_explanation_response:
    col1, col2, col3 = st.columns([1, 1, 5])
   
    with col1:
        if st.button("Provide Explanation", type="primary"):
            st.session_state.awaiting_explanation_response = False
            
            with st.spinner("Generating detailed explanations for violations..."):
                # Add section separator for explanations
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": '<div style="background-color: #FFFF00; color: black; padding: 8px; border-radius: 5px; font-weight: bold;"> DETAILED VIOLATION EXPLANATIONS</div>'
                })
                
                explanations = []
                for reg in st.session_state.non_compliant_regulations:
                    # Create a simple state for each regulation and call reasoning directly
                    state = {
                        "messages": [HumanMessage(content=st.session_state.current_claim)],
                        "regulation": reg,
                        "compliant": "no",
                        "explanation": None,
                        "ask_human": None,
                    }
                    
                    # Directly call the reasoning function for regulation-specific explanations
                    result = reasoning_node(state)
                    
                    if result.get("explanation"):
                        explanation_msg = f"{reg} Violation Explanation:\n{result['explanation']}\n"
                        st.session_state.messages.append({"role": "assistant", "content": explanation_msg})
                        explanations.append(f"{reg}: {result['explanation']}")
                
                st.session_state.current_explanations = explanations
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": f"Analysis Complete! Provided explanations for all {len(st.session_state.non_compliant_regulations)} non-compliant regulations.\n\nWould you like to search deeper in regulatory documents for more detailed evidence and citations?"
                })
                st.session_state.awaiting_rag_response = True
            st.rerun()
   
    with col2:
        if st.button("Skip Explanation"):
            st.session_state.awaiting_explanation_response = False
            st.session_state.messages.append({"role": "assistant", "content": "Understood. No explanation provided."})
            st.rerun()


if st.session_state.awaiting_rag_response:
    st.markdown("Would you like to search deeper in regulatory documents for more detailed evidence and citations?")
    col1, col2, col3 = st.columns([1, 1, 5])
    
    with col1:
        if st.button("Deep Document Search", type="primary"):
            st.session_state.awaiting_rag_response = False
            
            with st.spinner("Analyzing what to search for..."):
                try:
                    # Add section separator for deep document search
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": '<div style="background-color: #FFFF00; color: black; padding: 8px; border-radius: 5px; font-weight: bold;"> DEEP REGULATORY DOCUMENT RESEARCH</div>'
                    })
                    
                    regulation_results = []
                    for reg in st.session_state.non_compliant_regulations:
                        result = st.session_state.all_regulation_results.get(reg, "unknown")
                        regulation_results.append(f"{reg}: {result}")
                    
                    search_summary = extract_things_to_find(
                        st.session_state.current_claim,
                        ', '.join(regulation_results),
                        chr(10).join(st.session_state.current_explanations)
                    )
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"Research Focus Identified:\n{search_summary}\n\nNow searching regulatory documents..."
                    })
                    
                    with st.spinner("Searching regulatory documents for detailed evidence..."):
                        rag_result = advanced_rag_query(search_summary)
                    
                    final_prompt = f"""
You are a regulatory compliance expert. Create a comprehensive final analysis using:

1. Original Query: {st.session_state.current_claim}
2. Initial LLM Decision: {', '.join(regulation_results)}
3. Research Focus: {search_summary}
4. RAG Document Search Results: {rag_result}

Provide a final, authoritative compliance assessment that:
- Confirms or refines the initial compliance decision
- Cites specific regulatory documents and sections
- Provides actionable recommendations
- Highlights any discrepancies between initial assessment and document evidence

Structure your response clearly with sections for Final Decision, Evidence, and Recommendations.
"""
                    
                    final_response = ask_llm(final_prompt)
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f'<div style="background-color: #FFFF00; color: black; padding: 8px; border-radius: 5px; font-weight: bold;"> COMPREHENSIVE REGULATORY ANALYSIS</div>\n\n{final_response}'
                    })
                    
                except Exception as e:
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": f"Error during document search: {str(e)}\n\nFalling back to initial analysis provided above."
                    })
            
            st.rerun()
    
    with col2:
        if st.button("Skip Document Search"):
            st.session_state.awaiting_rag_response = False
            st.session_state.messages.append({
                "role": "assistant", 
                "content": "Understood. Analysis complete with LLM explanations only."
            })
            st.rerun()


if not st.session_state.awaiting_explanation_response and not st.session_state.awaiting_other_regulation_check and not st.session_state.awaiting_rag_response:
    user_input = st.chat_input("Enter a medical claim to check compliance...")
   
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
       
        state = {
            "messages": [HumanMessage(content=user_input)],
            "regulation": regulation,
            "compliant": None,
            "explanation": None,
            "ask_human": None,
        }
       
        thread_id = uuid4().hex
        config = {"configurable": {"thread_id": thread_id}}
       
        st.session_state.current_thread_id = thread_id
        st.session_state.current_config = config
        st.session_state.current_claim = user_input
       
        with st.spinner("Checking compliance..."):
            try:
                result = st.session_state.graph.invoke(state, config=config)
               
                compliant = result.get("compliant", "unknown")
               
                # Add section separator with yellow color
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": '<div style="background-color: #FFFF00; color: black; padding: 8px; border-radius: 5px; font-weight: bold;"> INITIAL COMPLIANCE ASSESSMENT</div>'
                })
                
                if compliant == "yes":
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"COMPLIANT with {regulation} regulations"
                    })
                    st.session_state.awaiting_other_regulation_check = True
                else:
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"NON-COMPLIANT with {regulation} regulations"
                    })
                    st.session_state.awaiting_other_regulation_check = True
               
            except Exception as e:
                st.error(f"Error processing claim: {str(e)}")
       
        st.rerun()


st.markdown("---")
st.markdown("Tip: Try different medical claims to see how the compliance checker works!")