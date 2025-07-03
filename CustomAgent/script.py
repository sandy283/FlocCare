# SQLite compatibility fix for ChromaDB on cloud platforms
import sys
try:
    import pysqlite3
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import streamlit as st
import ollama
import os
from uuid import uuid4
from concurrent.futures import ThreadPoolExecutor

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from rag_agent import advanced_rag_query

# --- LLM Functions ---
def call_llm(prompt, backend, model_name="gemma3:4b", gemini_api_key=""):
    import time
    import logging
    from datetime import datetime
    
    # Set up logging to file
    log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "llm_debug.log")
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        filemode='a'
    )
    
    start_time = time.time()
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    # Log what's being called
    logging.info(f"=== LLM CALL START [{timestamp}] ===")
    logging.info(f"Backend: {backend}")
    logging.info(f"Model: {model_name}")
    logging.info(f"Prompt length: {len(prompt)} characters")
    logging.info(f"Prompt preview: {prompt[:200]}...")
    
    if backend == "Gemini (Cloud)":
        import google.generativeai as genai
        api_key = gemini_api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            logging.info("Gemini API key not set")
            return "[Gemini API key not set. Please provide it in the sidebar.]"
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-1.5-flash')
            logging.info("Making Gemini SDK call...")
            gemini_response = model.generate_content(prompt)
            end_time = time.time()
            logging.info(f"Gemini response received in {end_time - start_time:.2f} seconds")
            response_text = gemini_response.text
            logging.info(f"Gemini response length: {len(response_text)} chars")
            logging.info(f"=== LLM CALL END [{timestamp}] ===")
            return response_text
        except Exception as e:
            logging.info(f"Gemini SDK exception: {e}")
            return f"[Gemini SDK error: {e}]"
    elif backend == "Ollama (Local)":
        try:
            logging.info("Making Ollama call...")
            response = ollama.chat(
                model=model_name,
                messages=[{"role": "user", "content": prompt}]
            )
            end_time = time.time()
            logging.info(f"Ollama response received in {end_time - start_time:.2f} seconds")
            logging.info(f"Response type: {type(response)}")
            
            # Try both access methods for compatibility
            try:
                result = response["message"]["content"]
                logging.info(f"Dict access worked, response length: {len(result)} chars")
                logging.info(f"Response preview: {result[:200]}...")
                logging.info(f"=== LLM CALL END [{timestamp}] ===")
                return result
            except (TypeError, KeyError):
                result = response.message.content
                logging.info(f"Object access worked, response length: {len(result)} chars")
                logging.info(f"Response preview: {result[:200]}...")
                logging.info(f"=== LLM CALL END [{timestamp}] ===")
                return result
        except Exception as e:
            logging.info(f"Ollama error: {e}")
            logging.info(f"=== LLM CALL END [{timestamp}] ===")
            return f"[Ollama error: {e}]"
    else:
        logging.info("Invalid backend selected")
        logging.info(f"=== LLM CALL END [{timestamp}] ===")
        return "[Invalid backend selected.]"

def ask_llm(question: str, backend_to_use: str = "Ollama (Local)", gemini_key: str = "") -> str:
    return call_llm(question, backend_to_use, gemini_api_key=gemini_key)

def classify_claim(claim: str, regulation: str, backend: str, api_key: str = "") -> str:
    """
    Clean, simple function to classify a medical claim as compliant or non-compliant.
    Returns: 'yes', 'no', or 'unknown'.
    """
    reg_contexts = {
        'FDA': 'Food and Drug Administration (USA) pharmaceutical and medical device regulations',
        'EMA': 'European Medicines Agency pharmaceutical and medical product regulations',
        'HSA': 'Health Sciences Authority (Singapore) pharmaceutical and medical device regulations'
    }
    prompt = (
        f"You are a pharmaceutical regulatory compliance expert. Classify the following medical claim as compliant or non-compliant with "
        f"{reg_contexts.get(regulation, regulation)}. "
        "Focus on issues like unsubstantiated health claims, lack of clinical evidence, misleading statements, and regulatory approval requirements. "
        "If the claim includes appropriate disclaimers (e.g., 'not intended to diagnose, treat, cure, or prevent any disease'), or uses cautious language (e.g., 'may help', 'results may vary'), and does not make absolute or misleading promises, it should be considered compliant. "
        "Respond with ONLY 'yes' if compliant, 'no' if non-compliant.\n\n"
        f"Medical Claim: {claim}"
    )
    print(f"[DEBUG] Classifying claim with {backend} for {regulation}")
    result = call_llm(prompt, backend, gemini_api_key=api_key).strip().lower()
    print(f"[DEBUG] Classification result: {result}")
    
    # Detect model/API errors and return 'unknown' if so
    if (
        "api key not set" in result
        or "sdk error" in result
        or "error" in result
        or "invalid backend" in result
        or "not set" in result
    ):
        return "unknown"
    if "yes" in result:
        return "yes"
    elif "no" in result:
        return "no"
    else:
        return "unknown"

def explain_compliance(claim: str, regulation: str, decision: str, backend: str, api_key: str = "") -> str:
    """
    Generate an explanation for why a claim was classified as compliant or non-compliant.
    """
    reg_contexts = {
        'FDA': 'Food and Drug Administration (USA) pharmaceutical and medical device regulations',
        'EMA': 'European Medicines Agency pharmaceutical and medical product regulations', 
        'HSA': 'Health Sciences Authority (Singapore) pharmaceutical and medical device regulations'
    }
    prompt = (
        f"You are a pharmaceutical regulatory compliance expert. Explain why the following medical claim was classified as '{decision}' under {reg_contexts.get(regulation, regulation)}. "
        "Focus on specific regulatory violations such as: unsubstantiated health claims, lack of clinical evidence, "
        "misleading advertising, required disclaimers, approval requirements, or prohibited language. "
        "Be specific about the regulatory principles that apply. Keep response concise but informative.\n\n"
        f"Medical Claim: {claim}\n"
        f"Classification: {decision.upper()}"
    )
    return call_llm(prompt, backend, gemini_api_key=api_key)

# --- Backend selector and LLM abstraction ---
with st.sidebar:
    st.header("Settings")
    backend = st.selectbox(
        "Choose LLM Backend:",
        ("Gemini (Cloud)", "Ollama (Local)"),
        help="Choose which backend to use for LLM responses. Gemini works everywhere; Ollama only works locally."
    )
    gemini_api_key = st.text_input("Gemini API Key (paste here for cloud backend)", value="", type="password")
    download_status = st.empty()
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
            st.session_state.current_claim = ex
           
            with st.spinner("Checking compliance..."):
                try:
                    # Use the simple classification function directly
                    compliant = classify_claim(ex, regulation, backend, gemini_api_key)
                   
                    # Add section separator for initial compliance result
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": '<div style="background-color: #FFFF00; color: black; padding: 8px; border-radius: 5px; font-weight: bold;">INITIAL COMPLIANCE ASSESSMENT</div>'
                    })
                    
                    if compliant == "yes":
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"COMPLIANT with {regulation} regulations"
                        })
                    elif compliant == "no":
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"NON-COMPLIANT with {regulation} regulations"
                        })
                    else:
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"COMPLIANCE STATUS UNKNOWN - Please check your API key or model connectivity"
                        })
                    
                    st.session_state.awaiting_other_regulation_check = True
                       
                except Exception as e:
                    st.error(f"Error processing claim: {str(e)}")
           
            st.rerun()

    if backend == "Gemini (Cloud)":
        st.markdown("Gemini backend is selected. Please ensure your API key is correct.")
    elif backend == "Ollama (Local)":
        if st.button("Download Ollama Model", key="download_ollama_model"):
            with st.spinner("Downloading Ollama model..."):
                try:
                    response = ollama.pull("gemma3")
                    if response.get("success"):
                        st.success("Ollama model downloaded and ready!")
                    else:
                        st.error(f"Error downloading Ollama model: {response.get('error', 'Unknown error')}")
                except Exception as e:
                    st.error(f"Error downloading Ollama model: {str(e)}")


# --- Replace ask_ollama with call_llm ---
def extract_things_to_find(claim, regulation_results, explanations, backend_to_use="Ollama (Local)"):
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
    response = ask_llm(prompt, backend_to_use)
    return response.strip()


# --- Streamlit Configuration ---


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
if "awaiting_explanation_response" not in st.session_state:
    st.session_state.awaiting_explanation_response = False
if "awaiting_other_regulation_check" not in st.session_state:
    st.session_state.awaiting_other_regulation_check = False
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
                    if reg == regulation:
                        # For the initial regulation, we need to re-check to get consistent results
                        compliant = classify_claim(st.session_state.current_claim, reg, backend, gemini_api_key)
                        st.session_state.all_regulation_results[reg] = compliant
                        # Add to non-compliant list if needed
                        if compliant == "no":
                            st.session_state.non_compliant_regulations.append(reg)
                        # Don't add message since this was already shown in initial assessment
                    else:
                        # Use the simple classification function for other regulations
                        compliant = classify_claim(st.session_state.current_claim, reg, backend, gemini_api_key)
                        st.session_state.all_regulation_results[reg] = compliant
                        if compliant == "no":
                            st.session_state.non_compliant_regulations.append(reg)
                        # Standardize message
                        if compliant == "yes":
                            msg = f"COMPLIANT with {reg} regulations"
                        elif compliant == "no":
                            msg = f"NON-COMPLIANT with {reg} regulations"
                        else:
                            msg = f"COMPLIANCE STATUS UNKNOWN for {reg} regulations"
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": msg
                        })
               
                # Add section separator for multi-regulation results
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": '<div style="background-color: #FFFF00; color: black; padding: 8px; border-radius: 5px; font-weight: bold;">MULTI-REGULATION COMPLIANCE RESULTS</div>'
                })

                # Count all non-compliant regulations including the initial one
                total_non_compliant = len(st.session_state.non_compliant_regulations)
                all_compliant = all(st.session_state.all_regulation_results.get(r, "unknown") == "yes" for r in all_regs)
                
                if all_compliant:
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": "All regulations passed! This claim appears to be compliant across all regulatory frameworks."
                    })
                elif total_non_compliant > 0:
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"Regulation Check Complete! \n\n{total_non_compliant} regulation(s) found non-compliant. Would you like detailed explanations for all violations?"
                    })
                    st.session_state.awaiting_explanation_response = True
                else:
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": "Compliance check could not be completed for all regulations."
                    })
            st.rerun()
   
    with col2:
        if st.button("Skip Other Regulations"):
            st.session_state.awaiting_other_regulation_check = False
            # Check if the initial regulation was non-compliant
            if any("NON-COMPLIANT" in msg["content"] for msg in st.session_state.messages if msg["role"] == "assistant"):
                st.session_state.non_compliant_regulations = [regulation]  # Current regulation only
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"Would you like an explanation for the non-compliant {regulation} result?"
                })
                st.session_state.awaiting_explanation_response = True
            else:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"Analysis complete for {regulation} regulations only."
                })
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
                    # Use the simple explanation function
                    explanation = explain_compliance(st.session_state.current_claim, reg, "no", backend, gemini_api_key)
                    
                    if explanation:
                        explanation_msg = f"{reg} Violation Explanation:\n{explanation}\n"
                        st.session_state.messages.append({"role": "assistant", "content": explanation_msg})
                        explanations.append(f"{reg}: {explanation}")
                
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
                        chr(10).join(st.session_state.current_explanations),
                        backend
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
                    
                    final_response = ask_llm(final_prompt, backend)
                    
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
        st.session_state.current_claim = user_input
       
        with st.spinner("Checking compliance..."):
            try:
                # Use the simple classification function
                compliant = classify_claim(user_input, regulation, backend, gemini_api_key)
               
                # Add section separator with yellow color
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": '<div style="background-color: #FFFF00; color: black; padding: 8px; border-radius: 5px; font-weight: bold;">INITIAL COMPLIANCE ASSESSMENT</div>'
                })
                
                if compliant == "yes":
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"COMPLIANT with {regulation} regulations"
                    })
                elif compliant == "no":
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"NON-COMPLIANT with {regulation} regulations"
                    })
                else:
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"COMPLIANCE STATUS UNKNOWN - Please check your API key or model connectivity"
                    })
                
                st.session_state.awaiting_other_regulation_check = True
                   
            except Exception as e:
                st.error(f"Error processing claim: {str(e)}")
       
        st.rerun()


st.markdown("---")
st.markdown("Tip: Try different medical claims to see how the compliance checker works!")