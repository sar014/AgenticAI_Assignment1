import os
import json
import time
import streamlit as st
from groq import Groq
from tools import TOOL_SCHEMAS, dispatch_tool
from prompts import SYSTEM_PROMPT
from dotenv import load_dotenv
from tools import get_user_preferences

load_dotenv()

# Ensure traces directory exists for logging evaluation data
os.makedirs("traces", exist_ok=True)

# Initialize Groq Client 
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
MODEL_NAME = "openai/gpt-oss-120b"

def run_agent_loop(user_query: str, chat_history: list, max_turns=5):
    """Executes the manual ReAct loop with per-turn token and metric tracing."""
    start_time = time.time()
    retrieved_evidence = []
    
    print(f"\n🔹 [BACKEND] run_agent_loop started for query: '{user_query}'")
    try:
        # Load persistent non-sensitive preferences
        saved_prefs = get_user_preferences()
        context_injection = f"\n[Persistent User Profile / Memory]: {json.dumps(saved_prefs)}" if saved_prefs else ""
        if saved_prefs:
            retrieved_evidence.append(f"Memory Loaded: {saved_prefs}")

        # Inject persistent memory into the system prompt
        dynamic_system_prompt = SYSTEM_PROMPT + context_injection

        # 1. Initialize messages with the system prompt
        messages = [{"role": "system", "content": dynamic_system_prompt}]
        
        # 2. Safely replay chat history, ensuring no 'None' content is passed
        for msg in chat_history:
            role = msg.get("role")
            content = msg.get("content")
            
            if content is None:
                content = ""
                
            clean_msg = {"role": role, "content": content}
            
            if "tool_calls" in msg and msg["tool_calls"]:
                clean_msg["tool_calls"] = msg["tool_calls"]
            if "tool_call_id" in msg:
                clean_msg["tool_call_id"] = msg["tool_call_id"]
            if "name" in msg:
                clean_msg["name"] = msg["name"]
                
            messages.append(clean_msg)
            
        # 3. Append current user query
        if not user_query:
            user_query = "Hello"
            
        messages.append({"role": "user", "content": user_query})
        turn_trace = []

        for turn in range(max_turns):
            print(f"🔸 [BACKEND] Turn {turn + 1}: Sending request to Groq LLM...")
            
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                tools=TOOL_SCHEMAS,
                temperature=0
            )

            print("DEBUG RESPONSE USAGE:", getattr(response, 'usage', 'NO USAGE ATTRIBUTE'))

            # Extract token usage for this specific turn
            turn_prompt_tokens = 0
            turn_completion_tokens = 0
            if hasattr(response, 'usage') and response.usage:
                turn_prompt_tokens = response.usage.prompt_tokens
                turn_completion_tokens = response.usage.completion_tokens
                print(f"DEBUG TURN {turn + 1} USAGE: prompt_tokens={turn_prompt_tokens}, completion_tokens={turn_completion_tokens}")

            response_message = response.choices[0].message
            print(f"🔸 [BACKEND] Received response from LLM.")
            
            # Check if model wants to call tools
            if response_message.tool_calls:
                print(f"🛠️ [BACKEND] Agent requested tool calls: {[tc.function.name for tc in response_message.tool_calls]}")
                
                messages.append({
                    "role": "assistant",
                    "content": response_message.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        } for tc in response_message.tool_calls
                    ]
                })
                
                for tool_call in response_message.tool_calls:
                    call_name = tool_call.function.name
                    call_args = json.loads(tool_call.function.arguments)
                    
                    print(f"⚙️ [BACKEND] Executing tool '{call_name}' with args: {call_args}")
                    tool_result = dispatch_tool(call_name, call_args)
                    print(f"📊 [BACKEND] Tool result observation: {tool_result}")
                    
                    # Track retrieved policy chunks if RAG search tool was used
                    if call_name == "search_student_policy" and isinstance(tool_result, dict):
                        retrieved_evidence.extend(tool_result.get("relevant_context", []))

                    # Log turn details including per-turn tokens
                    turn_trace.append({
                        "turn": turn + 1,
                        "tool_called": call_name,
                        "arguments": call_args,
                        "observation": tool_result,
                        "prompt_tokens": turn_prompt_tokens,
                        "completion_tokens": turn_completion_tokens
                    })
                    
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": call_name,
                        "content": json.dumps(tool_result)
                    })
            else:
                final_answer = response_message.content
                print(f"✅ [BACKEND] Final LLM Output: {final_answer}")
                
                end_time = time.time()
                latency = round(end_time - start_time, 2)
                
                # If final answer happened without tool calls on this turn, log it too
                turn_trace.append({
                    "turn": turn + 1,
                    "tool_called": None,
                    "arguments": None,
                    "observation": "Direct Answer",
                    "prompt_tokens": turn_prompt_tokens,
                    "completion_tokens": turn_completion_tokens
                })

                # Save trace log
                trace_record = {
                    "user_query": user_query,
                    "latency_seconds": latency,
                    "turns_taken": turn + 1,
                    "trace": turn_trace,
                    "retrieved_evidence": retrieved_evidence,
                    "final_answer": final_answer
                }
                
                with open("traces/run_logs.jsonl", "a") as f:
                    f.write(json.dumps(trace_record) + "\n")
                    
                return final_answer, turn_trace

        return "Max turns reached without a final resolution.", turn_trace

    except Exception as e:
        print(f"❌ [CRITICAL ERROR] Exception caught: {str(e)}")
        return f"⚠️ **Backend Error Occurred:** `{str(e)}`", []

# ==========================================
# 1. PAGE TITLE & HEADER (Always at the top)
# ==========================================
st.title("🎓 AIDTM Intelligent Student Support Agent")
st.markdown("Welcome! You can ask questions about your fee status (e.g., *'Check fees for STU-101'*) or institutional admission policies (*'What is the refund policy?'*).")

st.divider()

# ==========================================
# 2. INITIALIZE SESSION STATE MESSAGES
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! How can I assist you today—perhaps with admission eligibility, program details, fee schedules, refund policies, or a specific student's account?"
        }
    ]

# ==========================================
# 3. RENDER CHAT HISTORY (Middle section)
# ==========================================
for message in st.session_state.messages:
    if not message.get("content") or message.get("content") == "None":
        continue
        
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ==========================================
# 4. SINGLE CHAT INPUT (Rendered once at bottom)
# ==========================================
if prompt := st.chat_input("Type your question here...", key="unique_student_chat_input"):
    if prompt.strip() and prompt.strip().lower() != "none":
        print(f"\n💬 [UI] User submitted prompt: '{prompt}'")
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Agent is reasoning and checking records..."):
                answer, trace = run_agent_loop(prompt, st.session_state.messages[:-1])
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})