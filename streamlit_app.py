import streamlit as st
import asyncio
import nest_asyncio
import uuid
import logging
from typing import List, Dict, Any

# Patch event loop for Streamlit/Asyncio compatibility
nest_asyncio.apply()

# Import agent components
from agent.agent import rag_agent, AgentDependencies
from agent.db_utils import (
    initialize_database,
    close_database,
    create_session,
    get_session_messages,
    add_message,
    db_pool
)
from agent.graph_utils import initialize_graph, close_graph, graph_client
from agent.models import ToolCall

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Agentic RAG Knowledge Graph",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.user_id = "streamlit_user"

if "messages" not in st.session_state:
    st.session_state.messages = []

if "initialized" not in st.session_state:
    st.session_state.initialized = False

# Sidebar
with st.sidebar:
    st.title("🤖 Agent Settings")
    st.markdown("---")
    
    st.subheader("Search Preferences")
    use_vector = st.checkbox("Use Vector Search", value=True)
    use_graph = st.checkbox("Use Graph Search", value=True)
    
    if st.button("New Session"):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()
        
    st.markdown("---")
    st.caption(f"Session ID: {st.session_state.session_id}")

# Helper to run async function in a new loop properly
def run_async(coro):
    try:
        return asyncio.run(coro)
    except RuntimeError:
        # If we are already in a loop (e.g. jupyter or some streamlit servers), try to use it
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(coro)

# Async agent execution with precise resource management
async def process_chat(prompt: str):
    # FORCE RESET GLOBALS to ensure we use the current loop's connection
    # This is critical because db_utils.db_pool is a global that persists across Streamlit runs,
    # but the asyncio loop is recreated.
    if db_pool.pool:
        try:
            await db_pool.pool.close()
        except:
            pass
        db_pool.pool = None
    
    if graph_client.graphiti:
        try:
            await graph_client.graphiti.close()
        except:
            pass
        graph_client.graphiti = None
        graph_client._initialized = False

    try:
        # 1. Initialize fresh connections on the CURRENT loop
        await initialize_database()
        await initialize_graph()
        
        # 2. Ensure session exists
        await create_session(
            user_id=st.session_state.user_id,
            metadata={"source": "streamlit"},
            session_id=st.session_state.session_id
        )

        # 3. Execution
        deps = AgentDependencies(
            session_id=st.session_state.session_id,
            user_id=st.session_state.user_id,
            search_preferences={
                "use_vector": use_vector,
                "use_graph": use_graph,
                "default_limit": 10
            }
        )
        
        # Build context
        context_str = ""
        # We need to manually fetch messages because we just reset the DB connection
        # and checking st.session_state is unsafe as it might be out of sync with what the agent needs
        # But for prompt construction, st.session_state is fine.
        if len(st.session_state.messages) > 1:
            recent_msgs = st.session_state.messages[:-1][-5:]
            context_str = "\n".join([f"{m['role']}: {m['content']}" for m in recent_msgs])
        
        full_prompt = prompt
        if context_str:
            full_prompt = f"Previous conversation:\n{context_str}\n\nCurrent question: {prompt}"

        # Run Agent
        result = await rag_agent.run(full_prompt, deps=deps)
        response_text = result.data
        
        # Save to DB
        await add_message(st.session_state.session_id, "user", prompt, {})
        await add_message(st.session_state.session_id, "assistant", response_text, {})
        
        return response_text
        
    finally:
        # 4. Cleanup to prevent loop leaks
        await close_database()
        await close_graph()

# Initial loading of messages (one-off)
if not st.session_state.initialized:
    async def load_history():
        # Temporary connection just for history
        await initialize_database()
        msgs = await get_session_messages(st.session_state.session_id)
        await close_database()
        return msgs
        
    with st.spinner("Restoring session history..."):
        try:
            # Force reset pool if it exists from a previous run
            # We cannot await close() here because we are in sync context and the old loop might be dead
            if db_pool.pool:
                db_pool.pool = None
                
            history = run_async(load_history())
            if history:
                st.session_state.messages = [
                    {"role": msg["role"], "content": msg["content"]}
                    for msg in history
                ]
            st.session_state.initialized = True
        except Exception as e:
            st.warning(f"Could not restore history: {e}")
            st.session_state.initialized = True


# Main chat interface
st.title("Agentic RAG Knowledge Graph")
st.markdown("Ask questions about the tech companies in the knowledge graph.")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What would you like to know?"):
    # Add user message to state and display
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process with agent
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        try:
            with st.spinner("Thinking..."):
                # Run the full process in a single async run
                response = run_async(process_chat(prompt))
                
                message_placeholder.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                
        except Exception as e:
            st.error(f"Error generating response: {e}")
            logger.error(f"Agent error: {e}", exc_info=True)
