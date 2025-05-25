import os, time, logging
import streamlit as st
from llm.tools.lmodel_access import LModelAccess
from llm.tools.prompt_utils import PromptUtils
from streamlit.runtime.scriptrunner import get_script_run_ctx


sb_initial_state = "expanded"
openai_api_key = st.secrets.openrouter_api_key
mu = LModelAccess(openai_api_key)
models = mu.get_all_models()
selected_model = models[0]
log = st.logger.get_logger(__name__)
avatar_lkp = ({
    "Male" : "images/man.png",
    "Female" : "images/woman.png",
    "Hacker" : "images/hacker.png",
})

def app_setup():
    st.set_page_config(
        page_title="LLM Chatbot",
        page_icon=":earth_americas:",
        layout="wide",
        initial_sidebar_state=sb_initial_state,
    )
    init_sidebar()

def init_sidebar():
    with st.sidebar.expander(":blue[Chat Settings]", expanded=True):
        selected_model = st.selectbox("Model:", 
                                      models, key="active_model", 
                                      help="Choose an Open Source Model", 
                                      on_change=model_change)
        st.write(f"Active Model:  ***{selected_model}***")
        st.session_state.llm = mu.get_llm(selected_model, temperature=0.0)

        st.radio("Avatar:", 
            options=["Male", "Female", "Hacker"], 
            index=0, 
            horizontal=True,
            key="active_avatar",
            help="Choose your avatar",
            on_change=avatar_change,
        )
        st.session_state.user_avator = avatar_lkp[st.session_state.active_avatar]

def avatar_change():
    """
    Callback function to handle avatar change in the sidebar.
    """
    log.info(f"avatar_change to => {st.session_state.active_avatar}")
    # Update avatar based on user selection
    st.session_state.user_avator = avatar_lkp[st.session_state.active_avatar]

def model_change():    
    """
    Callback function to handle model change in the sidebar.
    """
    log.info(f"model_change to => {st.session_state.active_model}")
    # Reinitialize llm with chosen model
    st.session_state.llm = mu.get_llm(st.session_state.active_model, temperature=0.0) 

def get_session_id():
    session_id = get_script_run_ctx().session_id
    return session_id

def get_response(llm, user_prompt, session_id):             
    """
    Generate response from the LLM using the provided user prompt and session ID.
    Args:
        llm (ChatOpenAI): The LLM instance.
        user_prompt (str): The user's input prompt.
        session_id (str): The session ID for tracking.
    Returns:
        str: The generated response from the LLM.
    """
    prompt = PromptUtils.get_prompt(user_prompt)
    log.debug(f"Prompt: {prompt}")
    #chain = prompt | llm
    messages = llm.invoke(prompt)
    #messages = chain.invoke()
    return messages.content

# ------------------------
# 
#       Main App
# 
# ------------------------
def main():
    bot_avator = "images/chat-bot.png"
    
    app_setup()
    session_id = get_session_id()
    log.info(f"Created Session ID: {session_id}")

    if "messages" not in st.session_state:
        st.session_state.messages = []
   
    if "active_model" in st.session_state:
        log.debug(f"active_model set in session => {st.session_state.active_model}")
        assert "llm" in st.session_state, "llm not set in session state!"

        # Display chat messages from history
        for message in st.session_state.messages:
            if message["role"] == "user":
                 with st.chat_message("user", avatar=st.session_state.user_avator):
                    st.markdown(message["content"])
            else:
                with st.chat_message("assistant", avatar=bot_avator):
                    st.markdown(message["content"])

        if prompt := st.chat_input("Ask me anything?"):
            # Add to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user", avatar=st.session_state.user_avator):
                st.markdown(prompt)

        # Get llm response
        response = get_response(st.session_state.llm, prompt, session_id)
        # Add to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant", avatar=bot_avator):
            st.markdown(response)

# end main()



if __name__ == "__main__":
    main()