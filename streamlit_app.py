import os, time, logging, json, base64
import streamlit as st
from llm.tools.lmodel_access import LModelAccess
from llm.tools.prompt_utils import PromptUtils
from streamlit_oauth import OAuth2Component
from streamlit.runtime.scriptrunner import get_script_run_ctx


app_name = "LLM Chatbot"
app_dns = "https://chat-open-router.streamlit.app/"
openai_api_key = st.secrets.openrouter_api_key
log = st.logger.get_logger(__name__)
mu = LModelAccess(app_name, app_dns, openai_api_key)
models = mu.get_all_models()
selected_model = models[0]
sb_initial_state = "expanded"

avatar_lkp = ({
    "Male" : "images/man.png",
    "Female" : "images/woman.png",
    "Hacker" : "images/hacker.png",
})

def app_setup():
    st.set_page_config(
        page_title=app_name,
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

def get_user_info(id_token):
    """
    Get user information from the JWT ID token.
    Args:
        id_token (str): The JWT ID token containing claims aboaut a user's identity.
    Returns:
        email (str): Authenticated User email address.
    """
    # verify signature for security
    payload = id_token.split(".")[1]
    # add padding to the payload, if required
    payload += "=" * (-len(payload) % 4)
    payload = json.loads(base64.b64decode(payload))
    email = payload["email"]
    return email

AUTHORIZATION_URL = "https://accounts.google.com/o/oauth2/v2/auth"
# used to exchange the authorization code for an access token
TOKEN_URL = "https://oauth2.googleapis.com/token"
# used to revoke the access token when the user logs out
REVOKE_URL = "https://oauth2.googleapis.com/revoke"
CLIENT_ID = st.secrets.OAUTH_CLIENT_ID
CLIENT_SECRET = st.secrets.OAUTH_CLIENT_SECRET
REDIRECT_URI = app_dns
#REDIRECT_URI = "http://localhost:8501/"
SCOPE = "openid email profile"

oauth2 = OAuth2Component(CLIENT_ID, CLIENT_SECRET, AUTHORIZATION_URL, TOKEN_URL, TOKEN_URL, REVOKE_URL)

# ------------------------
# 
#       Main App
# 
# ------------------------
def main():
    bot_avator = "images/chat-bot.png"
    
    if 'token' not in st.session_state:
        result = oauth2.authorize_button("Continue with Google", 
                                         REDIRECT_URI, SCOPE, 
                                         icon="data:image/svg+xml;charset=utf-8,%3Csvg xmlns='http://www.w3.org/2000/svg' xmlns:xlink='http://www.w3.org/1999/xlink' viewBox='0 0 48 48'%3E%3Cdefs%3E%3Cpath id='a' d='M44.5 20H24v8.5h11.8C34.7 33.9 30.1 37 24 37c-7.2 0-13-5.8-13-13s5.8-13 13-13c3.1 0 5.9 1.1 8.1 2.9l6.4-6.4C34.6 4.1 29.6 2 24 2 11.8 2 2 11.8 2 24s9.8 22 22 22c11 0 21-8 21-22 0-1.3-.2-2.7-.5-4z'/%3E%3C/defs%3E%3CclipPath id='b'%3E%3Cuse xlink:href='%23a' overflow='visible'/%3E%3C/clipPath%3E%3Cpath clip-path='url(%23b)' fill='%23FBBC05' d='M0 37V11l17 13z'/%3E%3Cpath clip-path='url(%23b)' fill='%23EA4335' d='M0 11l17 13 7-6.1L48 14V0H0z'/%3E%3Cpath clip-path='url(%23b)' fill='%2334A853' d='M0 37l30-23 7.9 1L48 0v48H0z'/%3E%3Cpath clip-path='url(%23b)' fill='%234285F4' d='M48 48L17 24l-4-3 35-10z'/%3E%3C/svg%3E"
                                         )
        if result:
            st.session_state.token = result.get('token')
            log.debug(st.session_state.token)
            # decode JWT id_token jwt containing user auth info (email)
            id_token = st.session_state.token.get("id_token")
            st.session_state.auth_email = get_user_info(id_token)
            st.rerun()
    else:
        log.info(f"User {st.session_state.auth_email} is already authenticated with Google OAuth2")
        session_id = get_session_id()
        log.info(f"Created Session ID: {session_id}")
        app_setup()

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