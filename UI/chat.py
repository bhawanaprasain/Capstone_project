import streamlit as st
import requests

class ChatApp:
    """
    A simple chat application using Streamlit.
    """

    def __init__(self, api_url):
        """
        Initializes the ChatApp class.

        Parameters:
        - api_url (str): The API endpoint for making chat requests.
        """
        self.api_url = api_url
        self.session_state = st.session_state

        if "messages" not in self.session_state:
            self.session_state.messages = []

    def make_api_request(self, prompt):
        """
        Make API request using the given prompt.

        Parameters:
        - prompt (str): The user's input prompt.

        Returns:
        - str: The assistant's response.
        """
        payload = {
            "client_name": "",
            "question": prompt,
            "history": self.session_state.messages
        }

        api_response = requests.post(self.api_url, json=payload)
        response_content = api_response.json()[0]['message']
        return response_content

    def display_chat_history(self):
        """
        Display chat messages from history on app rerun.
        """
        for message in self.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    def run_app(self, prompt_prompt="What is up?"):
        """
        Run the simple chat application.

        Parameters:
        - prompt_prompt (str): The prompt to be displayed in the chat input.
        """
        st.title("AI for healthcare")

        self.display_chat_history()

        if prompt := st.chat_input(prompt_prompt):
            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(prompt)

            # Make API request
            response_content = self.make_api_request(prompt)

            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                st.markdown(response_content)

            # Add user and assistant messages to chat history
            self.session_state.messages.append({"role": "user", "content": prompt})
            self.session_state.messages.append({"role": "assistant", "content": response_content})

# Instantiate the ChatApp class with the API endpoint
chat_app = ChatApp(api_url="")
# Run the app
chat_app.run_app()
