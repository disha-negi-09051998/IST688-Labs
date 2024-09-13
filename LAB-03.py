import streamlit as st
from openai import OpenAI
import tiktoken  # Tokenizer from OpenAI

# Set a maximum token limit for the buffer (you can adjust this based on your needs).
max_tokens = 2048

# Function to calculate tokens for a message using OpenAI tokenizer
def calculate_token_count(messages, model_name="gpt-4o"):
    encoding = tiktoken.encoding_for_model(model_name)
    total_tokens = 0
    for message in messages:
        total_tokens += len(encoding.encode(message["content"]))
    return total_tokens

# Truncate conversation history to fit within max_tokens
def truncate_messages_by_tokens(messages, max_tokens, model_name="gpt-4o"):
    encoding = tiktoken.encoding_for_model(model_name)
    total_tokens = 0
    truncated_messages = []

    # Always retain the last user-assistant pair
    recent_pair = messages[-2:] if len(messages) >= 2 else messages

    # Calculate the token count for the most recent pair
    for message in recent_pair:
        total_tokens += len(encoding.encode(message["content"]))

    # Traverse the older messages in reverse order (newest to oldest)
    for message in reversed(messages[:-2]):  # Exclude the most recent pair
        message_token_count = len(encoding.encode(message["content"]))

        # Add message if it doesn't exceed the max_tokens limit
        if total_tokens + message_token_count <= max_tokens:
            # Insert the message at the beginning
            truncated_messages.insert(0, message)
            total_tokens += message_token_count
        else:
            break  # Stop if adding the next message would exceed the token limit

    # Combine older truncated messages with the recent pair
    truncated_messages.extend(recent_pair)

    return truncated_messages, total_tokens


# Show title and description.
st.title("LAB 03 -- Disha Negi 📄 Document question answering and Chatbot")
st.write(
    "Upload a document below and ask a question about it – GPT will answer! "
    "You can also interact with the chatbot. "
    "To use this app, you need to provide an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys). "
)

# Fetch the OpenAI API key from Streamlit secrets
openai_api_key = st.secrets["openai_api_key"]

if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.", icon="🗝️")
else:
    # Create an OpenAI client
    client = OpenAI(api_key=openai_api_key)

    # Let the user upload a file via ⁠ st.file_uploader ⁠.
    uploaded_file = st.file_uploader(
        "Upload a document (.txt or .md)", type=("txt", "md"))

    # Sidebar options for summarizing
    st.sidebar.title("Options")

    # Model selection
    openAI_model = st.sidebar.selectbox(
        "Choose the GPT Model", ("mini", "regular"))
    model_to_use = "gpt-4o-mini" if openAI_model == "mini" else "gpt-4o"

    # Summary options
    summary_options = st.sidebar.radio(
        "Select a format for summarizing the document:",
        (
            "Summarize the document in 100 words",
            "Summarize the document in detailed format",
            "Summarize the document in 5 bullet points"
        ),
    )

    if uploaded_file:
        # Process the uploaded file
        document = uploaded_file.read().decode()

        # Instruction based on user selection on the sidebar menu
        instruction = f"Summarize the document in {summary_options.lower()}."

        # Prepare the messages for the LLM
        messages = [
            {
                "role": "user",
                "content": f"Here's a document: {document} \n\n---\n\n {instruction}",
            }
        ]

        # Generate the summary using the OpenAI API
        stream = client.chat.completions.create(
            model=model_to_use,
            messages=messages,
            stream=True,
        )

        # Stream the summary response to the app
        st.write_stream(stream)

    # Set up the session state to hold chatbot messages with a token-based buffer
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = [
            {"role": "assistant", "content": "How can I help you?"}
        ]

    # Display the chatbot conversation
    st.write("## Chatbot Interaction")
    for msg in st.session_state.chat_history:
        chat_msg = st.chat_message(msg["role"])
        chat_msg.write(msg["content"])

    # Get user input for the chatbot
if prompt := st.chat_input("Ask the chatbot a question or interact:"):
    # Append the user input to the session state
    st.session_state.chat_history.append({"role": "user", "content": prompt})

    # Display the user input in the chat
    with st.chat_message("user"):
        st.markdown(prompt)

    # Calculate the token count and truncate if necessary
    truncated_messages, total_tokens = truncate_messages_by_tokens(
        st.session_state.chat_history, max_tokens, model_name=model_to_use
    )
    st.session_state.chat_history = truncated_messages

    # Generate a response from OpenAI using the same model with a simple and concise prompt
    simple_prompt = f"Answer in a way that a 10-year-old can understand: {prompt}"
    messages_for_gpt = st.session_state.chat_history.copy()
    messages_for_gpt[-1]['content'] = simple_prompt  # Replace the latest user message with the simplified prompt

    stream = client.chat.completions.create(
        model=model_to_use,
        messages=messages_for_gpt,
        stream=True,
    )

    # Stream the assistant's response
    with st.chat_message("assistant"):
        response = st.write_stream(stream)

    # Append the assistant's response to the session state
    st.session_state.chat_history.append(
        {"role": "assistant", "content": response})

    # Handle the user's "yes/no" responses
    if "yes" in prompt.lower():
        # If the user says "yes," provide more info and ask again
        st.session_state.chat_history.append(
            {"role": "assistant", "content": "Here's more information. Do you want more info?"}
        )
        with st.chat_message("assistant"):
            st.markdown("Here's more information. Do you want more info?")
    elif "no" in prompt.lower():
        # If the user says "no," ask what else the bot can help with
        st.session_state.chat_history.append(
            {"role": "assistant", "content": "What question can I help with next?"}
        )
        with st.chat_message("assistant"):
            st.markdown("What question can I help with next?")
    else:
        # Default follow-up question
        follow_up_question = "Do you want more info?"
        st.session_state.chat_history.append(
            {"role": "assistant", "content": follow_up_question})

        # Display the follow-up question
        with st.chat_message("assistant"):
            st.markdown(follow_up_question)

    # Reapply truncation to ensure the history fits within the token limit
    truncated_messages, _ = truncate_messages_by_tokens(
        st.session_state.chat_history, max_tokens, model_name=model_to_use
    )
    st.session_state.chat_history = truncated_messages

    # Display token usage
    st.write(f"Updated total tokens after response: {total_tokens}")
    st.write(f"Messages retained after response: {truncated_messages}")