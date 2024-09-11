import streamlit as st
from openai import OpenAI

# Show title and description.
st.title("LAB-02-Disha Negi📄 Document question answering")
st.write(
    "Upload a document below and ask a question about it – GPT will answer! "
    "To use this app, you need to provide an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys). "
)

# Ask user for their OpenAI API key via `st.text_input`.
# Alternatively, you can store the API key in `./.streamlit/secrets.toml` and access it
# via `st.secrets`, see https://docs.streamlit.io/develop/concepts/connections/secrets-management

# Fetch the OpenAI API key from Streamlit secrets.
openai_api_key = st.secrets["api_key"]

if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.", icon="🗝️")
else:

    # Create an OpenAI client.
    client = OpenAI(api_key=openai_api_key)

    # Let the user upload a file via `st.file_uploader`.
    uploaded_file = st.file_uploader(
        "Upload a document (.txt or .md)", type=("txt", "md")
    )

    # Sidebar options for summarization formats
    st.sidebar.title("Choose Summary Format")
    summary_options = st.sidebar.radio(
        "Select a format for summarizing the document:",
        (
            "Summarize the document in 100 words",
            "Summarize the document in 2 connecting paragraphs",
            "Summarize the document in 5 bullet points"
        ),
    )

    # Checkbox for model selection
    use_advanced_model = st.sidebar.checkbox("Use Advanced Model")
    
    # Select model based on checkbox status
    model = "gpt-4o" if use_advanced_model else "gpt-4o-mini"

    if uploaded_file:

        # Process the uploaded file
        document = uploaded_file.read().decode()

        # Instruction based on user selection
        instruction = f"Summarize the document in {summary_options.lower()}."

        # Prepare the messages for the LLM
        messages = [
            {
                "role": "user",
                "content": f"Here's a document: {document} \n\n---\n\n {instruction}",
            }
        ]

        # Generate the summary using the OpenAI API.
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
        )

        # Stream the response to the app using `st.write_stream`.
        st.write_stream(stream)

    # # Ask the user for a question via `st.text_area`.
    # question = st.text_area(
    #     "Now ask a question about the document!",
    #     placeholder="Can you give me a short summary?",
    #     disabled=not uploaded_file,
    # )

    # if uploaded_file and question:

    #     # Process the uploaded file and question.
    #     document = uploaded_file.read().decode()
    #     messages = [
    #         {
    #             "role": "user",
    #             "content": f"Here's a document: {document} \n\n---\n\n {question}",
    #         }
    #     ]

    #     # Generate an answer using the OpenAI API.
    #     stream = client.chat.completions.create(
    #         model="gpt-4o-mini",
    #         messages=messages,
    #         stream=True,
    #     )

    #     # Stream the response to the app using `st.write_stream`.
    #     st.write_stream(stream)
