import os

from langchain_community.chat_models import JinaChat
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts.chat import (ChatPromptTemplate,
                                         HumanMessagePromptTemplate,
                                         SystemMessagePromptTemplate)


def get_api_keys(api_name):
    API_KEY = os.getenv(f"{api_name}_API_KEY")

    if not API_KEY:
        with open(".env", "r") as f:
            for line in f:
                key, value = line.strip().split("=")
                if key == f"{api_name}_API_KEY":
                    API_KEY = value
                    
    if not API_KEY:
        raise ValueError(f"{api_name} API key not found")
    
    return API_KEY

JINA_API_KEY = get_api_keys("JINA")
JINACHAT_API_KEY = get_api_keys("JINACHAT")
OPENAI_API_KEY = get_api_keys("OPENAI")

# Initialize JinaChat with desired temperature
chat = JinaChat(temperature=0, jinachat_api_key=JINACHAT_API_KEY)

# Initialize JinaChat with desired temperature
chat = JinaChat(temperature=0, jinachat_api_key=JINACHAT_API_KEY)

# Define the template for the system message
template = (
    "You are a helpful assistant that translates {input_language} to {output_language}."
)
system_message_prompt = SystemMessagePromptTemplate.from_template(template)

# Define the template for the human message
human_template = "{text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

# Create a ChatPromptTemplate from the message templates
chat_prompt = ChatPromptTemplate.from_messages(
    [system_message_prompt, human_message_prompt]
)

# Get user input for text, input language, and output language
text = input("Enter the text you want to translate: ")
input_language = input("Enter the input language: ")
output_language = input("Enter the output language: ")

# Format the prompt with user inputs and convert to messages
messages = chat_prompt.format_prompt(
    input_language=input_language, output_language=output_language, text=text
).to_messages()

# Get the translation from JinaChat
response = chat(messages)

# Print the translated text
print(response.content)