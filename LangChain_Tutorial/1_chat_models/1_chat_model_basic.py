# Chat Model Documents: https://python.langchain.com/v0.2/docs/integrations/chat/
# OpenAI Chat Model Documents: https://python.langchain.com/v0.2/docs/integrations/chat/openai/
# Fireworks Chat Model Documents: https://python.langchain.com/v0.2/docs/integrations/chat/fireworks/

from dotenv import load_dotenv
from langchain_fireworks import ChatFireworks

# Load environment variables from .env
load_dotenv()

# Create a ChatOpenAI model
model = ChatFireworks(model= "accounts/fireworks/models/mixtral-8x7b-instruct")

# Invoke the model with a message
result = model.invoke("What is 81 divided by 9?")
print("Full result:")
print(result)
print("Content only:")
print(result.content)