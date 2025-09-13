from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from dotenv import load_dotenv
from langchain.agents.agent_types import AgentType
load_dotenv()

llm = ChatOpenAI(model="gpt-4o", temperature=0)
agent_executor = create_csv_agent(
    llm,
    "data.csv",
    # agent_type="zero-shot-react-description",
    verbose=True,
    allow_dangerous_code=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True
)

response = agent_executor.run("Show monthly sales trend for 2023.")

print(response)


