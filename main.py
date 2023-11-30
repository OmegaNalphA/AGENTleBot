from agent.agent import Agent
from agent.models import CreateAgentParams
from dotenv import load_dotenv
import sys

load_dotenv()

if __name__ == "__main__":
    objective = sys.argv[1]
    if len(sys.argv) > 2:
        log = sys.argv[2].lower() == "true"
    agent = Agent(CreateAgentParams(objective=objective, log=True))
    agent.run()
