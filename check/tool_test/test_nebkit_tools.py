from scripted_model import ScriptedModel
from langchain.agents import create_agent 
from langgraph.checkpoint.memory import InMemorySaver
from langchain.messages import AIMessage
from langchain_core.messages.tool import ToolCall
from catagent.tools.neb.reaction_system import ReactionSystemToolSets
import asyncio


tools = ReactionSystemToolSets().create()

model = ScriptedModel(responses = [
    

    AIMessage(
        content="call tool",
        tool_calls = [
            ToolCall(
                name = "create_reaction_system",
                args = {
                    "system_name": "CH4+CO2",
                    "reactants_mol_pathes": ["./CH4.xyz", "./CO2.xyz"],
                    "products_mol_pathes": ["./CO.xyz", "./H2.xyz"],
                    "json_log_path": "./CH4-CO2.json",
                    "visualization_dir": "./visual"
                },
                id="call1"
            )

        ]
    ),
    

    AIMessage(
            content="call tool",
            tool_calls = [
                ToolCall(
                    name = "generate_elementary_steps",
                    args = {
                        "system_name": "CH4+CO2",
                        "ntimes": -1,
                        "max_natoms": {'C': 1, 'O': 2},
                        "forbidden_pattern": [('O,O', '0-1'), ('H,O,C,O,H', '0-1,1-2,2-3,3-4')]
                    },
                    id="call2"
                )
    
            ]
        ),

    AIMessage(
        content="call tool",
        tool_calls = [
            ToolCall(
                name='render',
                args = {
                    "system_name": "CH4+CO2",
                    "backend": "rdkit"
                },
                id="call3"
            )
        ]
    ),

    AIMessage(content="Hello"),
    AIMessage(content="Hello2"),
    AIMessage(content="Hello3"),
    AIMessage(content="Hello4"),
    AIMessage(content="Hello5"),

])




agent = create_agent(model=model,
                     tools = tools)


async def run_agent():
    async for msg in agent.astream(input=dict(), 
                                   config={"configurable": {"thread_id": "temp-123"}}):
        print("+++++++++++++++++++")
        print(msg)

asyncio.run(run_agent())