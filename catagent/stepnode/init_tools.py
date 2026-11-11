from langchain.chat_models import BaseChatModel
from langgraph.types import Command
from langchain.tools import tool, ToolRuntime
from langchain.messages import ToolMessage
from langgraph.prebuilt import ToolNode
from langchain_core.tools.base import BaseTool
from langchain.agents.middleware import ToolCallRequest

from tools.other.incar_tool import (create_get_incar_draft_tool, 
                                    create_extract_user_defined_parameters_tool)

from tools.neb.neb_tool import (sampling_ini_fin_states_on_surface, 
                                build_initial_nebpath_on_surface, 
                                sampling_single_ads_configs_on_surface)
from .skill_tool import get_skill_summaries, get_skill_content

from tools.neb.reaction_system import ReactionSystemToolSets
from tools.other.util_tools import execute_shell, edit_file, read_file
from .models import ToolNamespace, AgentInterrupt, JobSubmission, AskForArgs
from typing import List,  Dict,  Any, Literal, Optional
import httpx, uuid 

TOOL_COLLECTIONS: Dict[str, ToolNamespace] = dict()
TOOLS_BY_NAME: Dict[str, BaseTool] = dict()


def init_job_manager_tools(job_manager_url: str, http_client:httpx.AsyncClient):

    @tool 
    async def get_providers_info():
        '''
        Return a list of provider informations. 
        Returns:
            A list, each item is a dict containing 4 fields:
            - name: Name of provider.
            - processing_units: Processing units provided by the provider, can be "cpu" and "gpu".
            - job_envs: A dict describing job environments. Key is the name of job environment, value is a list of softwares.
            - remote_files: A list of description of remote files provided by the provider.
        '''
        response = await http_client.get(f"{job_manager_url}/providers") 
        response.raise_for_status()
        return response.json()
    
    @tool 
    async def submit_job(runtime:ToolRuntime,
                         job_description: str,
                         local_dir: str, 
                         provider_name: str,
                         job_env_name: str,
                         result_patterns: Optional[List[str]] = None ):
        '''
        Submit a job to job manager.
        Args:
            job_description: Description of the job.
            local_dir: The local workdir where the files to be submitted exist.
            provider_name: The name of selected provider.
            job_env_name: The name of selected job environment.
            result_patterns: The file name or wildcard of the result to be recycled. Download all the calculation results by default.
        '''
        job_uuid = str(uuid.uuid4())
        payload = {
            "job_uuid":job_uuid ,
            "description": job_description,
            "local_dir": local_dir,
            "provider_id": provider_name,
            "job_env_name": job_env_name,
            # The job manager requires a list.  Models often omit optional
            # arguments, so normalize None to an empty list to avoid a 422.
            "result_patterns": result_patterns or []
            } 

        metadata = {k: v for k, v in payload.items() if k != "job_uuid"}

        def submission_failed(message: str) -> Command:
            return Command(
                update={
                    "messages": [ToolMessage(
                        status="error",
                        name="submit_job",
                        content=f"Job submission failed. \nError message: {message}. \nMetadata: {metadata}",
                        tool_call_id=runtime.tool_call_id
                    )]
                }
            )

        try:
            response = await http_client.post(f"{job_manager_url}/submit", json=payload)
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            # Include the response body: FastAPI puts validation details there
            # (for example which request field caused a 422).
            detail = (e.response.text or "").strip()
            if len(detail) > 4000:
                detail = detail[:4000] + "..."
            message = f"{e}"
            if detail:
                message += f"\nResponse body: {detail}"
            return submission_failed(message)
        except httpx.RequestError as e:
            return submission_failed(
                f"Could not reach the job manager at {job_manager_url}: {e}"
            )
        except Exception as e:
            return submission_failed(f"Unexpected error while submitting the job: {e}")

        try:
            data = response.json()
        except ValueError as e:
            return submission_failed(
                "The job manager returned an invalid JSON response. "
                f"The job may already have been accepted. Error: {e}"
            )

        if not isinstance(data, dict):
            return submission_failed(
                f"The job manager returned an unexpected response: {data!r}"
            )

        if data.get("status") == "error":
            return submission_failed(data.get("message", "Unknown job manager error."))

        obj = JobSubmission(pending=True,
                            message=f"Job {job_uuid} is submitted.",
                            tool_call_id=runtime.tool_call_id,
                            tool_name="submit_job",
                            job_uuid=job_uuid)
        return Command(
            update={
                "messages": [ToolMessage(status="success", 
                                         name="submit_job",
                                         content=f"Job is submitted.\n Metadata: {metadata}",
                                         tool_call_id=runtime.tool_call_id)],
                "interrupts": {runtime.tool_call_id: obj.model_dump(mode='python')}
            }
        )

    return [submit_job, get_providers_info]


def init_common_tools(model:BaseChatModel):
    collection = ReactionSystemToolSets().create()
    name_list = []
    for item in collection:
        if item.name in TOOLS_BY_NAME:
            raise RuntimeError(f"Tool {item.name} is already exist.")
        TOOLS_BY_NAME[item.name] = item 
        name_list.append(item.name)
    
    TOOL_COLLECTIONS["reaction_system_tools"] = ToolNamespace(
        description="Nebkit wrapper tools. Generate elementary reactions, edit reaction network. Given the surface structures, create initial and final states of all the generated reactions in batch.",
        tool_names=name_list
    )

    collection = [create_extract_user_defined_parameters_tool(model), 
                  create_get_incar_draft_tool(model)]
    name_list = []
    for item in collection:
        if item.name in TOOLS_BY_NAME:
            raise RuntimeError(f"Tool {item.name} is already exist.")
        TOOLS_BY_NAME[item.name] = item 
        name_list.append(item.name)
    TOOL_COLLECTIONS["vasp_incar_tools"] = ToolNamespace(
        description="Tools for generating vasp INCAR draft.",
        tool_names=name_list)

    collection = [sampling_ini_fin_states_on_surface, 
                  build_initial_nebpath_on_surface,
                  sampling_single_ads_configs_on_surface]
    name_list = []
    for item in collection:
        if item.name in TOOLS_BY_NAME:
            raise RuntimeError(f"Tool {item.name} is already exist.")
        TOOLS_BY_NAME[item.name] = item 
        name_list.append(item.name)
    TOOL_COLLECTIONS["structure_building_tools"] = ToolNamespace(
        description="Nebkit wrapper tools. Build initial and final states on a surface. Bind an adsorbate to a surface. Build the initial NEB paths. Use these tools for small-scale structure building tasks.",
        tool_names = name_list)

    
    @tool 
    def get_tool_collection_infos() -> Dict[str, str]:
        '''
        Return the descriptions of tool collections.
        Returns:
            A dict. Key is the name of a tool collection, value is the description of tools in this collection.
        '''
        return {key:TOOL_COLLECTIONS[key].description for key in TOOL_COLLECTIONS.keys()}

    @tool
    def update_accessible_tool_names(tool_collection_name:str,  runtime:ToolRuntime) -> Command:
        '''
        Given a tool collection name, update the tool names which are accessible to the agent.
        Args:
            name: Tool collection name.
        '''
        if not tool_collection_name in TOOL_COLLECTIONS:
            return Command(update={
                "messages": [
                    ToolMessage(status="error",
                                name="update_accessible_tool_names",
                                content=f"Error: {tool_collection_name} is not a name of tool collection. Availables collections: {list(TOOL_COLLECTIONS.keys())}",
                                tool_call_id = runtime.tool_call_id)
                ]
            })

        tool_names = TOOL_COLLECTIONS[tool_collection_name].tool_names
        accessible_names = runtime.state["accessible_tool_names"]
        update_tool_names = {name for name in tool_names if not name in accessible_names}
        
        return Command(update={
            "accessible_tool_names": update_tool_names,
            "messages": [
                ToolMessage(
                    content=f"Loaded tools: {update_tool_names}",
                    tool_call_id = runtime.tool_call_id,
                    name="update_accessible_tool_names"
                )
            ]
        })


    @tool 
    def pause_to_ask(runtime:ToolRuntime, 
                     message:str, 
                     type: Literal["any", "confirmation", "choice-single", "choice-multiple"] = "any",
                     meta_data:Dict[str, Any] = None ):
        '''
        Emit an interrupt to pause the agent for obtaining authorization, confirmation, or supplementary information from the user. For example, pause to ask user for a python environment path or let them manually install a package, ask the user to set key calculation parameters, such as DFT functional, MD temperature, linear solver type, etc., if they didn't provide.
        Args:
            message: The message sent to user.
            type: Type of interrupt, can be
                - "confirmation": Ask the user to confirm something.
                - "choice-single": Ask the user to choice a single value from the given options. Options should be explained in `meta_data`, keys are the options, values are the meaning.
                - "choice-multiple": Ask the user to choice multiple values from the given options.
                - "any": For complex data, use "any". For example, ask the user to set a range or a list.
            meta_data: Extra meta data. Can be the choices requiring the user to select or data fields requiring the user to fill. Key is the choice or field name, value is the meaning.
        '''
        return {} 

    return [get_tool_collection_infos, update_accessible_tool_names, pause_to_ask]



@tool 
def get_workdir(runtime:ToolRuntime):
    '''
    Return the workdir path of current step.
    '''
    return runtime.state["workdir"]

async def tool_call_wrapper(request:ToolCallRequest, handler):
    ask_arg_tool_msgs = {
        "generate_elementary_steps": "Please set the constraint condition to limit the number of reactions.",
        "delete_species": "Please input the indices of intermediate species you want to delete, separated by commas.",
        "delete_reactions": "Please input the indices of reactions you want to delete, separated by commas.",
        "merge_reactions": "Please input the index pairs of reactions you want to merge, the format is: (id1, id2), (id3, id4)."
    }
    req = request
    if req.tool is None:
        requested_name = req.tool_call["name"]
        if requested_name in TOOLS_BY_NAME:
            req = req.override(tool=TOOLS_BY_NAME[requested_name])
    tool_name = req.tool_call.get("name")
    #if not tool_name in req.state["skip_review_tool_names"]:
    if tool_name in ask_arg_tool_msgs:
        ask = AskForArgs(pending=True,
                        message=f"Tool {tool_name} requires args edit, {ask_arg_tool_msgs[tool_name]}:",
                        tool_call_id = req.tool_call["id"],
                        tool_name = tool_name,
                        current_args = req.tool_call["args"],
                        args_schema = req.tool.args_schema.model_json_schema() if req.tool.args_schema else {},
                        missing=[k for k, v in req.tool_call["args"].items() if v is None])
        
        return Command(update={"interrupts": {req.tool_call["id"]: ask.model_dump(mode='python')} })
    if tool_name == "pause_to_ask":
        obj = AgentInterrupt(pending=True,
                            message= req.tool_call["args"]["message"],
                            tool_call_id = req.tool_call["id"],
                            tool_name="pause_to_ask",
                            type = req.tool_call["args"]["type"],
                            meta_data = req.tool_call["args"]["meta_data"])
        
        return Command(update={ "interrupts": {req.tool_call["id"]: obj.model_dump(mode="python")} }) 
    #return await handler(req)
    try:
        return await handler(req)
    except Exception as e:
        error_msg = f"Tool error: ({e})"
        return Command(
            update={
                "messages": [
                    ToolMessage(status="error", 
                                name=req.tool_call.get("name"),
                                content=error_msg, 
                                tool_call_id=req.tool_call["id"])
                ],
                "tool_errors": [error_msg]
            }
        )     




def get_tool_node(model:BaseChatModel,
                  job_manager_url:str,
                  http_client:httpx.AsyncClient):
    
        
    static_tools = init_common_tools(model)
    static_tools.append(get_workdir)
    static_tools.append(execute_shell)
    static_tools.append(edit_file) 
    static_tools.append(read_file)
    static_tools.append(get_skill_summaries)
    static_tools.append(get_skill_content)
    static_tools.extend(init_job_manager_tools(job_manager_url, http_client))
    return (static_tools, ToolNode(static_tools, awrap_tool_call=tool_call_wrapper))