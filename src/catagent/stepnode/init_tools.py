from langchain.chat_models import BaseChatModel
from langgraph.types import Command
from langchain.tools import tool, ToolRuntime
from langchain.messages import ToolMessage
from langgraph.prebuilt import ToolNode
from langchain_core.tools.base import BaseTool, InjectedToolArg
from langchain.agents.middleware import ToolCallRequest
from langgraph.errors import GraphBubbleUp

from catagent.tools.neb.neb_tool import (build_ini_fin_structures_on_surface,
                                        build_nebpath_on_surface,
                                        build_adsorption_configs_on_surface)
from catagent.workflow.skill_tool import get_skill_content, get_skill_summaries

from catagent.tools.neb.reaction_system import ReactionSystemToolSets
from catagent.tools.other.util_tools import execute_shell, edit_file, read_file
from .models import ToolNamespace, AgentInterrupt, JobSubmission, AskForArgs, StepNodeContext
from typing import List,  Dict,  Any, Literal, Optional, Tuple, Annotated
import httpx, logging, uuid, os
from pathlib import Path

logger = logging.getLogger(__name__)

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
    async def submit_job(job_description: str,
                         local_dir: str, 
                         provider_name: str,
                         job_env_name: str,
                         result_patterns: Optional[List[str]] = None,
                         job_uuid: Annotated[Optional[str], InjectedToolArg] = None) -> Tuple[bool, Any]:
        '''
        Submit a job to job manager.
        Args:
            job_description: Description of the job.
            local_dir: The local workdir where the files to be submitted exist. One job_description.json and one run.sh file should exist in local_dir.
            provider_name: The name of selected provider.
            job_env_name: The name of selected job environment.
            result_patterns: The file name or wildcard of the result to be recycled. Download all the calculation results by default.
        '''
        if not os.path.isfile(os.path.join(local_dir, "run.sh")):
            return (False, f"Error: run.sh does not exist in {local_dir}.")
        if not os.path.isfile(os.path.join(local_dir, "job_description.json")):
            return (False, f"Error: job_description.json does not exist in {local_dir}.")
        base = Path(local_dir)
        for a in ["run.sh", "job_description.json"]:
            files = [str(fpath) for fpath in base.rglob(a) if fpath.is_file()]
            nfiles = len(files)
            if nfiles != 1:
                return (False, f"Error: The should be one and only one {a} file in {local_dir}, but find {nfiles}.")
                    
        # `job_uuid` is injected by `tool_call_wrapper`, not by the model.  It is
        # derived from (inner thread id, tool_call_id), which are both stable
        # when LangGraph replays a checkpoint.  A replay therefore addresses the
        # *same* jobmanager row instead of creating a second remote job.
        # The uuid4 fallback only applies to direct callers (tests / manual use).
        job_uuid = job_uuid or str(uuid.uuid4())
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

        def submission_failed(message: str):
            return (False, f"Job submission failed. \nError message: {message}. \nMetadata: {metadata}")

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

        return (True, payload)
    

    return [submit_job, get_providers_info]


def init_common_tools(model:BaseChatModel):
    collection = ReactionSystemToolSets().create()
    name_list = []
    for item in collection:
        if item.name in TOOLS_BY_NAME:
            raise RuntimeError(f"Tool {item.name} is already exist.")
        TOOLS_BY_NAME[item.name] = item 
        name_list.append(item.name)

    details = '''Tools:
- `create_reaction_system`: Call this tool firstly, you should create a reaction system before generating and editing reaction network.
- `generate_elementary_steps`(HITL): Generate reactions by consider all kinds of combinations, constraints should be set to limit the number of reactions.
- `render`: Render the reactions and intermediates into png pictures. The user should inspect these pictures to determine which reactions and intermediates will be deleted.
- `delete_species`(HITL): Delete intermediates from the system.
- `delete_reactions`(HITL): Delete reactions from the system.
- `merge_reactions`(HITL): Merge some reactions. 
- `dump`: Dump the reaction system object to a pkl file.
- `load_from_file`: Load a reaction system object from pkl file.
- `has_reaction_system`: Check if a reaction sytem exists.
- `delete_reaction_system`: To save memory, we should delete the object when edit task is done.
- `bind_to_surface`: Given a surface structure, bind all the initial and final states to the surface.
'''

    TOOL_COLLECTIONS["reaction_system_tools"] = ToolNamespace(
        description=f"Generate elementary reactions, edit reaction network. Given the surface structures, create initial and final states of all the generated reactions in batch.\n{details}",
        tool_names=name_list
    )


    collection = [build_ini_fin_structures_on_surface,
                  build_nebpath_on_surface,
                  build_adsorption_configs_on_surface]
    name_list = []
    for item in collection:
        if item.name in TOOLS_BY_NAME:
            raise RuntimeError(f"Tool {item.name} is already exist.")
        TOOLS_BY_NAME[item.name] = item 
        name_list.append(item.name)

    details = '''Tools:
- `build_ini_fin_structures_on_surface`: Given IS and FS molecules, attach them on a surface, automatically match the atomic indices, all surface sites are considered.
- `build_nebpath_on_surface`: Given IS and FS, build neb path. This tool will handle some special cases, for example, it will adjust the structure if molecule dissociates in vacuum. 
- `build_adsorption_configs_on_surface`: Attach an adsorbate on the surface, all surface sites are considered.
'''
    TOOL_COLLECTIONS["structure_building_tools"] = ToolNamespace(
        description=f"Build initial and final states on a surface. Bind an adsorbate to a surface. Build the initial NEB paths. Use these tools for small-scale structure building tasks.\n{details}",
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
    def load_tool_collection(tool_collection_name:str,  runtime:ToolRuntime[Any]) -> Command:
        '''
        Given a tool collection name, update the tool names which are accessible to the agent, so that they can be used by agent.
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
    def pause_to_ask(runtime:ToolRuntime[Any], 
                     message:str, 
                     type: Literal["any", "confirmation", "choice-single", "choice-multiple"] = "any",
                     meta_data:Dict[str, Any] = dict() ):
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

    return [get_tool_collection_infos, load_tool_collection, pause_to_ask]



@tool 
def get_workdir(runtime:ToolRuntime[StepNodeContext]):
    '''
    Return the workdir path of current step.
    '''
    return runtime.context["step_workdir"]

async def tool_call_wrapper(request:ToolCallRequest, handler):
    # ask_arg_tool_msgs = {
    #     "generate_elementary_steps": "Please set the constraint condition to limit the number of reactions.",
    #     "delete_species": "Please input the indices of intermediate species you want to delete, separated by commas.",
    #     "delete_reactions": "Please input the indices of reactions you want to delete, separated by commas.",
    #     "merge_reactions": "Please input the index pairs of reactions you want to merge, the format is: (id1, id2), (id3, id4)."
    # }
    ask_arg_tool_msgs = dict() # 不再拦截这些函数调用，让stepnode自己调用pause_to_ask暂停
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
                        # `args_schema` includes injected args such as `runtime: ToolRuntime[Any]`
                        # and cannot be converted to JSON schema. `tool_call_schema`
                        # is the model-facing schema without those injected args.
                        args_schema = req.tool.tool_call_schema.model_json_schema() if req.tool and req.tool.tool_call_schema else {},
                        missing=[k for k, v in req.tool_call["args"].items() if v is None])
        
        return Command(update={"interrupts": {req.tool_call["id"]: ask.model_dump(mode='python')} })
    if tool_name == "pause_to_ask":
        # The model may omit optional args (`type`, `meta_data`), so do not
        # index them directly: use the same defaults as the tool signature.
        ask_args = req.tool_call.get("args") or {}
        obj = AgentInterrupt(pending=True,
                            message= ask_args.get("message") or "",
                            tool_call_id = req.tool_call["id"],
                            tool_name="pause_to_ask",
                            type = ask_args.get("type") or "any",
                            meta_data = ask_args.get("meta_data") or dict())
        
        return Command(update={ "interrupts": {req.tool_call["id"]: obj.model_dump(mode="python")} }) 

    if tool_name == "submit_job":
        submit_tool = req.tool
        if submit_tool is None:
            return Command(update={
                "messages": [
                    ToolMessage(status="error",
                                content="Error: the submit_job tool is not available.",
                                tool_call_id=req.tool_call["id"],
                                name=tool_name)]
            })
        # Idempotent job id.
        #
        # A resumed/replayed StepNode re-executes the *same* tool call: the
        # checkpoint still contains the AIMessage that produced this
        # `tool_call_id`.  Including the inner thread id + tool_call_id in the
        # uuid makes every replay compute the same job_uuid, so the jobmanager's
        # existing "job_uuid already exists" check collapses the duplicate.  A
        # genuine retry is a new model turn and therefore gets a new
        # `tool_call_id` (and a new job).
        runtime = getattr(req, "runtime", None)
        runtime_config = getattr(runtime, "config", None) or {}
        inner_thread_id = (runtime_config.get("configurable") or {}).get("thread_id", "")
        tool_call_id = req.tool_call.get("id") or ""
        job_uuid = str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"catagent:{inner_thread_id}:{tool_call_id}",
            )
        )
        submit_args = dict(req.tool_call.get("args") or {})
        submit_args["job_uuid"] = job_uuid
        flag, out = await submit_tool.ainvoke(input=submit_args)
        if flag: # out is a dict, payload
            obj = JobSubmission(pending=True,
                            message=f"Job {out['job_uuid']} is submitted, metadata:\n{out}",
                            tool_call_id=req.tool_call["id"],
                            tool_name=tool_name,
                            job_uuid=out["job_uuid"])
            
            return Command( update={ "interrupts": {req.tool_call["id"]: obj.model_dump(mode='python')} } ) 
        else:
            return Command(update={
                "messages": [
                    ToolMessage(status="error",
                                content=out,
                                tool_call_id=req.tool_call["id"],
                                name=tool_name)]
            })

    #return await handler(req)
    try:
        return await handler(req)
    except GraphBubbleUp:
        raise
    except Exception as e:
        # Keep the model-facing message concise, but log the full traceback so
        # the line number is available in the server log.
        logger.exception("Tool %s failed (tool_call_id=%s)",
                         tool_name, req.tool_call.get("id"))
        error_msg = f"Tool error ({type(e).__name__}): {e}"
        return Command(
            update={
                "messages": [
                    ToolMessage(status="error", 
                                name=req.tool_call.get("name"),
                                content=error_msg, 
                                tool_call_id=req.tool_call["id"])
                ]
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