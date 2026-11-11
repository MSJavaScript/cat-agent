from langchain.chat_models import BaseChatModel
from langchain.agents import create_agent
from langgraph.types import Checkpointer
from langchain.tools import tool, ToolRuntime
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.graph.state import CompiledStateGraph
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from langchain.agents.structured_output import ToolStrategy
from typing import Callable

from dataclasses import dataclass
from typing import List, Literal, Optional, Dict
from jobmanager.base import BaseProvider
from pydantic import BaseModel, Field
from .util_tools import execute_shell, edit_file, preprocessing_validation
import os, httpx, logging


logger = logging.getLogger("jobmanager")

@dataclass
class JobManagerAgentContext:
    provider_id: str = None 
    job_env_name: str = None 
    phase: str = None 


class PreprocessingOutput(BaseModel):
    '''
    Structured output for preprocessing phase.
    '''
    batch_job: bool    = Field(description="If this is a batch job.", default=False)
    n_jobs: int        = Field(description="The total number of jobs.", default=1)
    n_groups: Optional[int] = Field(description="The number of groups, only valid for batch job.", default=None)
    group_dirs: Optional[List[str]] = Field(description="The dir names or dir wildcards for each job group. A group has ONLY one `run.sh` and `job_description.json` file in the group dir.", default=None)

class CheckingOutput(BaseModel):
    '''
    Structured output for checking phase.
    '''
    job_result_status: Literal["Success", "InputError", "EnvironmentError"] = Field(description="The status of the downloaded job results.", default=None)
    message:str = Field(description="The brief summary or error messages of job results, within 3 to 5 sentences.", default=None)

class JobManagerAgentOutput(BaseModel):
    '''
    Final structured output.
    '''
    phase: Literal["preprocessing", "checking"] = Field(description="Phase of the current task. Preprocessing of remote job submissions or checking the results of downloaded jobs.")
    preprocessing_output: Optional[PreprocessingOutput] = Field(description="Output for preprocessing phase.", default=None)
    checking_output: Optional[CheckingOutput] = Field(description="Output for checking phase.", default=None)
    

@wrap_model_call
async def phase_based_tools(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """Filter tools based on model context."""
    context = request.runtime.context 
    checking_phase_tools_name = { "execute_shell",  "edit_file" }
    if getattr(context, "phase", None) == "checking":
        tools = [t for t in request.tools if t.name in checking_phase_tools_name]
        request = request.override(tools=tools)
    return await handler(request)


class JobManagerAgent:
    def __init__(self, providers:Dict[str, BaseProvider]):
        self.providers = providers
        with open(os.path.join(os.path.dirname(__file__), "skill.md")) as f:
            lines = f.readlines()
            self.system_prompt = ''.join(lines)
        
    def _get_resources(self, provider_id:str):
        return self.providers[provider_id].get_resources()

    def _get_job_env(self, provider_id:str, job_env_name:str):
        return self.providers[provider_id].get_job_env(job_env_name)

    def _get_remote_files_list(self, provider_id:str):
        return self.providers[provider_id].get_remote_files_list()
     
    def create(self, model:BaseChatModel, checkpointer:Checkpointer):
        this = self 

        @tool 
        def get_remote_files_list(runtime: ToolRuntime[JobManagerAgentContext]) -> List[dict]:
            '''
            Return the list of remote files in a provider. NOTE: `provider_id` (name of the provider) is injected from the agent context.
            Returns:
                A list of dict containing the informations of remote files. Each dict contains the following keys:
                - description: The purpose of these files.
                - path: File path without file name. 
                - files: Name of files that can be accessed.
            '''
            ctx = runtime.context
            return this._get_remote_files_list(ctx.provider_id)

        @tool 
        def get_resources(runtime: ToolRuntime[JobManagerAgentContext]) -> List[dict]:
            '''
            Return the information of computation resources of a provider. NOTE: `provider_id` (name of the provider) is injected from the agent context.
            Returns:
                A list of dict containing the informations of computational resources. Each dict contains the following keys:
                - machine_name: Name of machine, valid for bohr, optional.
                - machine_type: "cpu" for "gpu", valid for bohr, optional.
                - partition_name: Name of queue partition, valid for slurm, optional.
                - partition_type: Type of queue partition, "cpu" for "gpu", valid for slurm, optional.
                - n_nodes: Number of nodes in the partition, valid for slurm, optional.
                - n_cpu_core: Number of CPU core for a machine or node, optional.
                - n_gpu_core: Number of GPU core for a machine or node, optional.
                - memeory_size: Size of memory in GB, optional.
                - video_memeory_size: Size of video memory in GB, optional.
                - price: Price per hour, optional.
            '''
            ctx = runtime.context
            return this._get_resources(ctx.provider_id)

        @tool 
        def get_job_env(runtime: ToolRuntime[JobManagerAgentContext])->dict:
            '''
            Return the information of a job environment of a provider. NOTE: `provider_id` (name of the provider) and `job_env_name` (Name of the job environment) are injected from the agent context.
            Returns:
                A dict containing the following keys:
                - name: Name of job environment. 
                - image_path: Docker image path, valid for bohr provider, optional.
                - softwares: Available softwares.
                - initialize_command: Commands for initialization, optional.
                - dataset_path: Path to the dataset, optional.
                - datasets: Available dataset names, optional.
            '''
            ctx = runtime.context
            return self._get_job_env(ctx.provider_id, ctx.job_env_name)
        
        agent = create_agent(model=model,
                             system_prompt=this.system_prompt,
                             checkpointer=checkpointer, 
                             tools=[execute_shell, edit_file, get_resources,
                                     get_job_env, get_remote_files_list, preprocessing_validation],
                             context_schema=JobManagerAgentContext,
                             response_format=ToolStrategy(JobManagerAgentOutput, handle_errors=True),
                             middleware=[phase_based_tools])
        return agent 


async def run_job_agent(
        agent:CompiledStateGraph,
        phase:Literal["preprocessing", "checking"],
        thread_id:str,
        local_dir:str,
        log_msg:bool = False,
        provider_id:str = None,
        job_env_name:str = None,
        http_client: httpx.AsyncClient = None,
        url:str = None) -> dict:
    '''
    向logger和一个url输出AIMessage、ToolCall和ToolMessage
    最后返回一个结果
    '''
    if phase == "checking":
        message = HumanMessage(f"Check the job results download in local dir {local_dir}. For the final output, make sure to generate a structured `CheckingOutput` object for the `checking_output` field.")
    else:
        message = HumanMessage(f"Preprocess the job inputs in local dir {local_dir}. For the final output, make sure to generate a structured `PreprocessingOutput` object for the `preprocessing_output` field.")

    context = JobManagerAgentContext(provider_id=provider_id, 
                                     job_env_name=job_env_name,
                                     phase=phase)
    
    thread_config = {"configurable": {"thread_id": thread_id}}
    structured_response = None 
    async for chunk in agent.astream(input={"messages": [message]},
                                     config=thread_config,
                                     context=context,
                                     stream_mode="updates",
                                     version="v2"):
        
        if chunk["type"] != "updates":
            continue 
        for _, update in chunk["data"].items():
            messages = update.get("messages", [])
            if not messages:
                continue

            msg = messages[-1]
            if isinstance(msg, AIMessage):
                payload = {"role": "agent",
                           "message": msg.content,
                           "tool_calls": f"{msg.tool_calls}" if msg.tool_calls else "",
                           "phase": phase}
                
                log_msg = f"[Agent]: {msg.content}" + f"\n[Tool calls]: {msg.tool_calls}" if msg.tool_calls else ""
            elif isinstance(msg, ToolMessage):
                payload = {"role":"tool",
                           "message": msg.content,
                           "phase": phase}
                log_msg = f"[Tool]: {msg.content}"
            else:
                payload = {"role":None, "message":msg, "phase": phase}
                log_msg = f"[Message]: {msg}"

            if log_msg:
                logger.info(log_msg)

            if update.get("structured_response") is not None:
                structured_response = update["structured_response"]

            if url and http_client:
                try:
                    response = await http_client.post(url=url, 
                                                  json=payload)
                    if not 200 <= response.status_code < 300:
                        logger.warning(f"Http error, return code: {response.status_code}")
                except Exception as e:
                    logger.warning(f"Http error: {e}")

    return structured_response.model_dump(mode="python") if structured_response else dict()
                



