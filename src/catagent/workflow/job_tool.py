from langchain.tools import tool 
import httpx

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
    
    return get_providers_info

    