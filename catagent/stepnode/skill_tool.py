
from langchain.tools import tool, ToolRuntime
from typing import Dict
from .models import SkillMetadata

@tool 
def get_skill_summaries(runtime:ToolRuntime):
    '''
    Return a dict containing the all the summaries of available skills.
    Returns:
        A dict. Key is the skill name, value is the summary of skill.
    ''' 
    loaded_skills:Dict[str, SkillMetadata] = runtime.state["loaded_skills"]
    return {key:loaded_skills[key]["summary"] for key in loaded_skills.keys()}


# 在reference这里跟workflow那里有区别。对StepNode开放所有的references访问。
@tool 
def get_skill_content(skill_name:str, runtime:ToolRuntime) -> dict:
    """
    Get the detailed content of skill by `skill_name`.
    Args:
        skill_name(str): Name of skill.
    Returns:
        A dict containing the following keys:
        - content: Full text content of the skill.
        - references: Related reference files of the skill.
        - scripts: Related scripts of the skill.
    """
    loaded_skills:Dict[str, SkillMetadata] = runtime.state["loaded_skills"]
    if not skill_name in loaded_skills:
        raise KeyError(f"Skill \"{skill_name}\" dose not exist.")

    content = ""
    with open(loaded_skills[skill_name]["path"], mode="r") as f:
        lines = f.readlines()
        content = "".join(lines)

    if len(content) == 0:
        raise RuntimeError("The content of skill.md is empty.")

    if loaded_skills[skill_name]["references"]:
        references = loaded_skills[skill_name]["references"]
    else:
        references = []


    if loaded_skills[skill_name]["scripts"]:
        scripts = loaded_skills[skill_name]["scripts"]
    else:
        scripts = []

    return { "content": content, "references": references, "scripts": scripts }







