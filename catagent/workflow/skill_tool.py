from langchain.chat_models import BaseChatModel
from langchain.tools import tool, ToolRuntime
from langgraph.types import Command
from langchain.messages import ToolMessage, HumanMessage
from typing import Dict
import os
from .models import SkillSummaryOutput, SkillArtifact, SkillMetadata

@tool 
def get_skill_summaries(runtime:ToolRuntime):
    '''
    Return a dict containing the all the summaries of available skills.
    Returns:
        A dict. Key is the skill name, value is the summary of skill.
    ''' 
    loaded_skills:Dict[str, SkillMetadata] = runtime.state["loaded_skills"]

    result = dict()
    for key in loaded_skills.keys():
        script_content = ""
        for script in loaded_skills[key]["scripts"]:
            script_content += f"|{script['path']}|{script['purpose']}|\n"
        if len(script_content) > 0:
            script_content = 'Available scripts:\n|Name|Purpose|\n|---|---|\n'+script_content

        result[key] = f"Summary:{loaded_skills[key]["summary"]}\n\n{script_content}"

    return result


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

# Skill管理的相关工具，加载skill，读取文件系统的某个skill
def init_load_skill_from_fs(model:BaseChatModel):
    @tool 
    def load_skill_from_filesystem(path:str, runtime:ToolRuntime) -> Command:   
        '''
        Load a skill from the file system.
        Args:
            path: Absolute path to skill.
        '''
        error_msg = ""
        if os.path.isdir(path):
            dir_path = path 
            for fname in os.listdir(path):
                if fname.lower() == "skill.md":
                    skill_path = os.path.join(dir_path, fname)
                    if os.path.isfile(skill_path):
                        break 
            else:
                error_msg = f"Dir {path} does not have a file named by skill.md or SKILL.md."
        elif os.path.isfile(path):
            basename = os.path.basename(path)
            if basename.lower() == "skill.md":
                dir_path = os.path.dirname(path)
                skill_path = path 
            else:
                error_msg = f"The filename of skill doc must be skill.md or SKILL.md, got {basename}."
        else:
            error_msg = f"{path} is not a file or directory."

        if len(error_msg) > 0:
            return Command(update={
                "messages": [ ToolMessage(status="error",
                                          content=f"Error:{error_msg}", 
                                          tool_call_id=runtime.tool_call_id,
                                          name="load_skill_from_filesystem")]
            })
        
    
        with open(skill_path, mode='r') as f:
            lines = f.readlines()
            content = ''.join(lines)

        ref_dir = os.path.join(dir_path, 'references')
        references = []
        references_basename = []
        if os.path.isdir(ref_dir):
            for f in os.listdir(ref_dir):
                abs_path = os.path.join(ref_dir, f)
                if os.path.isfile(abs_path):
                    references.append(abs_path)
                    references_basename.append(f)
        
        script_dir = os.path.join(dir_path, 'scripts')
        scripts = []
        scripts_basename = []
        if os.path.isdir(script_dir):
            for f in os.listdir(script_dir):
                abs_path = os.path.join(script_dir, f)
                if os.path.isfile(abs_path):
                    scripts.append(abs_path)
                    scripts_basename.append(f)


        message = HumanMessage(content=(
            "Summary the skill document and the purpose of related scripts and reference files\n\n"
            f"Available scripts: {scripts_basename if len(scripts_basename) > 0 else None}\n"
            f"Available reference files: {references_basename if len(references_basename) > 0 else None}\n"
            f"Skill full text:\n\n {content}"
        ))

        response:SkillSummaryOutput = model.with_structured_output(SkillSummaryOutput).invoke(input={"messages": [message]})
        
        references_artifacts = []
        scripts_artifacts = []
        for basename, fullname in zip(references_basename, references):
            references_artifacts.append(SkillArtifact(path = fullname, purpose=response.reference_purposes[basename]))
        for basename, fullname in zip(scripts_basename, scripts):
            scripts_artifacts.append(SkillArtifact(path=fullname, purpose=response.script_purposes[basename]))

        skill_content = SkillMetadata(name=response.name,
                                     path=path,
                                     summary=response.summary,
                                     references=references_artifacts,
                                     scripts=scripts_artifacts )
        return Command(update={
            "loaded_skills": {response.name: skill_content},
            "messages": [
                ToolMessage(
                    content=f"Skill {path} is loaded into agent state, with key {response.name}.",
                    tool_call_id=runtime.tool_call_id,
                    name="load_skill_from_filesystem"
                )
            ]
        })

    return load_skill_from_filesystem




