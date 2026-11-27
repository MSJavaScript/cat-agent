import dotenv, os, json, sys
sys.path.insert(0, "/mnt/t/programming/myagents/cat-agent-dev/src")
from catagent.workflow.models import SkillSummaryOutput, SkillMetadata, SkillArtifact
from langchain.messages import HumanMessage
from langchain_deepseek import ChatDeepSeek
from langchain.chat_models import BaseChatModel

def load_skill_from_filesystem(path:str, model:BaseChatModel):   
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
        print(error_msg)
        return None 
    

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

    response = model.with_structured_output(SkillSummaryOutput).invoke([message ])

    
    
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
    return skill_content



dotenv.load_dotenv("/home/luandong/bin/.env")
model = ChatDeepSeek(model="DeepSeek-V4.1-Flash-1017",
                     api_key=os.getenv("API_KEY"),
                     base_url=os.getenv("BASE_URL"))

result = []
path = "/mnt/t/programming/myagents/cat-agent-dev/src/catagent/skills"
for d in os.listdir(path):
    skill_path = os.path.join(path, d, "skill.md")
    if os.path.isfile(skill_path):
        skill_content = load_skill_from_filesystem(skill_path, model)
        result.append(skill_content)
        print("==============")
        print(json.dumps(skill_content, indent=4))

        
with open("skill_index.json", mode="w") as f:
    json.dump(result, f, indent=4)
    
