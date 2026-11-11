from langchain.chat_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from langchain.messages import HumanMessage, SystemMessage
from langchain.tools import tool
from typing import Dict, Any, Literal
from pydantic import BaseModel, Field

class UserDefinedParameters(BaseModel):
    parameters: Dict[str, Any] = Field(description="User defined INCAR parameters.", default_factory=dict) 

class SystemProperties(BaseModel):
    type:  Literal["Bulk", "Surface", "Isolated", "2D", "1D", "Unknown"] = Field(description='''The type of the system.
- "Bulk": Bulk system.
- "Surface": Surface or slab system.
- "Isolated": An isolated system, for example, molecule, cluster and nanoparticle.
- "2D": (quasi) 2D materials, for example, MoS2 monolayer, graphene.
- "1D": (quasi) 1D materials, for example, carbon nanotube.
- "Unknown": Cannot determine from the user's input or does not belong to the aforementioned types.
    ''', default="Unknown") 

    magnetism: Literal["Magnetic", "Nonmagnetic", "Unknown"] = Field(description='''The magnetic ordering properties of the system. 
- "Magnetic": There may be magnetic ordering in this system. For example, molecules with unpaired electrons (O2), materials containing Mn, Fe, Co and Ni.
- "Nonmagnetic": There is no magnetic ordering in this system.
- "Unknown":  Cannot determine from the user's input.
    ''', default="Unknown")

    conductivity: Literal["Metal", "Semiconductor", "Insulator", "Unknown"] = Field(description="The conductivity of the system.", default="Unknown")

    need_dft_plus_U:bool = Field(description="Does this system need DFT+U correction. If the system is strong correlated, for example, containing localized 3d, 4f or 5f electrons, then DFT+U corrections is required. Typical systems include: transition metal oxides, perovskite oxides, rare earth and actinide compounds containing f-electrons, and cathode materials for lithium-ion batteries.", default=False) 

    need_vdw:bool = Field(description="Does this system need vdW correction? The following systems generally require van der Waals (vdW) corrections: layered two-dimensional materials, molecular crystals, organic solids, adsorption and surface systems, etc.", default=False) 

    # decision_log: Dict[str, str] = Field(description="The rationale for determining the content of each field.", default_factory=dict)

class TaskType(BaseModel):
    static_calculation: bool = Field(description="Is this a static (single point) calculation. ", default=True) 

    static_analysis_type: Literal["None", "Band", "DOS", "Bader", "Potential", "Charge", "WaveFunction"] = Field(description='''The type of static calculation analysis.This should be `"None"` if the calculation is a pure static (single point) one.  Possible values are:
- "Band": Band calculation.
- "DOS": DOS or PDOS analysis.
- "Bader": Bader charge analysis.
- "Potential":   Potential analysis. Including  calculate the work function of a surface, get the averaged electrostatic potential, etc.
- "Charge": Charge analysis. Including charge difference, get linear-average or planar-averaged charge density, etc.
- "WaveFunction": Wave function analysis.
- "None": A pure static (single point) calculation, dose not need further analysis. Or the type is not included in the aforementioned categories.
''', default="None")

    ionic_dynamics_type: Literal["None", "Relaxation", "AIMD", "NEB", "Dimer", "Phonons"] = Field(description='''How does the ions change their positions.This should be `"None"` if `static_calculation` is `True`. Possible values are:
- "Relaxation":Structure optimization. IBRION=1,2 or 3
- "Phonons": Computing phonon modes or frequency analysis. IBRION=5,6,7 or 8
- "AIMD": Molecular dynamics, IBRION = 0
- "NEB": NEB calculation to get the transition state.
- "Dimer": Dimer calculation to get the transition state.
- "None": If the calculation is static or the type is not included in the aforementioned categories.
''', default="None")


#advanced_calculation: Literal["None", "MetaGGA", "SOC", "GW0", "HSE06", "BSE", "ELF", "Optical", "Dielectric"] = Field(description="The advanced calculation supported by vaspkit.", default="None")
#     4. `advanced_calculation`: If a certain advanced feature supported by vaspkit is **explicitly mentioned** in the user's prompt, set this field. Possible values are:
# - "MetaGGA": Meta-GGA calculation
# - "SOC": Spin Orbital Coupling (SOC)  calculation
# - "GW0": GW0 Calculation
# - "HSE06": HSE06 Calculation
# - "BSE": Bethe-Salpeter-equations (BSE) calculation
# - "ELF": Electron Localization Function (ELF) calculation
# - "Optical": Optical Properties
# - "Dielectric": Calculate the static dielectric matrix by density functional perturbation theory (DFTP).
# - "None": If no advanced calculation type is mentioned, or the type is not included in the aforementioned categories.

class IncarDraftState(BaseModel):
    initial_prompt: str = Field(description="The initial description of the system.")
    user_defined_parameters: Dict[str, Any] = dict()
    type:str = "Unknown"
    magnetism:str = "Unknown"
    conductivity:str = "Unknown"
    need_dft_plus_U:bool = False
    need_vdw:bool = False
    # decision_log: Dict[str, str] = dict()
    static_calculation: bool = False
    static_analysis_type:str = "None"
    ionic_dynamics_type: str = "None"
    # advanced_calculation: str = "None"

def get_UserDefinedParameters(model:BaseChatModel, user_prompt:str) -> Dict[str, Any]:
    '''
    Parse INCAR parameters from user prompts, which can be explicitly specified or described in natural language (only commonly used parameters are supported), for example, "use DFT+U correction", "ISPIN=2 and IBRION=1".

    Args:
        model: The chat model.
        user_prompt: The user prompts.
    
    Returns:
        A dict contains the parsed INCAR parameters.
    '''

    prompt = '''You are an expert VASP INCAR generator. Your task is to extract INCAR parameters from user prompt **only if they are explicitly mentioned or clearly implied** by common VASP terminology. Do NOT invent or guess parameters that are not mentioned. At last, make sure to generate a structued output `UserDefinedParameters` object, instead of plain text.

Output a dict containing the extracted parameters. Keys are standard INCAR parameter names (uppercase strings, e.g., "ISPIN", "ENCUT"). If no parameter is mentioned, output an empty dict: {}.

### Example: common phrase → INCAR mapping (use these as reference):
| Phrase / Pattern | INCAR parameter |
|----------------|----------------|
| "use spin polarization", "spin-polarized", "unrestricted magnetization" | `{"ISPIN": 2}` |
| "PBEsol functional", "use PBEsol" | `{"GGA": "PS"}` |
| "cutoff energy is X eV", "ENCUT = X", "plane-wave cutoff X" | `{"ENCUT": X}` |
| "IBRION = X", "use IBRION X" | `{"IBRION": X}` |
| "POTIM = X", "time step X" | `{"POTIM": X}` |
| "van der Waals correction", "IVDW = X", "use vdW-DF" | `{"IVDW": X}` (X is integer, e.g., 11 for DFT-D3) |
| "EDIFF = X", "SCF tolerance X eV" | `{"EDIFF": X}` |
| "NELM = X", "max SCF steps X" | `{"NELM": X}` |
| "NSW = X", "ionic steps X" | `{"NSW": X}` |

### Examples (few-shot):

**Example 1**  
User: "use spin polarization and set cutoff to 500 eV"  
Output: `{"ISPIN": 2, "ENCUT": 500}`

**Example 2**  
User: "using PBEsol functional with IBRION = 1 and POTIM = 0.1"  
Output: `{"GGA": "PS", "IBRION": 1, "POTIM": 0.1}`

**Example 3**  
User: "I want to run with ENCUT=520 and IVDW=11"  
Output: `{"ENCUT": 520, "IVDW": 11}`

**Example 4**  
User: "set the cutoff energy to 600 eV and use spin"  
Output: `{"ENCUT": 600, "ISPIN": 2}`

**Example 5**  
User: "cutoff energy is 520 eV, set LDAUU to be '0 2' "  
Output: `{"ENCUT": 520, "LDAUU":[0, 2]}`

**Example 6**  
User: "using parameters IBRION = 0, MDALGO = 2, MAGMOM = 5*1.0 2*0.0"  
Output: `{"IBRION": 0, "MDALGO": 2, "MAGMOM":["5*1.0", "2*0.0"]}`

**Example 7**  
User: "run a static calculation with default settings"  
(no explicit parameters) → Output: `{}`

### Important rules:
- For explicit assignments like "IBRION = 1" or "set ENCUT to 520", extract directly.
- For implicit mentions like "cutoff energy is 520 eV", infer the correct parameter name (`ENCUT`).
- If a value has units (eV, fs, etc.), keep only the numeric part (e.g., "520 eV" → 520).
- If the same parameter is mentioned multiple times, keep the last mentioned value.
- Only extract parameters that are **commonly used in INCAR** and directly requested.
- Do **not** extract KPOINTS or POSCAR-related settings (e.g., k-point mesh, atomic positions).
'''
    for _ in range(3):
        try:
            response = model.with_structured_output(UserDefinedParameters).invoke(
                [ SystemMessage(content=prompt), HumanMessage(content=user_prompt) ]
            )
            response = UserDefinedParameters.model_validate(response)
            break 
        except Exception as e:
            pass 
    return response.parameters

def get_SystemProperties(model:BaseChatModel, user_prompt:str) -> SystemProperties:
    prompt = "Based on the user's input, determine the properties of the computational system (type, magnetism, conductivity, whether DFT+U or vdW correction are required). At last, make sure to generate a structued output `SystemProperties` object, instead of plain text."
    for _ in range(3):
        try:
            response = model.with_structured_output(SystemProperties).invoke(
            [ SystemMessage(content=prompt), HumanMessage(content=user_prompt) ]
            )
            response = SystemProperties.model_validate(response)
            break 
        except Exception as e:
            pass 
    return response

def get_TaskType(model:BaseChatModel, user_prompt:str) -> TaskType:
    prompt = "Based on the user's input, determine the type of VASP task: whether it is a static calculation, the type of static calculation analysis, and the way of structural change (ionic dynamics) during the calculation. At last, make sure to generate a structued output `TaskType` object, instead of plain text."
    for _ in range(3):
        try:
            response = model.with_structured_output(TaskType).invoke(
                [ SystemMessage(content=prompt), HumanMessage(content=user_prompt) ])
            response = TaskType.model_validate(response)
            break 
        except Exception as e:
            pass 
    return response

STATIC_ANALYSIS_DICT = {
    "Band": {"LORBIT": 11},
    "DOS": {"LORBIT": 11,  "ISMEAR":-5},
    "Bader": {"LCHARG": True, "LAECHG": True},
    "Potential": {"LVHAR": True},
    "Charge": {"LCHARG": True},
    "WaveFunction": {"LWAVE": True}
}

DYNAMIC_CALC_TYPE = {
    "Relaxation": {"IBRION": 2, "EDIFFG": -0.05},
    "AIMD": {"IBRION": 0, "NSW": 1000, "POTIM": 1.0},
    "NEB": {"NSW": 100, "ICHAIN":0, "IMAGES":None, "SPRING": -5, "LCLIMB": True, "IOPT": 1, "IBRION": 3, "POTIM": 0},
    "Dimer": {"ICHAIN":2, "IOPT": 2, "IBRION": 3, "POTIM": 0, "EDIFF": 1.0e-7},
    "Phonons": {"IBRION":5, "NFREE":2, "POTIM": 0.015}
}

def create_extract_user_defined_parameters_tool(model:BaseChatModel):
    @tool 
    def extract_user_defined_parameters(user_prompt:str) -> Dict[str, Any]:
        '''
        Extract INCAR parameters from user's prompt. The prompt can clearly specify INCAR tags, such as "set ENCUT to 520 eV", or describe them by words, such as "use DFT+U correction" (which correspond to "LDAU = .TRUE." and related DFT+U tags).
        Args:
            user_prompt: User prompt.
        ''' 
        return get_UserDefinedParameters(model, user_prompt)
    return extract_user_defined_parameters


def create_get_incar_draft_tool(model:BaseChatModel):
    def node_1(state:IncarDraftState):
        response = get_UserDefinedParameters(model, state.initial_prompt)
        return {"user_defined_parameters":response}
    def node_2(state:IncarDraftState):
        response = get_SystemProperties(model, state.initial_prompt)
        return response.model_dump(mode="python")
    def node_3(state:IncarDraftState):
        response = get_TaskType(model, state.initial_prompt)
        return response.model_dump(mode="python")
    
    workflow = StateGraph(IncarDraftState)
    workflow.add_node(node_1)
    workflow.add_node(node_2)
    workflow.add_node(node_3)
    workflow.add_edge(START, "node_1")
    workflow.add_edge(START, "node_2")
    workflow.add_edge(START, "node_3")
    workflow.add_edge("node_1", END)
    workflow.add_edge("node_2", END)
    workflow.add_edge("node_3", END)
    graph = workflow.compile()

    @tool 
    def get_incar_draft(user_prompt:str) -> Dict[str, Any]:
        '''
        Generate a INCAR draft for VASP based on the user's prompt. 
        Args:
            user_prompt: A prompt containing the description of the task or calculation system. For example, "calculate the adsorption energy of CO2 on Ni(111) surface." or "run MD simulation on bulk SiO2, timestep is 3fs, NVT ensemble, ENCUT 400."
        Returns:
            A dict containing the INCAR parameters. **These parameters need to be refined according to specific skill docs or structures. For example, `NSW` tag should be adjusted to suitable values for a MD simulation. `MAGMOM` and `LDAUU` tags returned by this tool will be `None`, as we cannot determine them without knowing the structure we calculate.**
        '''
        
        response = graph.invoke({"initial_prompt": user_prompt})

        params = {
            "ENCUT": 400,
            "PREC":  "Normal",
            "ALGO":  "Fast",
            "GGA":   "PE",
            "EDIFF": 1.0e-5,
            "NCORE": 4,
            "NELM": 120,
            "NELMIN": 4,
            "LWAVE": False,
            "LCHARG": False
        }

        if response["need_dft_plus_U"]:
            params.update({
                "LDAU": True,
                "LDAUTYPE": 2,
                "LDAUL": None,
                "LDAUU": None,
                "LDAUJ": None,
                "LMAXMIX": None
            })
        if response["need_vdw"]:
            params.update({ "IVDW": 11 })
        if response["magnetism"] == "Magnetic":
            params.update( {"ISPIN": 2, "MAGMOM": None} )

        if response["conductivity"] == "Metal":
            params.update({"ISMEAR": 2, "SIGMA": 0.1})
        elif response["conductivity"] == "Semiconductor" or response["conductivity"] == "Insulator":
            params.update({"ISMEAR": -5})
        else:
            params.update({"ISMEAR": 0, "SIGMA": 0.05})

        if response["static_calculation"]:
            params.update({"NSW": 0, "IBRION": -1})
            if response["static_analysis_type"] in STATIC_ANALYSIS_DICT:
                params.update(STATIC_ANALYSIS_DICT[response["static_analysis_type"]])
        else:
            params.update({"NSW": 100})
            if response["type"] == "Bulk":
                params.update({"ISIF": 3})
            else:
                params.update({"ISIF": 2})
            if response["ionic_dynamics_type"] in DYNAMIC_CALC_TYPE:
                params.update(DYNAMIC_CALC_TYPE[response["ionic_dynamics_type"]])
        
        params.update(response["user_defined_parameters"])        
        return  params
    
    return get_incar_draft




