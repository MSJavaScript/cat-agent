---
name: stepnode-tools
description: Usage of tools in StepNode tool collections. Including structure-building and reaction network generation.
metadata:
  tags:
    - structure-building
    - neb
    - transition state search 
    - reaction network
---

# StepNode tools

This skill explains the usage of tools in StepNode subagent. These tools are organized in different collections. You can call `get_tool_collection_infos` to get all the available tool collections and call `load_tool_collection` to load a specific collection.

## "structure_building_tools" collection.
This collection contains tools which can build IS and FS, adsorption structure, and initial NEB path. 

- `sampling_ini_fin_states_on_surface`: Given IS and FS molecules, automatically match their atomic indices, then attach them on a surface, all surface sites are considered.
- `build_initial_nebpath_on_surface`: Given IS and FS, build neb path. This tool will handle some special cases, for example, it will adjust the structure if molecule dissociates in vacuum. 
- `sampling_single_ads_configs_on_surface`: Attach an adsorbate on the surface, all surface sites are considered.


## "reaction_system_tools" collection 

This collection have tools which can generate, edit and visualize the reaction network. For a total reaction, such as CH4 + CO2 <-> 2* CO + 2* H2, there are infinite elementary reactions in theory, because we can generate infinite intermediates by combing atoms and atomic groups. Considering the computation cost, we need to set constraints when generate a reaction network. There are also multiple isomers for a chemical formula, for example, CHOH have the same formula as CH2 O, you cannot distinguish reactions "HCO + H <-> HCOH" and "HCO + H <-> H2CO" from the chemical formulas. To distinguish them, we provide visualization tool to render the reaction into `.png`, let the user to select the reactions and intermediates they need. There are some tools need Human In The Loop (HITL) to provide args, when the tool is called, it automatically pause the agent to ask the user for args.

The tools below marked by HITL will automatically pause the agent to ask the user to modify the tool args. The details of network generation, e.g, constraints, will be asked by StepNode, Orchestrator should leave these details to StepNode.

- `create_reaction_system`: Call this tool firstly, you should create a reaction system before generating and editing reaction network.
- `generate_elementary_steps`(HITL): Generate reactions by consider all kinds of combinations, constraints should be set to limit the number of reactions. **No intermediate fragments are needed, they will be generated automatically**.
- `render`: Render the reactions and intermediates into png pictures. The user should inspect these pictures to determine which reactions and intermediates will be deleted.
- `delete_species`(HITL): Delete intermediates from the system.
- `delete_reactions`(HITL): Delete reactions from the system.
- `merge_reactions`(HITL): Merge some reactions. `generate_elementary_steps` only consider single-bond forming or breaking, reactions such as "NHOH + H <-> NH + H2O" will be split into two or three reactions, for example, "NHOH <-> NH + OH", "OH+H<->H2O". We need call `merge_reactions` to let the user determine which reactions should be merged.
- `dump`: Dump the reaction system object to a pkl file, so that it can be used in another run.
- `load_from_file`: Load a reaction system object from pkl file.
- `has_reaction_system`: Check if a reaction sytem exists.
- `delete_reaction_system`: To save memory, we should delete the object when edit task is done.
- `bind_to_surface`: Given a surface structure, bind all the initial and final states to the surface. For complex surface, such as Fe(110), there will be many configurations, user should select which will be kept to limit the computation cost.

#### General workflow to generate a reaction network
Take reaction CH4 + CO2 <-> 2* CO + 2* H2 as an example.
1. Generate all the reactants and products structures, using `*.xyz` format.
    ```bash
    python << 'EOF'
    from ase.io import write
    from ase.build import molecule
    for mol in ["CH4", "CO2", "H2", "CO"]:
      write(f"{mol}.xyz", molecule(mol))
    EOF
    ```
    Also generate the surface we want to study. You can call `build_slab.py` in `structure-building` skill to do this.

2. Call `create_reaction_system` to generate a reaction system. 
3. Call `generate_elementary_steps` to generate elementary steps. This tool generate a `json` file and an `extxyz` file. `json` file contains the indices of reactants and products of elementary reactions, for example
    ```json
    {
      "id": 1,
      "reac_str": "CO2 <-> CO + O",
      "reactant_indices": [
        1
      ],
      "product_indices": [
        2,
        6
      ]
    },
    ```
    `extxyz` file contains the structures of all the intermediates. You can call `pause_to_ask` to pause the agent and let user inspect these files by themself to determine which reactions and intermediates they want to delete. You can also call `render` tool to render the intermediates and reactions into `png`.
4. Call `delete_species`, `merge_reactions` and  `delete_reactions` one by one to let the user edit reaction network.  

    NOTE: Both step 3 and 4 can be called multiple times. You can call `pause_to_ask` to ask the user if a HITL tool should be run again, for example:
    ```text
    # after generate_elementary_steps, call `pause_to_ask`
    Reaction network has been generated, please inspect the *extxyz file and the pictures xxx. If you want to regenerate the network, type "again", if not, type "next" or "proceed", we will start to edit the reaction network.
    ```
    ```text
    # after delete_species
    Do you want to delete more intermediate species? Type "yes" to do it again, type "next", "no" or "proceed" to do the next step (select the reactions you want to merge).
    ```
    ```text
    # after delete_reactions
    Do you want to delete more reactions ? There are xx reactions left. Type "end", we will end the edit phase and build structures.
    ```
5. When reaction network edit complete, you can call `dump` to dump the reaction system object into a pkl file. Then call `bind_to_surface` to attach IS and FS of each elementary step to surface. At last, you should call `delete_reaction_system` to delete the object from memory.

