---
name: vasp
description: General knowledge of vasp usage.
metadata:
  tags:
    - ase
    - vasp
---

# VASP skill

This document explains some basic knowledge of vasp usage.

## Input files
Basic input files of VASP are `INCAR`, `POSCAR`, `KPOINTS` and `POTCAR`. Specific calculations, e.g. dimer and constrained molecular dynamics, need extra inputs. See the corresponding skill docs for more details.

### INCAR
Agent tool `generate_incar_draft` can produce a draft INCAR from the user's prompt. For example, pass the following prompt to the tool:
```
User: "I want to optimize the CO2 adsorption configuration on Ni(111) surface. With energy cutoff 420, RPBE functional."
```
`generate_incar_draft` will return a draft containing tags which **maybe** applied to this calculation. Be cautious that, this draft is only a reference. We should look at the skill doc corresponding the calculation and refine it. Also, it never set actual values of `MAGMOM`, `LDAUU`, `LDAUJ`, the tags which depend on the POSCAR.

NOTE: For tags that significantly affect the calculation results, such as `LDAUU`, request the user to provide their values if they haven't provide them. 

### POSCAR
See the "structure building" skill to see how to build a structure.

### KPOINTS
The number of k-points in a direction is inversely proportional to the lattice length. For a slab, the number of kpoints is usually 1 in the direction of vacuum region. For the calculation of molecule, we usually build a large cell to avoid periodic interaction, the kpoints are usually (1,1,1) (gamma only). 

Computational cost is proportional to the number of KPOINTS. For surface catalysis, the kpoints for a 4*4 surface are usually (3,3,1) or (4,4,1) (suppose the vacuum region is in z direction). The kpoints for surfaces with other sizes can be calculated by the inverse proportion relationship.

`KPOINTS` can be omitted if `KSPACING` tag is set in INCAR. KSPACING is the smallest allowed spacing between k points in units of $Å^{-1}$. The number of k points increases when the spacing is decreased. This tag is usually used for bulk systems, especially when we are preparing the training dataset for MLFF training or finetuing, and maintain consistent computational accuracy across unit cells of different sizes. We usually set `KSPACING=0.2` or `0.25`.

`scripts/generate_kpoints.py` can generate `KPOINTS` for normal calculation and band structure calculation. Use `-h` option to see the help messages.


### POTCAR
Given elements, `scripts/recommend_potcar.py` print the recommended POTCAR for the elements. For example:
```bash
python3 recommend_potcar.py -e "Na, K, Rh, Cd, Ga, Fe"
# print out "Na:Na_pv  K:K_sv  Rh:Rh_pv  Cd:Cd  Ga:Ga_d  Fe:Fe"
```
We can copy the corresponding POTCARs to local dir and use `cat` to generate a `POTCAR` for `POSCAR`.
```bash
potcar_lib_path= # path to the POTCARs
for m in `python3 recommend_potcar.py -e "Na, K, Rh, Cd, Ga, Fe"`
do
  cp ${potcar_lib_path}/${m#*:}/POTCAR POTCAR_${m%:*}
done

# Line 6 is the elements lists of POSCAR
eles=`sed -n '6p' POSCAR`
for e in ${eles}
do
  cat POTCAR_${e} >> POTCAR
done

# You can use grep to check the order of the elements in POTCAR
# make sure they are the same as in POSCAR
grep TITEL POTCAR | awk '{print $4}'
```

## Notes and practical tips
- The more tags in INCAR, the more easily the calculation encounters problems. Do not add a tag unless you know its exact function. 
- When input files are ready, pause. Let the user check the inputs. Especially INCAR.
- Keep `LWAVE = .FALSE.` and `LCHARG = .FALSE.` to reduce the size of output, unless you need `WAVECAR` or `CHGCAR`.
- In the same study, the tags should be consistent. For example, you should use the same DFT functional and `ENCUT`, kpoints should be adjusted for cells with different sizes.
- If you encounter convergence problem, give VASP an initial guess to help the calculation converge. This is usually done by running a low precision calculation and using the resultant `WAVECAR` as initial guess. For example, original job is `ENCUT = 520` and `ISPIN = 2`, you can use `ENCUT=350` and `ISPIN=1` firstly and produce a `WAVECAR`, then let VASP read it in subsequent calculation. Similar practices also include, firstly normal GGA then GGA+U or hybrid functional, firstly coarser kpoints then refine the KPOINTS, etc.
- Same technique can also be used in structure relaxation. You can firstly use coarser kpoints and low `ENCUT` to relax the structure. Then use the `CONTCAR` as starting point and restore INCAR tags to the values that the user requires.


## Scripts
Below are the related scripts in `scripts` dir. Run these scripts without parameters or with `-h` to get the helps.

| Name | Purpose|
| ---- | ---- |
|generate_kpoints.py| Generate KPOINTS |
|recommend_potcar.py|Print recommended POTCAR names|

## References
| Name | Purpose|
| ---- | ---- |
|job_submission.md| File template for job submission, only for StepNode |