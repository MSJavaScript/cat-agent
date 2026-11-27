# Several interrupts on one graph: collect all answers, resume once

This directory answers two questions about LangGraph interrupts, and then
applies the answer to cat-agent's StepNode and orchestrator.

```bash
cd learn/multi_interrupt_demo
python demo.py        # narrative output
python test_demo.py   # assertions only
```

No API key, network, or jobmanager is needed.

## Short answers

1. **One `ainvoke()` surfaces every interrupt of the superstep.** If two
   parallel nodes call `interrupt()`, even several seconds apart, the call
   returns once both nodes have reached their `interrupt()` (or finished). The
   result contains **both** interrupt ids. It never returns "only the first
   one" and leaves the second running in the background.
2. **Resuming one interrupt leaves the sibling pending.** A single
   `Command(resume={idA: answer})` wakes only node A. Node A's output is
   persisted, but the graph stays paused on B: `state.next == ("ask_b",)`.
   That one call does **not** start a second parallel run. Resuming B
   afterwards completes the graph normally.
3. **But an outer graph can turn separate answers into concurrent resumes.**
   That is the cat-agent failure. See below.

## What the demos show

### 1. One invoke, two interrupts (`experiment_1_one_invoke`)

```
[simultaneous]                    interrupts returned: 2  (A and B)
[B interrupts 5s after A]         interrupts returned: 2  (A and B)
```

LangGraph does not return from `ainvoke()` while another task of the same
superstep is still runnable. A delayed `interrupt()` only delays the return;
it does not split the result. `seen == []` because neither node got an answer
yet.

### 2. Resume only one (`experiment_2_partial_resume`)

```
before resume        : A, B pending
after resuming A     : seen=['A=answer-A']  still pending=[B]  next=('ask_b',)
after resuming B too : seen=['A=answer-A', 'B=answer-B']  next=()
```

The first answer is not lost, and B stays pending. This is the behaviour you
want: a partial resume is safe *as long as nothing else runs concurrently*.

### 3. The cat-agent shape (`experiment_3_outer_inner`)

Real topology:

```
outer graph
  └── step_node (or orchestrator_node)
        └── inner graph: a tool batch submits two jobs, both call interrupt()
              → two inner interrupts
  └── interrupt_node × 2      (one outer task per inner interrupt)
        ↑ answers arrive separately (two jobmanager callbacks)
```

LangGraph runs the next node only after **all** tasks of the current superstep
have finished. So when the first callback answers `interrupt_node` A, A's
`Command` is written, but nothing is dispatched while B is still paused. When
the second callback answers B, both completions are applied in the same
superstep.

Now compare the two ways of resuming the inner graph:

| mode | after both callbacks | inner resumes | result |
| --- | --- | --- | --- |
| `naive` | paused on **two new** interrupt ids | 2 (one answer each) | stall |
| `collector` | `next=()` | 1 (both answers) | step finishes |

In `naive` mode, each resume starts from the same inner checkpoint, answers
one question, sees the sibling still pending, and **re-raises it**. Both runs
return an interrupt for the sibling; `step_node` forwards them as new outer
interrupts. The two original callbacks were consumed, and the new outer ids
were never registered with the jobmanager, so no callback will answer them.
This is not hypothetical: the 2026-11-26 `case1` run stopped exactly here,
with both job results already `delivered` and the inner thread still at
step 149 holding both `__resume__` writes.

The collector mode is the fix now used by cat-agent:

* `interrupt_node` records its one answer in
  `state["pending_resumes"]["stepnode:<step_id>"]` (or `"orchestrator"`) and
  goes to `resume_collector`;
* `resume_collector_node` unions the answers of each group and emits **one**
  `Send("step_node", {"interrupt_resume": {id1: v1, id2: v2, ...}})` (or one
  `Send("orchestrator_node", ...)`);
* the inner graph sees every pending interrupt answered in a single
  `Command(resume=...)` and advances one superstep.

See `src/catagent/agent.py` (`interrupt_node`, `resume_collector_node`) and
`src/catagent/workflow/models.py` (`pending_resume_reducer`).

### 4. The orchestrator can batch `pause_to_ask` (`experiment_4_pause_to_ask`)

`pause_to_ask` is an ordinary tool. A model turn may contain three of them; the
tool node then parks three interrupts in one superstep:

```
one ainvoke surfaced 3 pause_to_ask interrupts
final next=()  tool replies=['answer=answer-0', 'answer=answer-1', 'answer=answer-2']
```

The orchestrator therefore has the same shape as the StepNode case. Its group
key is `"orchestrator"`, and the collector merges the answers before resuming
the inner orchestrator once.

## Why the demo pins the checkpointer reads

`InMemorySaver` returns a checkpoint synchronously. Two concurrent resumes
often happen to serialize: the second one reads the checkpoint the first one
just wrote, and everything looks fine. The production checkpointer is SQLite,
where the read and the write are separate I/O operations; both resumes can
load the same checkpoint before either writes.

`FrozenReadSaver` in `demo.py` pins the read to one snapshot during the two
callbacks, making the load-before-write window deterministic. It is a teaching
device, not a production checkpointer. The point is not that SQLite is broken;
the point is that **two resumes starting from one checkpoint is inherently
racy**, and cat-agent must never dispatch that shape in the first place.

## Files

| file | purpose |
| --- | --- |
| `demo.py` | the four experiments and the outer/inner harness |
| `test_demo.py` | assertions for all of the above (`MULTI INTERRUPT DEMO: OK`) |
