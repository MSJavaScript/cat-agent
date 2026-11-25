#!/usr/bin/env bash
source /mnt/t/softwares_install/lang-graph/bin/activate
export PYTHONPATH="/mnt/t/programming/myagents/cat-agent-dev/src:$PYTHONPATH"
python -m catagent.app --serve \
    --dotenv /home/luandong/bin/.env \
    --db /mnt/t/workdir/agent_test/db/catagent_checkpoints2.db \
    --skill-index-json src/catagent/skills/skill_index.json \
    --host 127.0.0.1 --port 5600 \
    --job-manager-url http://localhost:8123  --log-level warning \
    --log /mnt/t/workdir/agent_test/db/catagent.log

