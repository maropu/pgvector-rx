#!/bin/bash

while true; do
    COMMIT=$(git rev-parse --short=6 HEAD)
    LOGFILE="agent_logs/agent_${COMMIT}.log"

    copilot -p "$(cat AGENT_PROMPT.md)" \
            --model claude-opus-4.6 \
            --no-ask-user \
            --yolo \
            &> "$LOGFILE"

    if [ $? -ne 0 ]; then
        echo "Agent failed at commit $COMMIT" >> agent_errors.log
    fi

    sleep 3
done
