#!/bin/bash

# Autonomous agent loop inspired by Anthropic's engineering blog post:
# "Building a C Compiler in 5 Days with Claude Opus 4"
# https://www.anthropic.com/engineering/building-c-compiler

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
