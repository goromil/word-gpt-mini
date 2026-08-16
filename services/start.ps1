# Launch training inside WSL2 from Task Scheduler (run as SYSTEM)
# WSL auto-starts if not already running. nohup keeps training alive after this script exits.

wsl -d MACUBE -u george nohup bash /home/george/source/ai/word-gpt-mini/services/start.sh >/dev/null 2>&1
