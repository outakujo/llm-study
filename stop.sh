ps aux | grep 'python main.py' | grep -v grep | awk '{print $2}' | xargs kill
