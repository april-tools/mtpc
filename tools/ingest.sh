rm -f digest.txt
gitingest -e "data/**" -e "*.txt" -e "bin/*" -e "logs/**" -e "*.jsonl" -e "**/*plot*" .
