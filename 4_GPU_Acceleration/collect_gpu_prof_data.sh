/usr/bin/time -f "%e" -o wall.txt python $1
rocprof -i metrics_input.txt --stats --basenames on -o results.csv python $1
