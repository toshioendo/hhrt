# HHRT tools

## Simple Timeline Profiler

Requirement:
* Python
* Excel (it is Ok if your laptop has it. You can scp the profile result to your laptop)

Basic Usage:
* Choose a directory name for profile logs (Here, "aaa").
* `mkdir aaa; export HH_PROF_PATH=./aaa`
  * It is better to make the directory empty, if you reused it.
* Execute your application on HHRT. For example, `mpirun -np 8 ./7pstencil -p 2 4 1024 1024 2048`
* You will see files such as `hhprof-p?.log` in the log directory. A file per process has been made.
* Do `cat aaa/* | tools/hhprofconv.py > aaa.csv`. A CSV file is made.
* Copy the CSV file and `tools/hhprofsample.xlsx` to your laptop (with Excel) if necessary.
* Open the CSV file and hhprofsample.xlsx in Excel.
* Copy and paste all data in the CSV file into hhprofsample.xlsx. (Be careful for old sample data)
* You will see the timeline graph.

