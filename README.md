How to install
1. 	Clone git repo CellProfiler to your local machine from https://github.com/djangobrunink/CellProfiler.
	This is the full CellProfiler program with the zebrafish module in place. 

2. 	cd into directory from (1).

3. 	Ensure the following is installed on the machine (if CP was installed earlier, this is likely already in place):
	- Python 3.8 (64-bit)
	- Microsoft Visual C++ 2008 Redistributable Package and its 64-bit SP1
	- C++ Build Tools
	- Java Runtime Environment
	- Java Development Kit

4. 	Run the following commands:
	- "pip install cython future hypothesis numpy protobuf six mysqlclient"
	- "pip install -e ."

5. 	Run CellProfiler by running 'cellprofiler' from the command line.