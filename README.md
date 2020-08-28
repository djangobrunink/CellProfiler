How to install
1. 	Clone git repo CellProfiler to your local machine from https://github.com/djangobrunink/CellProfiler/tree/django.
	This is the full CellProfiler program with the zebrafish module in place. 

2. 	cd into directory from (1).

3. 	Ensure the following is installed on the machine (if CP was installed earlier, this is likely already in place):
	- Python 3.8 (64-bit)
	- Microsoft Visual C++ 2008 Redistributable Package (https://www.microsoft.com/en-us/download/details.aspx?id=15336) 
		- Its 64-bit SP1 (https://www.microsoft.com/en-us/download/details.aspx?id=26368)
	- C++ Build Tools (https://download.microsoft.com/download/5/f/7/5f7acaeb-8363-451f-9425-68a90f98b238/visualcppbuildtools_full.exe)
	- Java Runtime Environment (https://java.com/en/download/manual.jsp)
	- Java Development Kit version 1.8 (http://www.oracle.com/technetwork/java/javase/downloads/index.html)
Note:	Scroll down to JAVA SE 8u261. The JDK seems to require an Oracle account to install this version.

4. 	Run the following commands:
	- "pip install torch==1.6.0+cpu torchvision==0.7.0+cpu -f https://download.pytorch.org/whl/torch_stable.html"
	- "pip install cython future hypothesis numpy protobuf six mysqlclient"
	- "pip install -e ."

Note: 	If the last step errors out because of python-javabridge, there likely is an error with the JAVA_HOME variable.
	

5. 	Run CellProfiler by running 'cellprofiler' from the command line.