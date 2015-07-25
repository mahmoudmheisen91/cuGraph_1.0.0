# cuGraph_1.0.0:
*Highly efficient parallel algorithms for random graph generation on CUDA platform based on Erdős–Rényi model.*

### Introduction:
**cuGraph** is a simple framework for fastly generating random graphs with the help of **GPU** and export them to different file formats such as GML and MTX, the three main algorithms are:
- [ ] **PER**:**P**arallel **E**rdős–**R**ényi model.
- [X] **PZER**: **P**arallel **Z** (skip) **E**rdős–**R**ényi model.
- [ ] **prePZER**: **pre** (pre log) **P**arallel **Z** (skip) **E**rdős–**R**ényi model.

These algorithms are designed and published by Sadegh Nobari, Xuesong Lu, Panagiotis Karras and Stéphane Bressan, [Link](http://icdt.tu-dortmund.de/proceedings/edbticdt2011proc/WebProceedings/papers/edbt/a30-nobari.pdf), the fastest algorithm is prePZER.

Average achived __Speed Up__ is __X75 to X80__ over the three sequential versions of these algorithms, namely:
- [X] **ER**: **E**rdős–**R**ényi model.
- [x] **ZER**: **Z** (skip) **E**rdős–**R**ényi model.
- [X] **preZER**: **pre** (pre log) **Z** (skip) **E**rdős–**R**ényi model.

> NOTE:

> The GPU used is GeForce GTX 570.

> Global Memory size: 1 GB.

> Compute compatibility: 2.0.

> CUDA Version: 5.0

The **cuGraph** framework is capable of generating a graph with up to **30K** vertices with maximum **200M** edge on *1 GB* GPU under 80 ms, the core algorithm that make these algorithms fast is the **Scan** algorithm.

The Graph is internally represented by Adjancy Matrix data structure to expose massive parallelism, the scan algorithm is adopted from [here](https://research.nvidia.com/sites/default/files/publications/nvr-2008-003.pdf).


### Compile:
1. Download the project from [here](https://github.com/mahmoudmheisen91/cuGraph_1.0.0/archive/master.zip).
2. unzip the file:

   ```
   tar -xzvf cuGraph_1.0.0-master.zip
   ```
3. go to the directory where you downloaded the project:

   ```
   cd cuGraph_1.0.0-master
   ```
4. make the project:

   ```
   make
   ```


### Run:
1. *Option 1*: To run the project with predefined values, use the predefined make command:

   ```
   make run
   ``` 

2. *Option 2*: To run the project with your values of graph size and algorithm type:
   * Go to executable directory:
   	 ```
     cd output/bin
     ```
   * run as follow:
   
    ```
    ./cuGraph_1.0.0 <graph_vertices> <maximum_edges> <probability_of_edge> <algorithm> <file_name>
    ```


> LIMITATIONS:

> graph\_vertices: limited to _30000_ for 1 GB GPU.

> maximum\_edges: limited to graph\_vertices ^ 2.

> probability\_of\_edge: from 0 to less than 1.

> algorithm: ER, ZER, preZER, PER, PZER or prePZER.

> file\_name: allowed extenstions are: .txt, .gml, .mtx or no extention which is basicaly a txt file, the file will be saved in output/type/


### Examples:
1. Example 1: generate a graph with 10000 nodes, 100000000 edges with probability of 0.5, PER algorithm and saved as output/TXT/testrun

   ```
   make run
   ```

2. Example 2: generate a graph with 10000 nodes, 50000000 edges with probability of 0.9, PZER algorithm and saved as output/MTX/graph1.mtx

   ```
   ./cuGraph_1.0.0 10000 50000000 0.9 PZER output/MTX/graph1.mtx
   ```
   
### TODO:
TODO list includes but not limited to:

1. Use shared memory in Scan algorithm.

2. Scan algorithm is limited to 1M element at a time.

3. Use recursive call in Scan algorithm.

4. Use of Generic types in the Project.

5. Create a test branch to represent the graph with adjacany list and test the speedup and memory.

6. Modify PER and prePZER to make use of Scan alorithm.

7. Use low level API with Scan algorithm.

8. Modify the Project to link cuda code with c++ code in one class.

9. Try some parallel read/write to a file.

10. Modify L parameter in the algorithms.

