
# APAC HPC AI Competition 2024 - UPM Team3

Welcome to our project submission for the APAC HPC AI Competition 2024. This README file provides an overview of our project, the methodologies used, and the results obtained.

# Team Members
* Anis Humaira Azman - Team leader
* Nurul Farizatul Aina Mohammad Farizal
* Maimunah Hosni
* Wan Siti Aisyah Wan Kadir
* Siti Nurinsyirah Pahrin
* Azyan Syazwani Setia

# Project Description
Our project for the APAC HPC AI Competition 2024 focuses on the integration and optimization of HOOMD-blue and LLaMA2 LitGPT on the NCI and NSCC SG supercomputers.

# Objectives
The objective is to leverage these advanced computational tools and high-performance computing resources to solve complex problems in molecular dynamics and natural language processing efficiently, with a primary focus on achieving maximum optimization.

# Methodology
[HOOMD Blue]
* Calculate efficiency using total cores and speedup.
* Try allocating different cores and nodes to increase the number of steps.
* Result: Comparison on different number of nodes.

[Llama2]
* Optimization on MPI communication settings.
* Result: Comparison on training time.
  
#
* Programming Languages: Python
* HPC Resources: Gadi HPC system, Aspire2U NSCC

# Results
[HOOMD Blue]
* Steps per second increased as more nodes and cores were used, reflecting improved performance with additional computational resources.
* Execution time decreased with the addition of nodes and cores, indicating that using more nodes helps speed up task processing.
* The system’s speedup tends to increase as more cores are used, since tasks are divided and processed in parallel, reducing the overall time required.
* As more cores are added, the system’s efficiency decreases. While additional cores can speed up processing, they do not necessarily improve efficiency, as more resources are spent on managing the extra cores rather than on productive work.


# Challenges
Discuss any challenges or obstacles faced during the project and how they were addressed.

# Conclusion
Summarize the key findings and outcomes of your project.

