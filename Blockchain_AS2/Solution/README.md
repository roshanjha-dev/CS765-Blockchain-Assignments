README

1. The following Python libraries must be installed before running the code:
- pip install numpy networkx matplotlib

2. Run the simulator using the following command:
- python3 simulator.py n z0 z1 Ttx x y filename Tt

Parameters:
n → Number of peers
z0 → Percentage of slow peers
z1 → Percentage of low CPU peers (honest peers)
Ttx → Transaction interarrival rate
x → Transaction rate
y → Mining rate
filename → Name of the file for visualization output
Tt → Timeout duration

Example: python3 simulator.py 10 20 30 20 100 100 output.png 30

3. A script run.sh is provided to execute the simulator for various values of Tt, number of peers, and percentage of malicious peers. Run the script using:
- ./run.sh

4. We have implemented a mitigation strategy to counter selfish mining in the network. To run the mitigation-enabled simulation, use:
- python3 simulator_mitigation.py n z0 z1 Ttx x y filename Tt

Parameters:
n → Number of peers
z0 → Percentage of slow peers
z1 → Percentage of low CPU peers (honest peers)
Ttx → Transaction interarrival rate
x → Transaction rate
y → Mining rate
filename → Name of the file for visualization output
Tt → Timeout duration

