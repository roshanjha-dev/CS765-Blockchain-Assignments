README

1. Folder Structure
- All the source code files are organized under the codes/ folder.

2. File Descriptions

- codes/Token.sol
Defines a standard ERC-20 token contract used for trading on the DEX.

- codes/LPToken.sol
Implements the Liquidity Provider (LP) token, which is minted and burned to represent a user's share in the liquidity pool.

- codes/DEX.sol
Core contract that implements the DEX using a constant product market maker model. Handles swaps, liquidity addition/removal, fee accumulation, and price tracking.

- codes/arbitrage.sol
A smart contract for executing arbitrage opportunities between this DEX and an external market. It attempts to exploit price discrepancies profitably.

- codes/simu_DEX.js
A JavaScript simulation script run on Remix with Web3 integration. It:
a. Initializes account balances.

b. Randomly performs liquidity provision, removal, and token swaps.

c. Records metrics such as TVL, reserve ratios, swap volume, slippage, fees, and spot prices.

- codes/simu_arbitrage.js
This script simulates and tests an arbitrage trading strategy between two DEXes using a deployed Arbitrageur contract.

3. Plotting and Metrics

- plot.py
Used to generate a single image containing multiple graphs based on the simulation data.

- metrics.txt
Contains the recorded metrics output from running simu_DEX.js once. These metrics are used by plot.py to produce the plots.

- plot.png
A single image file that visualizes all the key metrics captured during the simulation.





