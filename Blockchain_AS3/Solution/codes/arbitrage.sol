// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "./Token.sol";
import "./DEX.sol";

contract Arbitrageur {
    DEX public dex1;
    DEX public dex2;
    Token public tokenA;
    Token public tokenB;
    uint256 public minProfit;
    
    event ArbitrageExecuted(address indexed trader, uint256 profit);
    event ProfitCalculated(uint256 profitAtoBtoA, uint256 profitBtoAtoB);
    event DebugSwap(string message, uint256 amount);

    constructor(
        address _dex1,
        address _dex2,
        address _tokenA,
        address _tokenB,
        uint256 _minProfit
    ) {
        dex1 = DEX(_dex1);
        dex2 = DEX(_dex2);
        tokenA = Token(_tokenA);
        tokenB = Token(_tokenB);
        minProfit = _minProfit;
    }

    function getMinProfit() public view returns (uint256) {
        return minProfit;
    }

    function performArbitrage(uint256 amountIn) external {
        // Calculate potential profits in both directions
        uint256 profitAtoBtoA = calculateProfitAtoBtoA(amountIn);
        uint256 profitBtoAtoB = calculateProfitBtoAtoB(amountIn);
        
        emit ProfitCalculated(profitAtoBtoA, profitBtoAtoB);

        // Set minProfit to 0 for testing purposes
        require(
            profitAtoBtoA > minProfit || profitBtoAtoB > minProfit,
            "No profitable opportunity"
        );

        if (profitAtoBtoA >= profitBtoAtoB) {
            executeAtoBtoA(amountIn);
        } else {
            executeBtoAtoB(amountIn);
        }
    }

    function calculateProfitAtoBtoA(uint256 amountIn) public view returns (uint256) {
        (uint256 reserve1A, uint256 reserve1B) = dex1.getReserves();
        (uint256 reserve2A, uint256 reserve2B) = dex2.getReserves();
        
        // A→B on DEX1 (reserveIn = reserve1A, reserveOut = reserve1B)
        uint256 amountAfterFee = (amountIn * 997) / 1000;
        uint256 amountB = getAmountOut(amountAfterFee, reserve1A, reserve1B);
        
        // B→A on DEX2 (reserveIn = reserve2B, reserveOut = reserve2A)
        amountAfterFee = (amountB * 997) / 1000;
        uint256 amountOut = getAmountOut(amountAfterFee, reserve2B, reserve2A);
        
        return amountOut > amountIn ? amountOut - amountIn : 0;
    }

    function calculateProfitBtoAtoB(uint256 amountIn) public view returns (uint256) {
        (uint256 reserve1A, uint256 reserve1B) = dex1.getReserves();
        (uint256 reserve2A, uint256 reserve2B) = dex2.getReserves();
        
        // B→A on DEX1 (reserveIn = reserve1B, reserveOut = reserve1A)
        uint256 amountAfterFee = (amountIn * 997) / 1000;
        uint256 amountA = getAmountOut(amountAfterFee, reserve1B, reserve1A);
        
        // A→B on DEX2 (reserveIn = reserve2A, reserveOut = reserve2B)
        amountAfterFee = (amountA * 997) / 1000;
        uint256 amountOut = getAmountOut(amountAfterFee, reserve2A, reserve2B);
        
        return amountOut > amountIn ? amountOut - amountIn : 0;
    }

    function executeAtoBtoA(uint256 amountIn) internal {
        // Transfer tokens from sender
        require(tokenA.transferFrom(msg.sender, address(this), amountIn), "Transfer from sender failed");
        emit DebugSwap("Received tokens from sender", amountIn);
        
        // Swap A->B on DEX1
        uint256 balanceA = tokenA.balanceOf(address(this));
        require(tokenA.approve(address(dex1), balanceA), "Approve for DEX1 failed");
        dex1.swap(address(tokenA), address(tokenB), balanceA);
        emit DebugSwap("Swapped A->B on DEX1", balanceA);
        
        // Swap B->A on DEX2
        uint256 balanceB = tokenB.balanceOf(address(this));
        require(tokenB.approve(address(dex2), balanceB), "Approve for DEX2 failed");
        dex2.swap(address(tokenB), address(tokenA), balanceB);
        emit DebugSwap("Swapped B->A on DEX2", balanceB);
        
        // Return funds and profit
        uint256 finalBalance = tokenA.balanceOf(address(this));
        require(finalBalance > amountIn, "Arbitrage failed");
        
        require(tokenA.transfer(msg.sender, finalBalance), "Transfer to sender failed");
        emit ArbitrageExecuted(msg.sender, finalBalance - amountIn);
    }

    function executeBtoAtoB(uint256 amountIn) internal {
        // Transfer tokens from sender
        require(tokenB.transferFrom(msg.sender, address(this), amountIn), "Transfer from sender failed");
        emit DebugSwap("Received tokens from sender", amountIn);
        
        // Swap B->A on DEX1
        uint256 balanceB = tokenB.balanceOf(address(this));
        require(tokenB.approve(address(dex1), balanceB), "Approve for DEX1 failed");
        dex1.swap(address(tokenB), address(tokenA), balanceB);
        emit DebugSwap("Swapped B->A on DEX1", balanceB);
        
        // Swap A->B on DEX2
        uint256 balanceA = tokenA.balanceOf(address(this));
        require(tokenA.approve(address(dex2), balanceA), "Approve for DEX2 failed");
        dex2.swap(address(tokenA), address(tokenB), balanceA);
        emit DebugSwap("Swapped A->B on DEX2", balanceA);
        
        // Return funds and profit
        uint256 finalBalance = tokenB.balanceOf(address(this));
        require(finalBalance > amountIn, "Arbitrage failed");
        
        require(tokenB.transfer(msg.sender, finalBalance), "Transfer to sender failed");
        emit ArbitrageExecuted(msg.sender, finalBalance - amountIn);
    }

    function setMinProfit(uint256 _minProfit) external {
        minProfit = _minProfit;
    }

    // Helper function from UniswapV2Library
    function getAmountOut(
        uint256 amountIn,
        uint256 reserveIn,
        uint256 reserveOut
    ) internal pure returns (uint256 amountOut) {
        uint256 amountInWithFee = amountIn * 997;
        uint256 numerator = amountInWithFee * reserveOut;
        uint256 denominator = reserveIn * 1000 + amountInWithFee;
        amountOut = numerator / denominator;
    }
}