// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/utils/math/Math.sol";
import "./LPToken.sol";

contract DEX {
    IERC20 public tokenA;
    IERC20 public tokenB;
    LPToken public lpToken;

    uint256 public reserveA;
    uint256 public reserveB;
    uint256 public constant FEE_PERCENT = 3; // 0.3% fee

    event LiquidityAdded(address indexed provider, uint256 amountA, uint256 amountB, uint256 lpTokens);
    event LiquidityRemoved(address indexed provider, uint256 amountA, uint256 amountB, uint256 lpTokens);
    event Swap(address indexed trader, address inputToken, address outputToken, uint256 inputAmount, uint256 outputAmount);

    constructor(address _tokenA, address _tokenB) {
        tokenA = IERC20(_tokenA);
        tokenB = IERC20(_tokenB);
        lpToken = new LPToken("Liquidity Provider Token", "LPT");
    }

    function addLiquidity(uint256 amountA, uint256 amountB) external {
        require(amountA > 0 && amountB > 0, "Amounts must be greater than zero");
        require(
            (reserveA == 0 && reserveB == 0) || 
            (amountA * reserveB == amountB * reserveA), 
            "Invalid ratio"
        );
        (uint256 adjustedA, uint256 adjustedB) = _calculateAdjustedDeposits(amountA, amountB);
        
        tokenA.transferFrom(msg.sender, address(this), adjustedA);
        tokenB.transferFrom(msg.sender, address(this), adjustedB);

        uint256 lpTokensToMint;
        if (reserveA == 0 && reserveB == 0) {
            lpTokensToMint = _sqrt(adjustedA * adjustedB);
        } else {
            lpTokensToMint = Math.min(
                (adjustedA * lpToken.totalSupply()) / reserveA,
                (adjustedB * lpToken.totalSupply()) / reserveB
            );
        }

        reserveA += adjustedA;
        reserveB += adjustedB;

        lpToken.mint(msg.sender, lpTokensToMint);

        // Refund excess tokens if any
        if (adjustedA < amountA) {
            tokenA.transfer(msg.sender, amountA - adjustedA);
        }
        if (adjustedB < amountB) {
            tokenB.transfer(msg.sender, amountB - adjustedB);
        }

        emit LiquidityAdded(msg.sender, adjustedA, adjustedB, lpTokensToMint);
    }

    function _calculateAdjustedDeposits(uint256 amountA, uint256 amountB) internal view returns (uint256, uint256) {
        if (reserveA == 0 && reserveB == 0) {
            return (amountA, amountB);
        }
        
        uint256 expectedB = (amountA * reserveB) / reserveA;
        if (expectedB <= amountB) {
            return (amountA, expectedB);
        }
        
        uint256 expectedA = (amountB * reserveA) / reserveB;
        return (expectedA, amountB);
    }

    function _sqrt(uint256 x) internal pure returns (uint256 y) {
        uint256 z = (x + 1) / 2;
        y = x;
        while (z < y) {
            y = z;
            z = (x / z + z) / 2;
        }
    }

    function removeLiquidity(uint256 lpTokens) external {
        require(lpTokens > 0, "LP tokens must be greater than zero");

        uint256 amountA = (lpTokens * reserveA) / lpToken.totalSupply();
        uint256 amountB = (lpTokens * reserveB) / lpToken.totalSupply();

        lpToken.burn(msg.sender, lpTokens);

        reserveA -= amountA;
        reserveB -= amountB;

        tokenA.transfer(msg.sender, amountA);
        tokenB.transfer(msg.sender, amountB);

        emit LiquidityRemoved(msg.sender, amountA, amountB, lpTokens);
    }

    function swap(address inputToken, address outputToken, uint256 inputAmount) external {
        require(inputAmount > 0, "Input amount must be greater than zero");
        require((inputToken == address(tokenA) && outputToken == address(tokenB)) ||
                (inputToken == address(tokenB) && outputToken == address(tokenA)), "Invalid token pair");

        IERC20(inputToken).transferFrom(msg.sender, address(this), inputAmount);

        uint256 fee = (inputAmount * FEE_PERCENT) / 1000;
        uint256 inputAmountAfterFee = inputAmount - fee;

        uint256 outputAmount;
        if (inputToken == address(tokenA)) {
            outputAmount = (reserveB * inputAmountAfterFee) / (reserveA + inputAmountAfterFee);
            reserveA += inputAmountAfterFee;
            reserveB -= outputAmount;
        } else {
            outputAmount = (reserveA * inputAmountAfterFee) / (reserveB + inputAmountAfterFee);
            reserveB += inputAmountAfterFee;
            reserveA -= outputAmount;
        }

        IERC20(outputToken).transfer(msg.sender, outputAmount);

        emit Swap(msg.sender, inputToken, outputToken, inputAmount, outputAmount);
    }

    function getReserves() external view returns (uint256, uint256) {
        return (reserveA, reserveB);
    }

    function getSpotPrice(address inputToken, address outputToken) external view returns (uint256) {
        require((inputToken == address(tokenA) && outputToken == address(tokenB)) ||
                (inputToken == address(tokenB) && outputToken == address(tokenA)), "Invalid token pair");

        if (inputToken == address(tokenA)) {
            return (reserveB * 1e18) / reserveA;
        } else {
            return (reserveA * 1e18) / reserveB;
        }
    }
}