// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";

contract LPToken is ERC20 {
    address public dex;

    constructor(string memory name, string memory symbol) ERC20(name, symbol) {
        dex = msg.sender; // Only the DEX contract can mint/burn LP tokens
    }

    modifier onlyDEX() {
        require(msg.sender == dex, "Only DEX can call this function");
        _;
    }

    function mint(address to, uint256 amount) external onlyDEX {
        _mint(to, amount);
    }

    function burn(address from, uint256 amount) external onlyDEX {
        _burn(from, amount);
    }
}