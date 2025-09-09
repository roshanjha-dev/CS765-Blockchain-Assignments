(async () => {
    try {
      console.log("Starting Arbitrage simulation...");
  
      // 1. Load contract artifacts
      const arbitrageMetadata = JSON.parse(
        await remix.call('fileManager', 'getFile', 'contract/artifacts/Arbitrageur.json')
      );
      const arbitrageABI = arbitrageMetadata.abi;
  
      const dexMetadata = JSON.parse(
        await remix.call('fileManager', 'getFile', 'contract/artifacts/DEX.json')
      );
      const dexABI = dexMetadata.abi;
  
      const tokenMetadata = JSON.parse(
        await remix.call('fileManager', 'getFile', 'contract/artifacts/Token.json')
      );
      const tokenABI = tokenMetadata.abi;
  
      // 2. Get accounts
      const accounts = await web3.eth.getAccounts();
      const [admin] = accounts;
  
      // 3. Use your deployed addresses
      const arbitrageAddress = "0x5A34239c7BE527af0ec3D2a61F19a864C3cA95F3"; // Your Arbitrage contract address
      const dex1Address = "0xB22E25cE40d0aA384F6f5580651C2807d849D4A5";
      const dex2Address = "0xA1e8D59A88afaC77C9f9DbCFB9cC322C62cb00D6";
      const tokenAAddress = "0x512293Bf9d0A7f1CEd3962E9823dF1eDA34F24d0";
      const tokenBAddress = "0xdB23C48ABFdF58A1D0ADA9d12aBe8F72904300A4";
  
      // 4. Create contract instances
      const arbitrage = new web3.eth.Contract(arbitrageABI, arbitrageAddress);
      const dex1 = new web3.eth.Contract(dexABI, dex1Address);
      const dex2 = new web3.eth.Contract(dexABI, dex2Address);
      const tokenA = new web3.eth.Contract(tokenABI, tokenAAddress);
      const tokenB = new web3.eth.Contract(tokenABI, tokenBAddress);
  
      console.log("Contracts loaded successfully!");

      const amtA = web3.utils.toWei("1000", "ether");
      const amtB = web3.utils.toWei("2000", "ether");
      const amtnewB = web3.utils.toWei("2100", "ether");

      let gas = await tokenA.methods.transfer(dex1Address, web3.utils.toWei("10000", "ether")).estimateGas({ from: accounts[0] });
      let data = tokenA.methods.transfer(dex1Address, web3.utils.toWei("10000", "ether")).encodeABI();
      await web3.eth.sendTransaction({ from: accounts[0], to: tokenAAddress, gas, data });

      gas = await tokenA.methods.transfer(dex2Address, web3.utils.toWei("10000", "ether")).estimateGas({ from: accounts[0] });
      data = tokenA.methods.transfer(dex2Address, web3.utils.toWei("10000", "ether")).encodeABI();
      await web3.eth.sendTransaction({ from: accounts[0], to: tokenAAddress, gas, data });

      gas = await tokenB.methods.transfer(dex1Address, web3.utils.toWei("10000", "ether")).estimateGas({ from: accounts[0] });
      data = tokenB.methods.transfer(dex1Address, web3.utils.toWei("10000", "ether")).encodeABI();
      await web3.eth.sendTransaction({ from: accounts[0], to: tokenBAddress, gas, data });

      gas = await tokenB.methods.transfer(dex2Address, web3.utils.toWei("10000", "ether")).estimateGas({ from: accounts[0] });
      data = tokenB.methods.transfer(dex2Address, web3.utils.toWei("10000", "ether")).encodeABI();
      await web3.eth.sendTransaction({ from: accounts[0], to: tokenBAddress, gas, data });

      gas = await tokenA.methods.approve(dex1Address, amtA).estimateGas({ from: admin });
      data = tokenA.methods.approve(dex1Address, amtA).encodeABI();
      await web3.eth.sendTransaction({ from: admin, to: tokenAAddress, gas, data });

      gas = await tokenB.methods.approve(dex1Address, amtB).estimateGas({ from: admin });
      data = tokenB.methods.approve(dex1Address, amtB).encodeABI();
      await web3.eth.sendTransaction({ from: admin, to: tokenBAddress, gas, data });

      // Add liquidity
      gas = await dex1.methods.addLiquidity(amtA, amtB).estimateGas({ from: admin });
      data = dex1.methods.addLiquidity(amtA, amtB).encodeABI();
      await web3.eth.sendTransaction({ from: admin, to: dex1Address, gas, data });


      gas = await tokenA.methods.approve(dex2Address, amtA).estimateGas({ from: admin });
      data = tokenA.methods.approve(dex2Address, amtA).encodeABI();
      await web3.eth.sendTransaction({ from: admin, to: tokenAAddress, gas, data });

      gas = await tokenB.methods.approve(dex2Address, amtnewB).estimateGas({ from: admin });
      data = tokenB.methods.approve(dex2Address, amtnewB).encodeABI();
      await web3.eth.sendTransaction({ from: admin, to: tokenBAddress, gas, data });

      // Add liquidity
      gas = await dex2.methods.addLiquidity(amtA, amtnewB).estimateGas({ from: admin });
      data = dex2.methods.addLiquidity(amtA, amtnewB).encodeABI();
      await web3.eth.sendTransaction({ from: admin, to: dex2Address, gas, data });
  
      // Check minProfit setting
      const minProfit = await arbitrage.methods.getMinProfit().call();
      console.log("Current minProfit:", web3.utils.fromWei(minProfit), "tokens");
  
      // Set minProfit to 0 for testing
      await arbitrage.methods.setMinProfit(0).send({ from: admin });
      console.log("Set minProfit to 0 for testing");
      
      // Check reserves before arbitrage
      const reserves1Before = await dex1.methods.getReserves().call();
      const reserves2Before = await dex2.methods.getReserves().call();
      
      console.log("DEX1 Reserves Before:", web3.utils.fromWei(reserves1Before[0]), "A /", web3.utils.fromWei(reserves1Before[1]), "B");
      console.log("DEX2 Reserves Before:", web3.utils.fromWei(reserves2Before[0]), "A /", web3.utils.fromWei(reserves2Before[1]), "B");
      
      console.log("\n=== Testing Arbitrage Direction ===");
      
      // Test both directions
      const amountIn = web3.utils.toWei("10", "ether");
      const profitAtoB = await arbitrage.methods.calculateProfitAtoBtoA(amountIn).call();
      const profitBtoA = await arbitrage.methods.calculateProfitBtoAtoB(amountIn).call();
      
      console.log("Profit A->B->A:", web3.utils.fromWei(profitAtoB), "A");
      console.log("Profit B->A->B:", web3.utils.fromWei(profitBtoA), "B");
      
      console.log("\n=== Preparing for Arbitrage ===");
      
      // Make sure admin has enough tokens
      const adminBalanceA = await tokenA.methods.balanceOf(admin).call();
      const adminBalanceB = await tokenB.methods.balanceOf(admin).call();
      
      console.log("Admin Token A Balance:", web3.utils.fromWei(adminBalanceA));
      console.log("Admin Token B Balance:", web3.utils.fromWei(adminBalanceB));
      
      // For A->B->A path
      if (Number(profitAtoB) > Number(profitBtoA)) {
        console.log("Executing A->B->A arbitrage path");
        
        // Approve tokens
        await tokenA.methods.approve(arbitrageAddress, amountIn).send({ from: admin });
        console.log("Approved Token A for arbitrage contract");
        
        // Execute arbitrage with high gas limit and value
        try {
          const receipt = await arbitrage.methods.performArbitrage(amountIn).send({ 
            from: admin, 
            gas: 3000000
          });
          
          console.log("Arbitrage transaction successful!");
          console.log("Gas used:", receipt.gasUsed);
        } catch (error) {
          console.error("Arbitrage transaction failed:", error.message);
          
          // Try to get specific error
          try {
            await web3.eth.call({
              from: admin,
              to: arbitrageAddress,
              data: arbitrage.methods.performArbitrage(amountIn).encodeABI(),
              gas: 3000000
            });
          } catch (callError) {
            console.error("Specific error:", callError.message);
          }
        }
      } 
      // For B->A->B path
      else {
        console.log("Executing B->A->B arbitrage path");
        
        // Approve tokens
        await tokenB.methods.approve(arbitrageAddress, amountIn).send({ from: admin });
        console.log("Approved Token B for arbitrage contract");
        
        // Execute arbitrage
        try {
          const receipt = await arbitrage.methods.performArbitrage(amountIn).send({ 
            from: admin, 
            gas: 3000000
          });
          
          console.log("Arbitrage transaction successful!");
          console.log("Gas used:", receipt.gasUsed);
        } catch (error) {
          console.error("Arbitrage transaction failed:", error.message);
          
          // Try to get specific error
          try {
            await web3.eth.call({
              from: admin,
              to: arbitrageAddress,
              data: arbitrage.methods.performArbitrage(amountIn).encodeABI(),
              gas: 3000000
            });
          } catch (callError) {
            console.error("Specific error:", callError.message);
          }
        }
      }
      
      // Check reserves after arbitrage
      const reserves1After = await dex1.methods.getReserves().call();
      const reserves2After = await dex2.methods.getReserves().call();
      
      console.log("\nDEX1 Reserves After:", web3.utils.fromWei(reserves1After[0]), "A /", web3.utils.fromWei(reserves1After[1]), "B");
      console.log("DEX2 Reserves After:", web3.utils.fromWei(reserves2After[0]), "A /", web3.utils.fromWei(reserves2After[1]), "B");
      
      console.log("\nSimulation complete!");
    } catch (error) {
      console.error("Global error:", error);
    }
  })();