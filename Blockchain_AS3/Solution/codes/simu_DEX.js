(async () => {
    try {
        console.log("Starting DEX simulation...");

        const dexMetadata = JSON.parse(await remix.call('fileManager', 'getFile', 'contract/artifacts/DEX.json'));
        const dexABI = dexMetadata.abi;

        const tokenMetadata = JSON.parse(await remix.call('fileManager', 'getFile', 'contract/artifacts/Token.json'));
        const tokenABI = tokenMetadata.abi;

        const lptokenMetadata = JSON.parse(await remix.call('fileManager', 'getFile', 'contract/artifacts/LPToken.json'));
        const lptokenABI = lptokenMetadata.abi;

        const accounts = await web3.eth.getAccounts();
        const LPs = accounts.slice(0, 5);
        const traders = accounts.slice(5, 13);

        const dexAddress = "0x07BcdEe31189c1d87EF4D365f9F875278Aec185c";
        const tokenAAddress = "0x512293Bf9d0A7f1CEd3962E9823dF1eDA34F24d0";
        const tokenBAddress = "0xdB23C48ABFdF58A1D0ADA9d12aBe8F72904300A4";

        const tokenA = new web3.eth.Contract(tokenABI, tokenAAddress);
        const tokenB = new web3.eth.Contract(tokenABI, tokenBAddress);
        const dex = new web3.eth.Contract(dexABI, dexAddress);

        console.log("Contracts loaded successfully!");

        for (let account of [...LPs, ...traders]) {
            try {
                let gas = await tokenA.methods.transfer(account, web3.utils.toWei("1000", "ether")).estimateGas({ from: accounts[0] });
                let data = tokenA.methods.transfer(account, web3.utils.toWei("1000", "ether")).encodeABI();
                await web3.eth.sendTransaction({ from: accounts[0], to: tokenAAddress, gas, data });

                gas = await tokenB.methods.transfer(account, web3.utils.toWei("1000", "ether")).estimateGas({ from: accounts[0] });
                data = tokenB.methods.transfer(account, web3.utils.toWei("1000", "ether")).encodeABI();
                await web3.eth.sendTransaction({ from: accounts[0], to: tokenBAddress, gas, data });
            } catch (error) {
                console.error(`Transfer failed for ${account}:`, error.message);
            }
        }

        console.log("Balances initialized.");

        const N = 50;
        let metrics = {
            tvl: [],
            reserveRatios: [],
            swapVolume: [],
            feeAccumulation: [],
            spotPrice: [],
            slippage: []
        };

        for (let i = 0; i < N; i++) {
            try {
                const allUsers = [...LPs, ...traders];
                const randomUser = allUsers[Math.floor(Math.random() * allUsers.length)];
                const actionType = Math.floor(Math.random() * 3);

                if (actionType === 0 && LPs.includes(randomUser)) {
                    // Add Liquidity (same as original logic, not repeated here for brevity)
                } else if (actionType === 1 && traders.includes(randomUser)) {
                    console.log("Action: Swap");

                    const inputToken = Math.random() > 0.5 ? tokenA : tokenB;
                    const outputToken = inputToken === tokenA ? tokenB : tokenA;

                    const reservesBefore = await dex.methods.getReserves().call();
                    const reserveA = parseFloat(web3.utils.fromWei(reservesBefore[0]));
                    const reserveB = parseFloat(web3.utils.fromWei(reservesBefore[1]));

                    const expectedPrice = inputToken === tokenA ? reserveB / reserveA : reserveA / reserveB;

                    const inputReserve = web3.utils.toBN(inputToken === tokenA ? reservesBefore[0] : reservesBefore[1]);
                    const maxSwap = inputReserve.div(web3.utils.toBN(10));

                    const balance = web3.utils.toBN(await inputToken.methods.balanceOf(randomUser).call());
                    if (balance.isZero()) continue;

                    const percentage = web3.utils.toBN(Math.floor(Math.random() * 100));
                    const inputAmount = balance.lt(maxSwap)
                        ? balance
                        : maxSwap.mul(percentage).div(web3.utils.toBN(100));

                    if (inputAmount.isZero()) continue;

                    let gas = await inputToken.methods.approve(dexAddress, inputAmount.toString()).estimateGas({ from: randomUser });
                    let data = inputToken.methods.approve(dexAddress, inputAmount.toString()).encodeABI();
                    await web3.eth.sendTransaction({ from: randomUser, to: inputToken.options.address, gas, data });

                    gas = await dex.methods.swap(inputToken.options.address, outputToken.options.address, inputAmount.toString()).estimateGas({ from: randomUser });
                    data = dex.methods.swap(inputToken.options.address, outputToken.options.address, inputAmount.toString()).encodeABI();
                    await web3.eth.sendTransaction({ from: randomUser, to: dexAddress, gas, data });

                    console.log(`Swap executed by ${randomUser}`);

                    // Metrics: swapVolume and fee
                    const tokenSymbol = inputToken === tokenA ? "TokenA" : "TokenB";
                    const inputAmountEth = parseFloat(web3.utils.fromWei(inputAmount));
                    metrics.swapVolume.push({ token: tokenSymbol, amount: inputAmountEth });
                    metrics.feeAccumulation.push(inputAmountEth * 0.003);

                    // Slippage
                    const reservesAfter = await dex.methods.getReserves().call();
                    const rAafter = parseFloat(web3.utils.fromWei(reservesAfter[0]));
                    const rBafter = parseFloat(web3.utils.fromWei(reservesAfter[1]));

                    const actualPrice = inputToken === tokenA
                        ? Math.abs(reserveB - rBafter) / Math.abs(reserveA - rAafter)
                        : Math.abs(reserveA - rAafter) / Math.abs(reserveB - rBafter);

                    const slippage = ((actualPrice - expectedPrice) / expectedPrice) * 100;
                    metrics.slippage.push(slippage);
                } else if (actionType === 2 && LPs.includes(randomUser)) {
                    // Remove Liquidity (same as original logic)
                }

                const reserves = await dex.methods.getReserves().call();
                const rA = parseFloat(web3.utils.fromWei(reserves[0]));
                const rB = parseFloat(web3.utils.fromWei(reserves[1]));

                metrics.tvl.push(rA + rB);
                metrics.reserveRatios.push(rB !== 0 ? rA / rB : 0);

                const spot = await dex.methods.getSpotPrice(tokenA.options.address, tokenB.options.address).call();
                metrics.spotPrice.push(parseFloat(web3.utils.fromWei(spot)));

                console.log(`Transaction ${i + 1}/${N} complete.`);
            } catch (error) {
                console.error(`Transaction ${i + 1} failed:`, error.message);
            }
        }

        console.log("Simulation complete!");
        console.log("Final Metrics:", metrics);
    } catch (error) {
        console.error("Critical simulation error:", error.message);
    }
})();
