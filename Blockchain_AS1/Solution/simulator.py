import hashlib
import time
import random
import threading
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import networkx as nx


class Transaction:
    def __init__(self, sender, receiver, amount, is_coinbase=False):
        self.sender = sender
        self.receiver = receiver
        self.amount = amount
        self.timestamp = time.time()
        self.is_coinbase = is_coinbase
        self.txid = self.generate_txid()
    
    # Generate a unique transaction ID using SHA-256 hashing
    def generate_txid(self): 
        return hashlib.sha256(f"{self.sender}{self.receiver}{self.amount}{self.timestamp}".encode()).hexdigest()

    # String representation of the transaction
    def __repr__(self):  
        if self.is_coinbase:
            return f"Tx({self.txid[:6]}: Coinbase - peer {self.receiver} mines 50 BTC)"
        return f"Tx({self.txid[:6]}: peer {self.sender} pays peer {self.receiver} {self.amount} BTC)"
    
    # Typical Bitcoin transaction size
    def size_in_bytes(self):
        return 250  

class Block:
    MAX_SIZE_BYTES = 1_000_000  # 1 MB
    EMPTY_BLOCK_SIZE = 1_000    # 1 KB

    def __init__(self, index, previous_hash, transactions, miner_id):
        self.index = index
        self.previous_hash = previous_hash
        self.transactions = transactions
        self.timestamp = time.time()
        self.miner_id = miner_id
        self.nonce = 0
        self.hash = self.calculate_hash()  # Compute the initial hash of the block
        self.block_id = self.generate_block_id()  # Generate a unique block identifier
        
    # Generates a unique block ID using SHA-256 hashing
    def generate_block_id(self):  
        return hashlib.sha256(f"{self.index}{self.timestamp}{self.miner_id}".encode()).hexdigest()
    
    # Computes the hash of the block based on its attributes
    def calculate_hash(self):   
        block_data = f"{self.index}{self.previous_hash}{self.transactions}{self.timestamp}{self.nonce}"
        return hashlib.sha256(block_data.encode()).hexdigest()
    
    # Returns the total size of the block, including transactions
    def get_size(self):   
        transaction_size = sum(tx.size_in_bytes() for tx in self.transactions)
        return max(self.EMPTY_BLOCK_SIZE, transaction_size)
    
    # Checks if the block size does not exceed the maximum allowed size
    def validate_size(self):   
        return self.get_size() <= self.MAX_SIZE_BYTES
    
    # Returns a readable string representation of the block
    def __repr__(self):    
        return f"Block({self.index}, Miner: {self.miner_id}, Hash: {self.hash[:6]})"

class Blockchain:
    def __init__(self, difficulty=2):
        self.chain = [self.create_genesis_block()]
        self.lock = threading.Lock()
        self.difficulty = difficulty
        self.orphaned_blocks = {}
        self.confirmed_transactions = set()
        self.pending_chains = {}
    
    # Creates and returns the first block in the blockchain
    def create_genesis_block(self):
        return Block(0, "0", [], "Genesis")
        
    # Adds a new block to the blockchain if valid
    def add_block(self, block):
        with self.lock:
            if block.previous_hash == self.get_latest_block().hash:   # Directly connects to main chain
                self.chain.append(block)
                self._try_add_orphaned_blocks(block.hash)
                for tx in block.transactions:
                    self.confirmed_transactions.add(tx.txid)   # Mark transactions as confirmed
                return True
            
            # Detect potential fork
            if block.previous_hash in [b.hash for b in self.chain]:
                # This could be a fork
                alternative_chain = [b for b in self.chain if b.hash == block.previous_hash]
                alternative_chain.append(block)
                
                # Store potential alternative chain
                self.pending_chains[block.hash] = alternative_chain
                
                # Resolve fork if alternative chain is longer
                if len(alternative_chain) > len(self.chain):
                    self._resolve_fork(alternative_chain)
                
                return False
            
            # Store as orphaned block if no direct connection found
            self.orphaned_blocks.setdefault(block.previous_hash, []).append(block)
            return False

    # Resolve blockchain fork by replacing the main chain with the longest valid alternative chain
    def _resolve_fork(self, alternative_chain):
        if self._validate_chain(alternative_chain):
            self.chain = alternative_chain   # Replace the main chain
            
            # Clear confirmed transactions from the replaced chain
            self.confirmed_transactions.clear()
            
            # Re-add confirmed transactions from the new chain
            for block in self.chain:
                for tx in block.transactions:
                    self.confirmed_transactions.add(tx.txid)
    
    # Validate the integrity of a potential blockchain
    def _validate_chain(self, chain):
        for i in range(1, len(chain)):
            # Check if each block correctly points to the previous block
            if chain[i].previous_hash != chain[i-1].hash:
                return False
        return True

    # Tries to attach orphaned blocks to the chain
    def _try_add_orphaned_blocks(self, parent_hash):
        if parent_hash in self.orphaned_blocks:
            for orphan in self.orphaned_blocks[parent_hash]:
                self.add_block(orphan)
            del self.orphaned_blocks[parent_hash]

    # Printing blockchain details for debugging purposes
    def print_chain_details(self):
        print("\nBlockchain Details:")
        for i, block in enumerate(self.chain):
            print(f"Block {i}: {block}")
            for tx in block.transactions:
                print(f"  {tx}")
        
        print("\nOrphaned Blocks:")
        for prev_hash, blocks in self.orphaned_blocks.items():
            print(f"Orphaned under {prev_hash[:6]}:")
            for block in blocks:
                print(f"  {block}")
    
    # Returns the latest block in the blockchain
    def get_latest_block(self):
        return self.chain[-1]
    
    # Returns the total number of blocks in the blockchain
    def get_chain_length(self):
        return len(self.chain)

class Peer:
    def __init__(self, peer_id, blockchain, speed, cpu_type, Ttx):   # Initializes a Peer
        self.peer_id = peer_id
        self.blockchain = blockchain
        self.speed = speed
        self.cpu_type = cpu_type
        self.mempool = []
        self.Ttx = Ttx
        self.running = True
        self.balance = 100 #random.randint(10, 100)
        self.peers_connected = set()
        self.latency_dict = {}
        self.tx_history = {}
        self.tx_history_lock = threading.Lock()
        self.mining_event = threading.Event()
        self.block_history = set()
        self.blockchain_tree = []
        self.raw_hash_power = 0.1 if cpu_type == "low CPU" else 1.0        
        
    # Establishes a connection with another peer in the network
    def add_connection(self, peer):
        self.peers_connected.add(peer)

    # Calculates the network latency between this peer and another peer based on network conditions and message size
    def calculate_latency(self, peer, message_length):
        rho_ij = random.uniform(0.01, 0.5)
        c_ij = 100 * 1e6 if self.speed == "fast" and peer.speed == "fast" else 5 * 1e6
        mean_dij = 96 * 1e3 / c_ij
        d_ij = np.random.exponential(mean_dij)
        latency = rho_ij + (message_length / c_ij) + d_ij
        return latency

    # Handles incoming transactions by verifying and propagating them to connected peers while ensuring no duplicate transactions are processed
    def receive_transaction(self, transaction, sender_peer):
        if transaction.txid in self.tx_history:
            return
        latency = self.latency_dict.get(transaction.sender, 0.1)
        time.sleep(latency)
        self.mempool.append(transaction)
        # for m in self.mempool:
        #     print(f"{self.peer_id}: {m.txid}")
        with self.tx_history_lock:
            if transaction.txid not in self.tx_history:
                self.tx_history[transaction.txid] = set()
            self.tx_history[transaction.txid].add(self.peer_id)
            self.tx_history[transaction.txid].add(sender_peer.peer_id)
        for peer in self.peers_connected:
            if peer != sender_peer:
                peer.receive_transaction(transaction, self)


    # Processes an incoming block by verifying its validity and adding it to the blockchain. If valid, propagates the block to connected peers
    def receive_block(self, block, sender_peer):
        # If we've already seen this block, stop propagation
        if block.block_id in self.block_history:
            return False
            
        # Add to block history first to prevent cycles
        self.block_history.add(block.block_id)
        
        if not block.validate_size():
            return False
        
        # Validate each transaction in the block
        for tx in block.transactions:
            if not tx.is_coinbase and not self.validate_transaction(tx):
                return False
                
        time_ = time.time()
        self.blockchain_tree.append([time_, block])

        # Simulate network delay based on block size
        latency = self.calculate_latency(sender_peer, block.get_size() * 8)
        time.sleep(latency)
        
        success = self.blockchain.add_block(block)
        if success:
            self.mining_event.set()
            self.start_mining()
  
        # Only propagate to peers who haven't seen the block
        for peer in self.peers_connected:
            if peer != sender_peer and block.block_id not in peer.block_history:
                peer.receive_block(block, self)
                
        return success

    # Checks if a transaction is valid by ensuring the sender has enough balance.
    def validate_transaction(self, tx):
        sender_balance = self.calculate_balance(tx.sender)
        return sender_balance >= tx.amount

    # Computes the balance of a given peer by iterating through the blockchain and summing mining rewards and transaction history.
    def calculate_balance(self, peer_id):
        balance = 0

        # Add mining rewards
        for block in self.blockchain.chain:
            if block.miner_id == peer_id:
                balance += 50  # Mining reward

        # Process transaction history
        for block in self.blockchain.chain:
            for tx in block.transactions:
                if tx.sender == peer_id:
                    balance -= tx.amount
                if tx.receiver == peer_id:
                    balance += tx.amount
        return balance
    
    # Starts a new mining thread if the peer is still running.
    def start_mining(self):
        if not self.running:
            return
        mining_thread = threading.Thread(target=self.mine_new_block)
        mining_thread.start()


    # Mines a new block by selecting valid transactions, computing the proof-of-work, and adding the block to the blockchain.
    def mine_new_block(self):
        delay_factor = 10 if self.cpu_type == "low CPU" else 1
        current_length = self.blockchain.get_chain_length()
        I = 10
        Tk = np.random.exponential(I / self.hash_power)
        #time.sleep(Tk)

        # Simulate initial processing delay
        # time.sleep(random.randint(0,5))

        # Ensure no new block has been added in the meantime
        if current_length == self.blockchain.get_chain_length():
            coinbase_tx = Transaction(None, self.peer_id, 50, is_coinbase=True)
            valid_transactions = [coinbase_tx]
            current_size = coinbase_tx.size_in_bytes()
            
            # Filter valid transactions from mempool
            mempool_copy = self.mempool.copy()
            used_transactions = set()
            
            for tx in mempool_copy:
                if (self.validate_transaction(tx) and 
                    tx.txid not in self.blockchain.confirmed_transactions and 
                    tx.txid not in used_transactions):
                    
                    tx_size = tx.size_in_bytes()
                    if current_size + tx_size <= Block.MAX_SIZE_BYTES:
                        valid_transactions.append(tx)
                        current_size += tx_size
                        used_transactions.add(tx.txid)
                        self.mempool.remove(tx)
            
            new_block = Block(
                len(self.blockchain.chain),
                self.blockchain.get_latest_block().hash,
                valid_transactions,
                self.peer_id
            )

            # Proof-of-work loop
            while new_block.hash[:self.blockchain.difficulty] != "0" * self.blockchain.difficulty:
                new_block.nonce += 1
                new_block.hash = new_block.calculate_hash()
                if new_block.nonce % 1000 == 0:
                    time.sleep(0.001 * delay_factor)
            
            # Add to own block history before propagating
            self.block_history.add(new_block.block_id)
            self.blockchain.add_block(new_block)
            
            # Propagate to peers
            for peer in self.peers_connected:
                if new_block.block_id not in peer.block_history:
                    peer.receive_block(new_block, self)
        
    # Periodically generates and propagates transactions to connected peers.
    def generate_transactions(self, peers):
        while self.running:
            receiver_id = random.choice([p.peer_id for p in peers if p.peer_id != self.peer_id])
            amount = random.randint(1, self.balance)
            if amount > 0:
                tx = Transaction(self.peer_id, receiver_id, amount)
                for peer in self.peers_connected:
                    self.latency_dict[peer.peer_id] = self.calculate_latency(peer, len(str(tx).encode()) * 8)
                    peer.receive_transaction(tx, peer)
                
                print(f"Peer {self.peer_id} generated {tx} sent to peers {[p.peer_id for p in self.peers_connected]}")
            time.sleep(np.random.exponential(self.Ttx))    

    # Stops the peer from running and signals any ongoing mining process.
    def stop(self):
        self.running = False
        self.mining_event.set()

# Check if all peers are connected in the network
def is_connected(peers):
    if not peers:
        return False

    visited = set()
    queue = deque([peers[0]])

    while queue:
        peer = queue.popleft()
        if peer not in visited:
            visited.add(peer)
            queue.extend(peer.peers_connected - visited)

    return len(visited) == len(peers)

# Create a network of peers with random connections
def create_peers(n, z0, z1, Ttx, blockchain):
    peers = []
    num_slow = int(n * z0 / 100)
    num_low_cpu = int(n * z1 / 100)
    # Create all peers first
    for i in range(n):
        peer = Peer(
            peer_id=i,
            blockchain=blockchain,
            speed = "slow" if i < num_slow else "fast",
            cpu_type = "low CPU" if i < num_low_cpu else "high CPU",
            Ttx=Ttx
        )
        peers.append(peer)
    
    # Now connect peers with limited connections
    for i in range(n):
        # Determine random number of connections (3-6)
        num_connections = random.randint(3, 6)
        
        # Get possible peers to connect to (excluding self)
        possible_peers = [p for p in peers if p.peer_id != i]
        
        # If peer already has max connections, skip
        if len(peers[i].peers_connected) >= 3:
            continue
            
        # Select random peers to connect to
        for _ in range(num_connections):
            if possible_peers and len(peers[i].peers_connected) < 6:
                # Select a random peer that isn't at max connections
                available_peers = [p for p in possible_peers 
                                 if len(p.peers_connected) < 6 and 
                                 p not in peers[i].peers_connected]
                
                if not available_peers:
                    break
                    
                new_peer = random.choice(available_peers)
                peers[i].add_connection(new_peer)
                new_peer.add_connection(peers[i])
                possible_peers.remove(new_peer)
    
    # Ensure network is connected
    if not is_connected(peers):
        # Add minimum connections to connect the network
        connect_network(peers)
        
    return peers

# Ensure the network is connected by adding minimum necessary connections
def connect_network(peers):
    components = find_components(peers)
    while len(components) > 1:
        comp1 = components[0]
        comp2 = components[1]
        # Connect two components with a single edge
        peer1 = random.choice(list(comp1))
        peer2 = random.choice(list(comp2))
        if len(peer1.peers_connected) < 6 and len(peer2.peers_connected) < 6:
            peer1.add_connection(peer2)
            peer2.add_connection(peer1)
        components = find_components(peers)

# Find all connected components in the network
def find_components(peers):
    components = []
    visited = set()
    
    for peer in peers:
        if peer not in visited:
            component = set()
            queue = deque([peer])
            while queue:
                current = queue.popleft()
                if current not in component:
                    component.add(current)
                    visited.add(current)
                    queue.extend(p for p in current.peers_connected 
                               if p not in component)
            components.append(component)
    
    return components

# Visualizing the blockchain
def visualize_blockchain(blockchain):
    # Create a directed graph
    G = nx.DiGraph()
    
    # Track block positions and connections
    block_positions = {}
    
    # Function to add blocks to graph with horizontal spacing
    def add_block_to_graph(block, x=0, y=0, seen_x=set()):
        # Adjust x position if already occupied
        while x in seen_x:
            x += 1
        seen_x.add(x)
        
        # Create block label
        block_label = f"Block {block.index}\nMiner: {block.miner_id}\nHash: {block.hash[:10]}"
        
        # Add node
        G.add_node(block.block_id[:10], label=block_label)
        block_positions[block.block_id[:10]] = (x, y)
        
        # Find and connect to parent block
        if block.previous_hash:
            parent_block = next((b for b in blockchain.chain + 
                                 [b for orphans in blockchain.orphaned_blocks.values() for b in orphans] + 
                                 [b for forks in blockchain.pending_chains.values() for b in forks] 
                                 if b.hash == block.previous_hash), None)
            
            if parent_block:
                G.add_edge(parent_block.block_id[:10], block.block_id[:10])
                add_block_to_graph(parent_block, x, y-1, seen_x)
    
    # Tracking set to prevent x-coordinate overlap
    seen_x = set()
    
    # Add main chain blocks
    for i, block in enumerate(blockchain.chain):
        add_block_to_graph(block, x=i, y=0, seen_x=seen_x)
    
    # Add orphaned blocks
    orphan_x = len(blockchain.chain)
    for prev_hash, orphan_blocks in blockchain.orphaned_blocks.items():
        for j, block in enumerate(orphan_blocks):
            add_block_to_graph(block, x=orphan_x + j, y=-j, seen_x=seen_x)
    
    # Add pending chain (fork) blocks
    fork_x = len(blockchain.chain) + len(blockchain.orphaned_blocks)
    for fork_hash, fork_chain in blockchain.pending_chains.items():
        for k, block in enumerate(fork_chain):
            add_block_to_graph(block, x=fork_x + k, y=-k, seen_x=seen_x)
    
    # Create plot with explicit figure and axes
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # Draw the graph
    nx.draw(G, pos=block_positions, ax=ax, with_labels=False, 
            node_color='lightblue', 
            node_size=3000, 
            arrowsize=20, 
            arrows=True)
    
    # Add node labels
    node_labels = nx.get_node_attributes(G, 'label')
    nx.draw_networkx_labels(G, pos=block_positions, ax=ax, labels=node_labels, font_size=8)
    
    ax.set_title('Blockchain Structure Visualization')
    ax.axis('off')
    
    # Save the plot 
    plt.savefig('blockchain_structure.png', dpi=300, bbox_inches='tight')
    print("Blockchain visualization saved as 'blockchain_structure.png'")
    plt.close(fig)

def normalise_hash_powers(peers):
    # Normalize hash power
    total_raw_power = sum(peer.raw_hash_power for peer in peers)
    for p in peers:
        p.hash_power = p.raw_hash_power / total_raw_power

# Simulation Parameters
n = 25  # Number of peers
z0 = 20  # 20% of peers are slow
z1 = 20  # 20% of peers are low CPU
Ttx = 30  # Mean transaction interarrival time in seconds
blockchain = Blockchain()

# Create Peers with a connected network topology
peers = create_peers(n, z0, z1, Ttx, blockchain)
normalise_hash_powers(peers)

# Start Mining and Transaction Generation
mining_threads = []
tx_threads = []

for peer in peers:
    tx_thread = threading.Thread(target=peer.generate_transactions, args=(peers,))
    tx_threads.append(tx_thread)
    tx_thread.start()

# Run simulation for 30 seconds
try:
    time.sleep(30)
finally:
    for peer in peers:
        peer.stop()
    for thread in tx_threads:
        thread.join()

for peer in peers:
    if peer.mining_event:
        peer.mining_event.clear()  # Reset the event
    mining_thread = threading.Thread(target=peer.mine_new_block)
    mining_threads.append(mining_thread)
    mining_thread.start()
    peer.start_mining()  # Start initial mining

try:
    time.sleep(30)
finally:
    for peer in peers:
        peer.stop()
    for thread in mining_threads:
        thread.join()

# Print Blockchain
print("\nFinal Blockchain:")
for block in blockchain.chain:
    print(f"\n{block}")
    for tx in block.transactions:
        print(f"  {tx}")

# Print blockchain tree of each peer to file
with open("blockchain_tree.txt", "w") as file:
    for p in peers:
        file.write("Peer: " + str(p.peer_id) + "\n") 
        for i in p.blockchain_tree:
            file.write(" ".join(map(str, i)) + "\n")  
        file.write("\n")  

# visualize blockchain
visualize_blockchain(blockchain)
