import hashlib
import argparse
import time
import math
import random
import heapq
import copy
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
from collections import deque
import threading
from datetime import datetime

class Event:
    def __init__(self, time, event_type, peer, data=None):
        self.time = time
        self.event_type = event_type
        self.peer = peer
        self.data = data

    def __lt__(self, other):
        # priority = {
        #     "broadcast_private_chain": 0,  # Highest priority
        #     "handle_get": 1,              # Second highest priority
        #     "mine_block": 2
        # }
        
        # # Assign default priority for unknown events (lower priority)
        # self_priority = priority.get(self.event_type, 2)
        # other_priority = priority.get(other.event_type, 2)

        # if self_priority != other_priority:
        #     return self_priority < other_priority  # Lower value means higher priority

        return self.time < other.time  # If same priority, sort by time


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
        genesis_block = self.create_genesis_block()
        # Changed from list to dictionary with block_id as keys
        self.blocks = {}
        # Maintain the main chain as a list of block_ids
        self.main_chain = []
        self.difficulty = difficulty
        self.orphaned_blocks = {}
        self.confirmed_transactions = set()
        self.pending_chains = {}
        # Maintain a mapping of block hash to block_id for easy lookups
        self.hash_to_block_id = {}
        # Track child blocks for each block to maintain fork structure
        self.children = defaultdict(list)  # hash -> list of block_ids
    
    # Creates and returns the first block in the blockchain
    def create_genesis_block(self):
        return Block(0, "0", [], "Genesis")
        
    # Adds a new block to the blockchain if valid
    def add_block(self, block):
        # Always add the block to our collection
        self.blocks[block.block_id] = block
        self.hash_to_block_id[block.hash] = block.block_id
        
        # Add to the children mapping
        if block.previous_hash in self.hash_to_block_id:
            parent_block_id = self.hash_to_block_id[block.previous_hash]
            self.children[parent_block_id].append(block.block_id)
        
        # Check if the block connects to the latest block in the main chain
        latest_block_id = self.main_chain[-1]
        latest_block = self.blocks[latest_block_id]
        
        if block.previous_hash == latest_block.hash:
            # Direct extension of the main chain
            self.main_chain.append(block.block_id)
            for tx in block.transactions:
                self.confirmed_transactions.add(tx.txid)
            return True
            
        # Check if it's a fork from an earlier block in the main chain
        for i, block_id in enumerate(self.main_chain):
            current_block = self.blocks[block_id]
            if block.previous_hash == current_block.hash:
                # This is a fork from an earlier block
                # Create a new fork chain
                fork_chain = self.main_chain[:i+1] + [block.block_id]
                
                # Store as a pending chain
                self.pending_chains[block.hash] = fork_chain
                
                # Check if this fork is longer than the main chain
                if len(fork_chain) > len(self.main_chain):
                    self._resolve_fork(fork_chain)
                    return True
                
                return False
        
        # If we get here, it's an orphaned block
        self.orphaned_blocks.setdefault(block.previous_hash, []).append(block.block_id)
        return False

    # Resolve blockchain fork by replacing the main chain with the longest valid alternative chain
    def _resolve_fork(self, new_chain):
        if self._validate_chain(new_chain):
            # Save the old main chain for reference
            old_chain = self.main_chain
            
            # Find the split point
            split_index = 0
            for i in range(min(len(old_chain), len(new_chain))):
                if old_chain[i] != new_chain[i]:
                    split_index = i
                    break
                    
            # Remove confirmed transactions from replaced blocks
            for i in range(split_index, len(old_chain)):
                block_id = old_chain[i]
                block = self.blocks[block_id]
                for tx in block.transactions:
                    if tx.txid in self.confirmed_transactions:
                        self.confirmed_transactions.remove(tx.txid)
            
            # Add confirmed transactions from new blocks
            for i in range(split_index, len(new_chain)):
                block_id = new_chain[i]
                block = self.blocks[block_id]
                for tx in block.transactions:
                    self.confirmed_transactions.add(tx.txid)
            
            # Update the main chain
            self.main_chain = new_chain
            
            # Try to attach any orphaned blocks
            latest_block = self.blocks[new_chain[-1]]
            self._try_add_orphaned_blocks(latest_block.hash)
    
    # Validate the integrity of a potential blockchain
    def _validate_chain(self, chain):
        for i in range(1, len(chain)):
            current_block = self.blocks[chain[i]]
            previous_block = self.blocks[chain[i-1]]
            # Check if each block correctly points to the previous block
            if current_block.previous_hash != previous_block.hash:
                return False
        return True

    # Tries to attach orphaned blocks to the chain
    def _try_add_orphaned_blocks(self, parent_hash):
        if parent_hash in self.orphaned_blocks:
            orphan_block_ids = self.orphaned_blocks[parent_hash]
            del self.orphaned_blocks[parent_hash]
            
            for block_id in orphan_block_ids:
                orphan_block = self.blocks[block_id]
                self.add_block(orphan_block)  # Attempt to add the orphan

    # Printing blockchain details for debugging purposes
    def print_chain_details(self):
        print("\nBlockchain Details:")
        print("Main Chain:")
        for block_id in self.main_chain:
            block = self.blocks[block_id]
            print(f"Block {block.index}: {block}")
            for tx in block.transactions:
                print(f"  {tx}")
        
        print("\nAll Blocks:")
        for block_id, block in self.blocks.items():
            if block_id not in self.main_chain:
                print(f"Fork Block {block.index}: {block}")
                for tx in block.transactions:
                    print(f"  {tx}")
        
        print("\nOrphaned Blocks:")
        for prev_hash, block_ids in self.orphaned_blocks.items():
            print(f"Orphaned under {prev_hash[:6]}:")
            for block_id in block_ids:
                block = self.blocks[block_id]
                print(f"  {block}")
    
    # Returns the latest block in the blockchain
    def get_latest_block(self):
        latest_block_id = self.main_chain[-1]
        return self.blocks[latest_block_id]
    
    # Returns the total number of blocks in the main chain
    def get_chain_length(self):
        return len(self.main_chain)
        
    # Get all blocks in a specific fork
    def get_fork(self, block_id):
        if block_id not in self.blocks:
            return []
            
        # Start with the given block and work backwards
        fork = []
        current_id = block_id
        
        while current_id is not None:
            current_block = self.blocks[current_id]
            fork.insert(0, current_id)
            
            # Find the parent
            prev_hash = current_block.previous_hash
            if prev_hash in self.hash_to_block_id:
                current_id = self.hash_to_block_id[prev_hash]
            else:
                break
                
        return fork
        
    # Get all forks in the blockchain
    def get_all_forks(self):
        forks = []
        visited = set()
        
        # Find leaf blocks (blocks with no children)
        leaf_blocks = []
        for block_id in self.blocks:
            if block_id not in self.children or not self.children[block_id]:
                leaf_blocks.append(block_id)
                
        # For each leaf, get its fork
        for leaf_id in leaf_blocks:
            fork = self.get_fork(leaf_id)
            
            # Check if this fork is different from others we've found
            if frozenset(fork) not in visited:
                forks.append(fork)
                visited.add(frozenset(fork))
                
        return forks
    

class Peer:
    def __init__(self, peer_id, blockchain, speed, cpu_type, Ttx, global_malicious_blockchain):
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
        self.block_history = set()  # Set of block_ids that this peer has seen
        self.block_history_new = {}
        self.blockchain_tree = []
        self.raw_hash_power = 0.1 if cpu_type == "low CPU" else 1.0  
        self.malicious = True if cpu_type == "high CPU" else False
        self.received_hashes = {}
        self.received_hashes = defaultdict(list)
        self.pending_get_requests = {}
        self.timeout_duration = 5  # Default timeout (Tt), can be set via command line
        self.request_timer = {}   
        self.duplicate_hash = {}
        # Instead of creating a new malicious blockchain, reference the global one
        self.global_malicious_blockchain = global_malicious_blockchain
        self.is_ring_master = False  # New attribute for ring master status
        self.other_chain = {}
        
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
    
    def receive_block(self, block, sender_peer):
        # If we've already seen this block, stop propagation
        if block.block_id in self.block_history:
            return False
            
        # Add to block history first to prevent cycles
        self.block_history.add(block.block_id)
        
        if not block.validate_size():
            return False
                
        time_ = time.time()
        self.blockchain_tree.append([time_, block])

        # Simulate network delay based on block size
        latency = self.calculate_latency(sender_peer, block.get_size() * 8)
        
        # Only propagate to peers who haven't seen the block
        for peer in self.peers_connected:
            if peer.malicious:
                if peer != sender_peer and block.block_id not in peer.block_history:
                    peer.receive_block(block, self)

        # Use the global malicious blockchain
        success = self.global_malicious_blockchain.add_block(block)
        return success

    # Checks if a transaction is valid by ensuring the sender has enough balance.
    def validate_transaction(self, tx):
        sender_balance = self.calculate_balance(tx.sender)
        return sender_balance >= tx.amount

    # Computes the balance of a given peer by iterating through the blockchain and summing mining rewards and transaction history.
    def calculate_balance(self, peer_id):
        balance = 0
        
        # First, get the main chain block_ids
        main_chain_block_ids = self.blockchain.main_chain

        # Add mining rewards
        for block_id in main_chain_block_ids:
            block = self.blockchain.blocks[block_id]
            if block.miner_id == peer_id:
                balance += 50  # Mining reward

        # Process transaction history
        for block_id in main_chain_block_ids:
            block = self.blockchain.blocks[block_id]
            for tx in block.transactions:
                if tx.sender == peer_id:
                    balance -= tx.amount
                if tx.receiver == peer_id:
                    balance += tx.amount
        
        return balance
    
    # def detect_selfish_mining_statistically(blockchain, peers):
    #     """
    #     Implement statistical detection of selfish mining and return a list of suspicious miners
    #     """
    #     # Track intervals between successive blocks from the same miner
    #     miner_intervals = defaultdict(list)
    #     suspicious_miners = set()
        
    #     # Analyze blockchain for suspicious patterns
    #     for i in range(1, len(blockchain.main_chain) - 1):
    #         current_block_id = blockchain.main_chain[i]
    #         next_block_id = blockchain.main_chain[i + 1]
            
    #         current_block = blockchain.blocks[current_block_id]
    #         next_block = blockchain.blocks[next_block_id]
            
    #         # If same miner found consecutive blocks
    #         if current_block.miner_id == next_block.miner_id:
    #             interval = next_block.timestamp - current_block.timestamp
    #             miner_intervals[current_block.miner_id].append(interval)
        
    #     # Analyze intervals for suspicious patterns
    #     for miner_id, intervals in miner_intervals.items():
    #         if len(intervals) < 3:
    #             continue
                
    #         # Calculate statistics
    #         avg_interval = sum(intervals) / len(intervals)
    #         std_dev = (sum((i - avg_interval) ** 2 for i in intervals) / len(intervals)) ** 0.5
            
    #         # Suspiciously low variance might indicate pre-mined blocks
    #         if std_dev < 0.5 * avg_interval and len(intervals) > 5:
    #             print(f"Suspicious mining pattern detected for miner {miner_id}")
    #             suspicious_miners.add(miner_id)
    #             # Penalize this miner
    #             suspect_peer = next((p for p in peers if p.peer_id == miner_id), None)
    #             if suspect_peer:
    #                 suspect_peer.hash_power *= 0.7  # Significant penalty
        
    #     # Update suspicious miners list for all peers
    #     for peer in peers:
    #         if not hasattr(peer, 'suspicious_miners'):
    #             peer.suspicious_miners = set()
    #         peer.suspicious_miners.update(suspicious_miners)
        
    #     return suspicious_miners

    
    # Method to check balances in a specific fork
    def calculate_balance_in_fork(self, peer_id, fork_block_id):
        balance = 0
        
        # Get the fork chain
        fork_chain = self.blockchain.get_fork(fork_block_id)
        
        # Add mining rewards and process transactions in this fork
        for block_id in fork_chain:
            block = self.blockchain.blocks[block_id]
            
            # Mining rewards
            if block.miner_id == peer_id:
                balance += 50
                
            # Transactions
            for tx in block.transactions:
                if tx.sender == peer_id:
                    balance -= tx.amount
                if tx.receiver == peer_id:
                    balance += tx.amount
        
        return balance
    
    # Method to mine a block and add it to a specific fork
    def mine_block_on_fork(self, transactions, fork_block_id=None):
        # If no fork specified, mine on the main chain
        if fork_block_id is None:
            latest_block = self.blockchain.get_latest_block()
            previous_hash = latest_block.hash
            index = len(self.blockchain.main_chain)
        else:
            # Mine on the specified fork
            if fork_block_id not in self.blockchain.blocks:
                return None  # Fork doesn't exist
                
            fork_block = self.blockchain.blocks[fork_block_id]
            previous_hash = fork_block.hash
            
            # Calculate the index based on the fork's length
            fork_chain = self.blockchain.get_fork(fork_block_id)
            index = len(fork_chain)
        
        # Create the new block
        new_block = Block(index, previous_hash, transactions, self.peer_id)
        
        # Add the block to the blockchain
        self.blockchain.add_block(new_block)
        
        return new_block
    
    # Method to get all available forks
    def get_available_forks(self):
        return self.blockchain.get_all_forks()
    

def create_peers(n, z0, z1, Ttx, blockchain, global_malicious_blockchain):
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
            Ttx=Ttx,
            global_malicious_blockchain=global_malicious_blockchain
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
    
    # Implement the malicious ring master concept
    setup_ring_master(peers)
        
    return peers

# NEW FUNCTION: Set up the ring master for malicious peers
def setup_ring_master(peers):
    # Get all malicious peers
    malicious_peers = [p for p in peers if p.malicious]
    
    if not malicious_peers:
        return  # No malicious peers
    
    # Select a random malicious peer as the ring master
    ring_master = random.choice(malicious_peers)
    ring_master.is_ring_master = True
    
    # Calculate the total hash power of all malicious peers
    total_malicious_hash_power = sum(p.raw_hash_power for p in malicious_peers)
    
    # Assign all malicious hash power to the ring master
    ring_master.raw_hash_power = total_malicious_hash_power
    
    # Set other malicious peers' hash power to nearly zero (but not zero to avoid division by zero)
    for p in malicious_peers:
        if p != ring_master:
            p.raw_hash_power = 0.00001  # Very small value
    
    print(f"Peer {ring_master.peer_id} has been selected as the ring master with hash power {total_malicious_hash_power}")

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

# Check if the network is connected (all peers can reach each other)
def is_connected(peers):
    if not peers:
        return True
        
    # Use BFS to check connectivity
    visited = set()
    queue = deque([peers[0]])  # Start with the first peer
    
    while queue:
        current = queue.popleft()
        visited.add(current)
        
        for neighbor in current.peers_connected:
            if neighbor not in visited:
                queue.append(neighbor)
    
    # If all peers are visited, the network is connected
    return len(visited) == len(peers)

# Fix the event_mine_block function to ensure proper connections between blocks
def event_mine_block(peers, peer, event_queue, current_time):
    """Mine a new block and schedule its propagation."""
    # Skip mining if the peer is malicious but not the ring master
    if peer.malicious and not peer.is_ring_master:
        return
    
    current_length = peer.blockchain.get_chain_length()
    if current_length == peer.blockchain.get_chain_length():
        coinbase_tx = Transaction(None, peer.peer_id, 50, is_coinbase=True)
        valid_transactions = [coinbase_tx] + peer.mempool[:5]  # Limit block size
        
        if len(valid_transactions) > 2:
            if not peer.malicious:
                # Honest mining behavior
                latest_block = peer.blockchain.get_latest_block()
                new_block = Block(current_length, latest_block.hash, valid_transactions, peer.peer_id)
                
                # Add the hash mapping before adding the block
                peer.blockchain.hash_to_block_id[new_block.hash] = new_block.block_id
                
                success = peer.blockchain.add_block(new_block)
                
                if success:
                    peer.mempool = []  # Clear mempool after mining
                    print(f"Peer {peer.peer_id} mined Block {new_block.index}, hash: {new_block.hash[:6]}, previous: {new_block.previous_hash[:6]}")

                    peer.received_hashes[new_block.hash].append(peer.peer_id)
                    peer.block_history_new[new_block.hash] = new_block

                    # Propagate block hash to all connected peers
                    for p in peer.peers_connected:
                        latency = peer.calculate_latency(p, len(str(new_block.hash).encode()) * 8)
                        heapq.heappush(event_queue, Event(current_time + latency, "propagate_hash", p, (new_block.hash, peer)))

                    #check_selfish_mining_conditions(peer, peers, new_block, event_queue, current_time)
                    # Add this to the event_mine_block function after a successful block addition
                    if success and not peer.malicious:
                        # Schedule honest view updates for all honest peers
                        for p in peers:
                            if not p.malicious:
                                heapq.heappush(event_queue, Event(current_time + 0.1, "update_honest_view", p, None))

            
            elif peer.is_ring_master:  # Only the ring master mines blocks
                # Get the latest block in the malicious blockchain
                if len(peer.global_malicious_blockchain.main_chain) == 0:
                    peer.global_malicious_blockchain.main_chain.append(peer.blockchain.get_latest_block().block_id)
                    peer.global_malicious_blockchain.blocks[peer.blockchain.get_latest_block().block_id] = peer.blockchain.get_latest_block()
                    latest_block = peer.blockchain.get_latest_block()

                else:
                    latest_block = peer.global_malicious_blockchain.get_latest_block()
                
                k = latest_block.index + 1
                #latest_block = peer.global_malicious_blockchain.get_latest_block()
                    
                # Mine a block for the private chain
                new_block = Block(
                    k, 
                    latest_block.hash, 
                    valid_transactions, 
                    peer.peer_id  # Use the ring master's ID as the miner
                )
                
                # Add the hash mapping before adding the block
                peer.global_malicious_blockchain.hash_to_block_id[new_block.hash] = new_block.block_id
                
                peer.mempool = []
                print(f"Ring master {peer.peer_id} mined Block {new_block.index} for private chain, hash: {new_block.hash[:6]}, previous: {new_block.previous_hash[:6]}")
                
                success = peer.global_malicious_blockchain.add_block(new_block)
                
                if success:
                    peer.received_hashes[new_block.hash].append(peer.peer_id)
                    peer.block_history_new[new_block.hash] = new_block
                    
                    # Propagate only to malicious peers (synchronize private chain)
                    for p in peer.peers_connected:
                        if p.malicious:
                            # Simulate fast overlay network with lower latency
                            reduced_latency = peer.calculate_latency(p, new_block.get_size() * 8) * 0.2  # 80% reduction in latency
                            if new_block.block_id not in p.block_history:
                                #heapq.heappush(event_queue, Event(current_time, "add_to_private_chain", p, (new_block)))
                                print(f"Synchronized block {new_block.index} to malicious peer {p.peer_id} via overlay network")
                    
                    # Check if the selfish mining attack conditions are met
                    check_selfish_mining_conditions(peer, peers, new_block, event_queue, current_time)

def fix_blockchain_add_block(blockchain, block):
    """Fix for the add_block method in the Blockchain class"""
    # Always add the block to our collection

    blockchain.blocks[block.block_id] = block
    
    # Make sure the hash is in the mapping
    blockchain.hash_to_block_id[block.hash] = block.block_id
    
    # Check if the block connects to the latest block in the main chain
    latest_block_id = blockchain.main_chain[-1]
    latest_block = blockchain.blocks[latest_block_id]
    
    if block.previous_hash == latest_block.hash:
        # Direct extension of the main chain
        blockchain.main_chain.append(block.block_id)
        # Add parent-child relationship
        blockchain.children[latest_block_id].append(block.block_id)
        return True
        
    # Check if it's a fork from an earlier block in the main chain
    for i, block_id in enumerate(blockchain.main_chain):
        current_block = blockchain.blocks[block_id]
        if block.previous_hash == current_block.hash:
            # This is a fork from an earlier block
            # Create a new fork chain
            fork_chain = blockchain.main_chain[:i+1] + [block.block_id]
            
            # Add parent-child relationship
            blockchain.children[block_id].append(block.block_id)
            
            # Check if this fork is longer than the main chain
            if len(fork_chain) > len(blockchain.main_chain):
                # This fork becomes the new main chain
                blockchain.main_chain = fork_chain
                return True
            
            return False
    
    # If we get here, it's an orphaned block
    if block.previous_hash in blockchain.hash_to_block_id:
        parent_id = blockchain.hash_to_block_id[block.previous_hash]
        blockchain.children[parent_id].append(block.block_id)
        
    blockchain.orphaned_blocks.setdefault(block.previous_hash, []).append(block.block_id)
    return False

def check_selfish_mining_conditions(peer, peers, new_block, event_queue, current_time):
    """Check if conditions for selfish mining attack are met and initiate propagation if needed."""
    
    # Find the point of convergence
    honest_chain = peer.blockchain.main_chain
    malicious_chain = peer.global_malicious_blockchain.main_chain
    
    divergence_point = 0
    for i in range(min(len(honest_chain), len(malicious_chain))):
        if honest_chain[i] != malicious_chain[i]:
            divergence_point = i
            break
    
    # Calculate lengths from the point of convergence
    honest_chain_length = len(honest_chain) - divergence_point
    malicious_chain_length = len(malicious_chain) - divergence_point
    
    # Condition 1: Length of honest chain equals malicious private chain length
    # Condition 2: Length of honest chain equals malicious private chain length - 1
    if honest_chain_length == malicious_chain_length or honest_chain_length == malicious_chain_length - 1:
        print(f"Selfish mining condition met: Honest chain={honest_chain_length}, Malicious chain={malicious_chain_length} (from convergence point)")
        
        # Command all malicious nodes to broadcast the private chain
        for p in peers:
            if p.malicious:
                heapq.heappush(event_queue, Event(current_time, "broadcast_private_chain", p, (new_block, peers)))

def event_broadcast_private_chain(peer, data, event_queue, current_time):
    """Broadcast the entire private chain to the original network."""
    block, peers = data
    if not peer.malicious:
        return # Only malicious peers should broadcast the private chain

    # Identify the point where the chains diverged
    honest_chain_ids = peer.blockchain.main_chain
    malicious_chain_ids = peer.global_malicious_blockchain.main_chain
    
    if len(malicious_chain_ids) > 0:

        # Broadcast blocks in order to ensure proper chain construction
        for block_to_broadcast in peer.global_malicious_blockchain.main_chain:
            block_exists = False
            for bid in honest_chain_ids:
                if peer.blockchain.blocks[bid].block_id == block_to_broadcast:
                    block_exists = True
                    break
            if not block_exists:
                # Make the block available for retrieval first
                block = peer.global_malicious_blockchain.blocks[block_to_broadcast]
                peer.block_history_new[block.hash] = block_to_broadcast
                peer.blockchain.add_block(block)
                
                # Broadcast to honest peers only
                for p in peers:
                    if not p.malicious:
                        latency = peer.calculate_latency(p, len(str(block.hash).encode()) * 8)
                        # Propagate the hash
                        heapq.heappush(event_queue, Event(current_time, "propagate_hash", p, (block.hash, peer)))
                        
                        # CRITICAL: Force immediate block addition to honest peer's blockchain
                        p.block_history_new[block.hash] = block
                        p.blockchain.add_block(block)
                        
                        # Schedule immediate update of honest peer's view
                        heapq.heappush(event_queue, Event(current_time, "update_honest_view", p, None))

def event_update_honest_view(peer, data, event_queue, current_time, peers):
    """Force honest miners to update their view of the blockchain and mine on the longest chain."""
    if peer.malicious:
        return  # Only for honest miners
    
    # Run statistical detection of selfish mining before updating view
    suspicious_miners = detect_selfish_mining_statistically(peer.blockchain, peers)
    
    # Find all forks in the blockchain
    all_forks = peer.blockchain.get_all_forks()
    
    # Find the longest fork that isn't from suspicious miners
    longest_fork = peer.blockchain.main_chain
    longest_length = len(longest_fork)

    # Modify the fork checking in event_update_honest_view
    for fork in all_forks:
        # Count suspicious blocks in this fork
        suspicious_blocks = 0
        for block_id in fork:
            block = peer.blockchain.blocks[block_id]
            if block.miner_id in suspicious_miners:
                suspicious_blocks += 1
        
        # Skip this fork if it has too many suspicious blocks
        if suspicious_blocks > len(fork) * 0.4:  # More than 40% suspicious
            print(f"Honest peer {peer.peer_id} rejected fork with {suspicious_blocks}/{len(fork)} suspicious blocks")
            continue
            
        # Only choose this fork if it's longer than our current choice
        if len(fork) > longest_length:
            longest_fork = fork
            longest_length = len(fork)

    
    # for fork in all_forks:
    #     # Skip this fork if the last block was mined by a suspicious miner
    #     if len(fork) > 0:
    #         last_block_id = fork[-1]
    #         last_block = peer.blockchain.blocks[last_block_id]
    #         if last_block.miner_id in suspicious_miners:
    #             print(f"Honest peer {peer.peer_id} rejected suspicious fork from miner {last_block.miner_id}")
    #             continue
        
    #     # Only choose this fork if it's longer than our current choice
    #     if len(fork) > longest_length:
    #         longest_fork = fork
    #         longest_length = len(fork)
    
    # If a longer fork is found, update the main chain
    if longest_fork != peer.blockchain.main_chain:
        print(f"Honest peer {peer.peer_id} updating to new longest chain after selfish mining check")
        peer.blockchain.main_chain = longest_fork
        
        # Clear mempool of transactions already in the new chain
        confirmed_txs = set()
        for block_id in longest_fork:
            block = peer.blockchain.blocks[block_id]
            for tx in block.transactions:
                confirmed_txs.add(tx.txid)
        peer.mempool = [tx for tx in peer.mempool if tx.txid not in confirmed_txs]
        
        # Schedule a new mining attempt on the new chain
        heapq.heappush(event_queue, Event(current_time + 0.1, "mine_block", peer))



def detect_selfish_mining_statistically(blockchain, peers):
    """
    Implement statistical detection of selfish mining
    Returns a set of suspicious miner IDs
    """
    # Track intervals between successive blocks from the same miner
    miner_intervals = defaultdict(list)
    suspicious_miners = set()
    
    # Analyze blockchain for suspicious patterns
    for i in range(1, len(blockchain.main_chain) - 1):
        current_block_id = blockchain.main_chain[i]
        next_block_id = blockchain.main_chain[i + 1]
        
        current_block = blockchain.blocks[current_block_id]
        next_block = blockchain.blocks[next_block_id]
        
        # If same miner found consecutive blocks
        if current_block.miner_id == next_block.miner_id:
            interval = next_block.timestamp - current_block.timestamp
            miner_intervals[current_block.miner_id].append(interval)
    
    # Analyze intervals for suspicious patterns
    for miner_id, intervals in miner_intervals.items():
        if len(intervals) < 3:
            continue
            
        # Calculate statistics
        avg_interval = sum(intervals) / len(intervals)
        std_dev = (sum((i - avg_interval) ** 2 for i in intervals) / len(intervals)) ** 0.5
        
        # Suspiciously low variance might indicate pre-mined blocks
        if std_dev < 0.5 * avg_interval and len(intervals) > 5:
            print(f"Suspicious mining pattern detected for miner {miner_id}")
            suspicious_miners.add(miner_id)
            
            # Penalize this miner (optional)
            suspect_peer = next((p for p in peers if p.peer_id == miner_id), None)
            if suspect_peer:
                suspect_peer.hash_power *= 0.7  # Significant penalty

        
                for peer in peers:
                    if not hasattr(peer, 'suspicious_miners'):
                        peer.suspicious_miners = set()
                    peer.suspicious_miners.update(suspicious_miners)
    
    return suspicious_miners




# Add a new event handler for updating honest miners' view
def update_honest_view(peer):
    """Force honest miners to update their view of the blockchain and mine on the longest chain."""
    if peer.malicious:
        return  # Only for honest miners
    
    # Re-validate the blockchain to find the longest chain
    # This will ensure the miner recognizes any new blocks that form a longer chain
    
    # Find all forks in the blockchain
    all_forks = peer.blockchain.get_all_forks()
    
    # Find the longest fork
    longest_fork = peer.blockchain.main_chain
    for fork in all_forks:
        if len(fork) > len(longest_fork):
            longest_fork = fork
    
    # If a longer fork is found, update the main chain
    if longest_fork != peer.blockchain.main_chain:
        print(f"Honest peer {peer.peer_id} updating to new longest chain after selfish mining attack")
        peer.blockchain.main_chain = longest_fork
        
        # Schedule a mining attempt on the new chain
        #heapq.heappush(event_queue, Event(current_time + 0.1, "mine_block", peer))

def event_create_transaction(peer, peers, event_queue, current_time):
    """Generate a transaction and schedule it for propagation."""
    receiver_id = random.choice([p.peer_id for p in peers if p.peer_id != peer.peer_id])
    amount = random.randint(1, peer.balance)
    
    if amount > 0:
        tx = Transaction(peer.peer_id, receiver_id, amount)
        
        # Ensure the transaction is recorded before propagating
        with peer.tx_history_lock:
            peer.tx_history[tx.txid] = set()
            peer.tx_history[tx.txid].add(peer.peer_id)

        # Schedule propagation event
        for p in peer.peers_connected:
            if tx.txid not in p.tx_history:
                latency = peer.calculate_latency(p, len(str(tx).encode()) * 8)
                heapq.heappush(event_queue, Event(current_time + latency, "propagate_transaction", p, tx))

        print(f"Peer {peer.peer_id} generated {tx} sent to peers {[p.peer_id for p in peer.peers_connected]}")


def event_propagate_transaction(peer, tx, event_queue, current_time):
    """Propagate an existing transaction to connected peers without infinite loops."""
    
    # If this peer has already seen the transaction, stop propagation
    if tx.txid in peer.tx_history:
        return  

    # Add transaction to history
    with peer.tx_history_lock:
        peer.tx_history[tx.txid] = set()
        peer.tx_history[tx.txid].add(peer.peer_id)
    
    # Store the transaction in mempool
    peer.mempool.append(tx)

    # Propagate transaction to peers who have not seen it yet
    for p in peer.peers_connected:
        if tx.txid not in p.tx_history:
            latency = peer.calculate_latency(p, len(str(tx).encode()) * 8)
            heapq.heappush(event_queue, Event(current_time + latency, "propagate_transaction", p, tx))

def event_propagate_hash(peer, data, event_queue, current_time, Tt):
    block_hash, sender_peer = data  
    sender_id = sender_peer.peer_id
    if block_hash in peer.received_hashes:
        peer.duplicate_hash[block_hash] = sender_peer
        return 
        
    peer.received_hashes[block_hash].append(sender_id)

    if block_hash in peer.block_history_new: #if peer already has the complete block, do not receive it again. propagate it to its peers
        for p in peer.peers_connected:
            latency = peer.calculate_latency(p, len(str(block_hash).encode()) * 8)
            heapq.heappush(event_queue, Event(current_time + latency, "propagate_hash", p, (block_hash, peer)))
            return

    peer.request_timer[block_hash] = current_time + Tt

    heapq.heappush(event_queue, Event(current_time, "handle_get", peer, (block_hash, sender_peer)))

def event_handle_get(peer, data, event_queue, current_time):
    block_hash, sender_peer = data
    sender_id = sender_peer.peer_id
    
    if current_time < peer.request_timer[block_hash]:
        # Within timeout, get the block
        peer.block_history_new[block_hash] = sender_peer.block_history_new[block_hash]
        blk_id = peer.blockchain.hash_to_block_id[block_hash]
        block = peer.blockchain.blocks[blk_id]
        peer.blockchain.add_block(block)
    else:
        # Timeout exceeded, try another peer
        if peer.duplicate_hash:
            hash, new_sender = peer.duplicate_hash.popitem()
            peer.request_timer[block_hash] = current_time + 30
            heapq.heappush(event_queue, Event(current_time + 2, "handle_get", peer, (block_hash, new_sender)))

def event_propagate_block(peer, block, event_queue, current_time):
    """Propagate a block to connected peers."""
    if block.block_id in peer.block_history:
        return False
        
    # Add to block history first to prevent cycles
    peer.block_history.add(block.block_id)
    
    if not block.validate_size():
        return False
    
    # Validate each transaction in the block
    for tx in block.transactions:
        if not tx.is_coinbase and not peer.validate_transaction(tx):
            return False
            
    time_ = time.time()
    peer.blockchain_tree.append([time_, block])
    
    success = peer.blockchain.add_block(block)
    if not success:
        return
        
    for p in peer.peers_connected:
        latency = peer.calculate_latency(p, block.get_size() * 8)
        heapq.heappush(event_queue, Event(current_time + latency, "propagate_block", p, block))
    
    print(f"Block {block.index} propagated by Peer {peer.peer_id}")

def event_simulate_network(n, z0, z1, Ttx, x, y, filename, Tt):
    """Simulates the network with the selfish mining attack."""
    # Create blockchains
    global_blockchain = Blockchain()  # Regular blockchain for honest nodes
    
    # Instead of creating a separate malicious blockchain, we'll create it as a copy
    # after properly initializing the honest blockchain
    
    # Ensure the genesis block is properly set
    genesis_block = Block(0, "0", [], "Genesis")
    
    # Set up the genesis block in the honest blockchain
    global_blockchain.blocks[genesis_block.block_id] = genesis_block
    global_blockchain.main_chain = [genesis_block.block_id]
    global_blockchain.hash_to_block_id[genesis_block.hash] = genesis_block.block_id
    global_blockchain.children[genesis_block.block_id] = []
    
    # Now create the malicious blockchain as a deep copy of the honest one
    # This ensures they start with identical genesis blocks
    global_malicious_blockchain = copy.deepcopy(global_blockchain)
    
    # Verify both blockchains have the same genesis block
    honest_genesis_id = global_blockchain.main_chain[0]
    malicious_genesis_id = global_malicious_blockchain.main_chain[0]
    
    honest_genesis = global_blockchain.blocks[honest_genesis_id]
    malicious_genesis = global_malicious_blockchain.blocks[malicious_genesis_id]
    
    print("Genesis Block Verification:")
    print(f"Honest Genesis: {honest_genesis.hash[:6]}")
    print(f"Malicious Genesis: {malicious_genesis.hash[:6]}")
    print(f"Blocks are the same: {honest_genesis.hash == malicious_genesis.hash}")
    
    # Monkey patch the add_block method to use our fixed version
    global_malicious_blockchain.add_block = lambda block: fix_blockchain_add_block(global_malicious_blockchain, block)
    
    peers = create_peers(n, z0, z1, Ttx, global_blockchain, global_malicious_blockchain)
    
    # Identify malicious peers and ring master
    malicious_peers = [p for p in peers if p.malicious]
    ring_master = next((p for p in malicious_peers if p.is_ring_master), None)
    
    # Debug output
    print("\nPeer Information:")
    for p in peers:
        role = "Ring Master" if p.is_ring_master else ("Malicious" if p.malicious else "Honest")
        print(f"Peer {p.peer_id}: Role={role}, Type={p.cpu_type}, Speed={p.speed}")
    
    print("\nNetwork Connections:")
    for p in peers:
        print(f"Peer {p.peer_id} is connected to: {[cp.peer_id for cp in p.peers_connected]}")

    normalise_hash_powers(peers)

    event_queue = []
    current_time = 0

    # Adjust transaction and mining rates
    k = x  # Transaction rate
    m = y  # Mining rate
    
    for peer in peers:
        # Schedule transaction creation for all peers
        l = math.ceil(k * (0.1 if peer.malicious and not peer.is_ring_master else peer.hash_power))
        for i in range(l):
            heapq.heappush(event_queue, Event(current_time + random.expovariate(1/Ttx), "create_transaction", peer, peers))
        
        # Schedule mining only for non-malicious peers and the ring master
        if not peer.malicious or peer.is_ring_master:
            n = math.ceil(m * peer.hash_power)
            for o in range(n):
                heapq.heappush(event_queue, Event(current_time + random.expovariate(1/10), "mine_block", peer))
    
    # Save intermediate visualizations during simulation
    save_intervals = [100, 500]  # Save at these time points
    next_save_idx = 0
    
    while event_queue and current_time < 1000:
        event = heapq.heappop(event_queue)
        current_time = event.time
        
        # Process events
        if event.event_type == "create_transaction":
            event_create_transaction(event.peer, peers, event_queue, current_time)
        elif event.event_type == "propagate_transaction":
            event_propagate_transaction(event.peer, event.data, event_queue, current_time)
        elif event.event_type == "mine_block":
            event_mine_block(peers, event.peer, event_queue, current_time)
        elif event.event_type == "propagate_block":
            event_propagate_block(event.peer, event.data, event_queue, current_time)
        elif event.event_type == "propagate_hash":
            event_propagate_hash(event.peer, event.data, event_queue, current_time, Tt)
        elif event.event_type == "handle_get":
            event_handle_get(event.peer, event.data, event_queue, current_time)
        # elif event.event_type == "add_to_private_chain":
        #     event_add_to_private_chain(event.peer, event.data, event_queue, current_time)
        elif event.event_type == "broadcast_private_chain":
            event_broadcast_private_chain(event.peer, event.data, event_queue, current_time)
        # In the main event loop, add this case
        # elif event.event_type == "update_honest_view":
        #     event_update_honest_view(event.peer, event.data, event_queue, current_time)
        elif event.event_type == "update_honest_view":
            event_update_honest_view(event.peer, event.data, event_queue, current_time, peers)

    print("\nSimulation completed.")
    
    print("\nHonest Blockchain:")
    print_blockchain(global_blockchain)
    
    print("\nMalicious Private Blockchain:")
    print_blockchain(global_malicious_blockchain)
    
    # Print success metrics
    honest_blocks = len(global_blockchain.main_chain) - 1  # Subtract genesis block
    malicious_blocks = len(global_malicious_blockchain.main_chain) - 1  # Subtract genesis block

    # Save final visualizations with the new functions
    visualize_combined_blockchain(global_blockchain, global_malicious_blockchain, peers, filename)
    
    print(f"\nResults:")
    print(f"Honest blocks mined: {honest_blocks}")
    print(f"Malicious blocks mined: {malicious_blocks}")
    
    if honest_blocks + malicious_blocks > 0:
        malicious_percentage = (malicious_blocks / (honest_blocks + malicious_blocks)) * 100
        print(f"Malicious blocks percentage: {malicious_percentage:.2f}%")
        
        # Calculate the effectiveness of the selfish mining attack
        malicious_blocks_in_main_chain = sum(1 for block_id in global_blockchain.main_chain 
                                          if block_id in global_blockchain.blocks and
                                          global_blockchain.blocks[block_id].miner_id in 
                                          [p.peer_id for p in malicious_peers])
        
        malicious_success_rate = (malicious_blocks_in_main_chain / len(global_blockchain.main_chain)) * 100
        
        print(f"Malicious blocks in main chain: {malicious_blocks_in_main_chain}")
        print(f"Malicious success rate: {malicious_success_rate:.2f}%")
        
        if ring_master:
            print(f"Ring master (Peer {ring_master.peer_id}) coordinated the selfish mining attack")
    
    return global_blockchain, global_malicious_blockchain

def print_blockchain(blockchain):
    """Print the blocks in the main chain of a blockchain"""
    print(f"Chain length: {len(blockchain.main_chain)}")
    
    for block_id in blockchain.main_chain:
        block = blockchain.blocks[block_id]
        print(f"\nBlock {block.index}, Miner: {block.miner_id}, Hash: {block.hash[:6]}")
        for tx in block.transactions:
            print(f"  {tx}")

def visualize_combined_blockchain(honest_blockchain, malicious_blockchain, peers, filename):
    """Visualize both honest and malicious blockchains in a structured, linear format."""
    G = nx.DiGraph()
    blocks_by_hash = {}

    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # filename = f"blockchain_visualization_{timestamp}.png"
    
    # Create a set of malicious peer IDs for quick lookup
    malicious_peer_ids = {p.peer_id for p in peers if p.malicious}
    
    # Track blocks from both chains
    for blockchain, is_malicious in [(honest_blockchain, False), (malicious_blockchain, True)]:
        for block_id, block in blockchain.blocks.items():
            if block.hash not in blocks_by_hash:
                blocks_by_hash[block.hash] = {
                    'block_id': block_id,
                    'block': block,
                    'malicious': block.miner_id in malicious_peer_ids  # Check if miner is malicious
                }
    
    # Add nodes with colors based on miner's status (red for malicious miners, blue for honest miners)
    for block_hash, block_info in blocks_by_hash.items():
        block = block_info['block']
        # Color based on miner's status, not which blockchain it's from
        color = 'lightcoral' if block_info['malicious'] else 'lightblue'
        
        G.add_node(block_hash, block_id=block_info['block_id'], 
                   index=block.index, miner=block.miner_id, 
                   color=color, prev_hash=block.previous_hash)
    
    # Add edges based on previous_hash, but REVERSE the direction
    for block_hash, node_data in G.nodes(data=True):
        if node_data['prev_hash'] != "0":  # Skip genesis block
            if node_data['prev_hash'] in G:
                # Reverse the edge direction: child -> parent instead of parent -> child
                G.add_edge(block_hash, node_data['prev_hash'])
            else:
                print(f"Warning: Missing parent for block {block_hash[:6]}")
    
    # Layout for clear HORIZONTAL linear visualization (RL = right to left)
    pos = nx.nx_agraph.graphviz_layout(G, prog='dot', args='-Grankdir=RL')
    
    # Draw the graph
    plt.figure(figsize=(16, 10), dpi=300)
    nx.draw(G, pos, node_color=[G.nodes[n]['color'] for n in G.nodes], node_size=100, edge_color='gray', arrows=True)
    
    # # Add labels with block index and miner
    # labels = {node: f"B{data['index']}\nM:{data['miner']}" for node, data in G.nodes(data=True)}
    # nx.draw_networkx_labels(G, pos, labels=labels, font_size=6)
    
    plt.title("Structured Blockchain Visualization (Horizontal)")
    plt.savefig(filename)



def normalise_hash_powers(peers):
    # Normalize hash power
    total_raw_power = sum(peer.raw_hash_power for peer in peers)
    for p in peers:
        p.hash_power = p.raw_hash_power / total_raw_power


def compute_ratios(n, z0, z1, Ttx, Tt, peers):
    # Run the simulation
    global_blockchain, global_malicious_blockchain = event_simulate_network(n, z0, z1, Ttx, x, y, filename)
    
    # Find the ring master
    ring_master = next((p for p in peers if p.is_ring_master), None)
    if not ring_master:
        return None, None
    
    # Get the longest chain
    longest_chain = ring_master.global_malicious_blockchain.main_chain
    total_blocks_in_longest_chain = len(longest_chain)
    
    # Count malicious blocks in the longest chain
    malicious_peer_ids = [p.peer_id for p in peers if p.malicious]
    malicious_blocks_in_longest_chain = sum(
        1 for block_id in longest_chain
        if global_malicious_blockchain.blocks[block_id].miner_id in malicious_peer_ids
    )
    
    # Count total blocks generated by malicious nodes
    total_blocks_by_malicious_nodes = sum(
        1 for block_id, block in global_malicious_blockchain.blocks.items()
        if block.miner_id in malicious_peer_ids
    )
    
    # Calculate ratios
    ratio_1 = malicious_blocks_in_longest_chain / total_blocks_in_longest_chain if total_blocks_in_longest_chain > 0 else 0
    ratio_2 = malicious_blocks_in_longest_chain / total_blocks_by_malicious_nodes if total_blocks_by_malicious_nodes > 0 else 0
    
    return ratio_1, ratio_2

# # Event: Run Event Simulator
# def event_run_simulator():
#     n = sys.argv[0]  # Number of peers
#     z0 = 20  # 20% of peers are slow
#     z1 = 0  # 20% of peers are low CPU
#     Ttx = 30  # Mean transaction interarrival time in seconds
#     event_simulate_network(n, z0, z1, Ttx)


parser = argparse.ArgumentParser(description="Simulate a network with given parameters.")

# Define integer command-line arguments
parser.add_argument("n", type=int, help="Number of nodes")
parser.add_argument("z0", type=int, help="Initial value of z0")
parser.add_argument("z1", type=int, help="Initial value of z1")
parser.add_argument("Ttx", type=int, help="Transmission time (Ttx)")
parser.add_argument("x", type=int, help="Transaction rate")
parser.add_argument("y", type=int, help="Mining rate")
parser.add_argument("filename", type=str, help="Filename")
parser.add_argument("Tt", type=int, help="Timeout")

# Parse arguments
args = parser.parse_args()

# Access arguments
n = args.n
z0 = args.z0
z1 = args.z1
Ttx = args.Ttx
x = args.x
y = args.y
filename = args.filename
Tt = args.Tt

# Run the simulation
event_simulate_network(n, z0, z1, Ttx, x, y, filename, Tt)


