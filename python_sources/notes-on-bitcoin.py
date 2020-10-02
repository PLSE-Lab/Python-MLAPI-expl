#!/usr/bin/env python
# coding: utf-8

# # Notes on bitcoin and blockchain
# These notes are a long hard look at Bitcoin, prompted by the fact that I have homework in the shape of writing about an aspect of Bitcoin.
# 
# ## Blockchain
# This section based on [the canonical Blockchain demo](https://andersbrownworth.com/blockchain/).
# 
# * A **block** contains a number, a nonce, a previous value (which references its predacessor's hash value), and a data payload. A block is **signed** if its hash conforms to a proof-of-work rule. A simple proof-of-work rule is "the SHA-256 hash starts with four zeroes". The proof of work actually used in Bitcoin is of course more complicated, for details on that see the notebook [Proof of work systems and in bitcoin](https://www.kaggle.com/residentmario/proof-of-work-systems-and-in-bitcoin).
# * The process of fidning a block value that satisfied the proof of work target is known as **mining**.
# * The origin block presumably has a zeroed-out previous reference that is accepted as tautologically true.
# * A **blockchain** is a linear chain of blocks. In data structure terms it's essentially a linked list.
# * The blockchain is tamper-resistant: manipulating block data will change its hash, cascading new references and new hash values through every block that occurs after it. It is impossible for these changes to satisfy the zero-signage property, so this tampering would require mining every subsequent block after it as well.
# 
# 
# * The Bitcoin network is **trustless**. How is this achieved?
# * Every peer in the Bitcoin network has a complete copy of the ledger. Applications that work on top of the blockchain determine that their blockchain has not been manipulated, up to their preferred certain level of confidence, by:
#   * Comparing the latest hash for the block number they are working with with that same hash as reported by their peers. The more of their peers with the same data, the greater their confidence that the blockchain is correct.
#   * Waiting until the blocks they are interested in are deeper in the chain. Blocks further away from latest are exponentially harder to fake. Blocks more than a few links deep are essentially immutable.
# * The data payload in Blockchain is a combination of two things: transactions, and newly minted coin assignments.
# * Specifically, transactions must point to a set of recipients and/or coin allocations for the party sending the money whose sum is greater than or equal to the amount of bitcoins being transacted. Anyone wanting to verify this node can then check these past nodes to verify that. Transactions involving receipts whose sum is greater than the amount disbursed cleverly include a transaction from the user to themselves to account for the difference.
# * Proof that a transaction was added to the blockchain by the user performing the transaction is provided via asymmetric key encryption. The party performing the transaction signs a pointer from their public key to the recipient's public key using their private key. Anyone with the sender's public key can verify the identity of the sender and integrity of the transaction parties and amount.
# 
# 
# ## Quick practical FAQ
# * Q: How big is the Bitcoin blockchain?
# 
#   A: As of late 2019 it is ~250 GB. Every new block was about 1 MB, and verifying a new block on a single node takes about 30 seconds and about 1.25 GB of memory (numbers from [a 2017 article](https://hackernoon.com/if-we-lived-in-a-bitcoin-future-how-big-would-the-blockchain-have-to-be-bd07b282416f)). The new block creation rate is about 50 GB per year.
# 
# 
# * Q: Do you really have to keep the entire transaction log locally?
#   
#   A: In Bitcoin, yes; there is no way to know that a newly seen transaction is valid without scanning backwards in time through the transaction history. Obviously you won't need to visit *every* single node, but if transactions occur that involve bitcoins that have not moved in many years, you will have to reach back pretty far into that 250 GB stack to verify them.
#   
#   In Ethereum, no. Ethereum has the benifit of transaction pruning, using one of a lot of different implementations of insanely complex algorithms. These algorithms exist in Ethereum and not in Bitcoin because Ethereum, released 2015, had the benefit of six years of hindsight in Bitcoin, releaed 2009.
# 
# 
# * Q: How many transactions run on Bitcoin? How quickly do they clear?
# 
#   A: Bitcoin can process roughly 7 transactions per second. For the purposes of comparison, ethereum can process 15, Ripple can process 1,500. Visa, the credit card, processes 22,000 transactions per second.
#   
#   Transactions have historically taken anywhere between 10 minutes and 8 days to clear, with a rolling average (as of [early 2018](https://blocksplain.com/2018/02/28/transaction-speeds/)) of 1 hour.
# 
# 
# * Q: What are transaction fees?
# 
#   A: Transaction fees are fees you pay to perform a transaction. They are generally around 4 dollars, but can be as high as 28 dollars when the transaction volume gets too high.
#   
#   Transaction fees are optional disbursements which point to "the miner". Miners are free to prioritize mining blocks including transactions that paid for priority queueing. Obviously the more powerful the miner, the likelier they are to succeed in mining a block containing the prioritized transaction, the faster the transaction will propogate to other nodes on the network, and the more valuable paying the miner a fee for priority queueing will be.
#   
#   
# * Q: What is the bottleneck?
#   
#   A: Blocks are a certain well-defined size and are only mined in ten minute intervals on average. The 7 transactions per second limit is a fundamental limitation of block size; if the network had a larger block size, throughput would be higher. But it's not possible to change block size after-the-fact.
#   
#   
# * Q: What are some proposed solutions for the network throughput bottleneck?
# 
#   A: [This article](https://www.freecodecamp.org/news/future-of-bitcoin-cc6936ba0b99/) has a good overview of some ideas. A lot of it seems to come down to building second and third order systems on top of blockchain, which, for small enough dollar amounts and trusted enough users, allow users to perform lookahead transactions without fully committing to the network until later&mdash;essentially credit with bitcoin.
#   
#   These systems are known as **off-chain** solutions.
# 
# * Q: How often is mining difficulty recalibrated?
# 
#   A: Every 2016 blocks. The calculation is made based on a closed-form formula set by design fiat.
# 
# 
# ## Mining bitcoins
# * The pool of unconfirmed transactions is known as the **mempool**.
# * Unconfirmed transactions are passed around the network. Blocks have a certain well-defined size, so a miner can only attempt mining a block with so many transactions, so, as stated in the previous section, they prioritize the ones with the highest transaction fees.
# * An interesting detail of the implementation is that the fees earned for a transaction are contingent on any past transactions as well. So if you are suffering from a stuck transaction that you badly want to clear, you can create and propogate a child transaction contingent on the stuck transaction whose only value is to give the miner a bigger reward for including it in their block proof of work. This incentivizes the miner to pack both transactions into its candidate block. This technique discussed [here](https://bitzuma.com/posts/how-to-clear-a-stuck-bitcoin-transaction/).
# 
# 
# * In addition to transaction fees, bitcoin miners are paid for their work by being allowed to mint new coins for themselves as part of the block. The amount of coins disbursed is halved every 210,000 blocks, which is approximately 4 years. The algorithm will eventually reduce the coin allocation portion all the way down to zero.
# * The monetary reward for mining a block is large, but the chance of successfully doing so is small. Miners mostly operate in **pools** to reduce variance. When a block is successfully mined by a pool, the pool's participants are rewarded in proportion to the amount of work they contributed to the search.
# 
# 
# * How are successfully mined blocks announced to the network? The miner fans the block out to all of its peers, which fan the block out to all of *their* peers. Peers verify that the block is valid, then add it to their blockchain.
# * A race condition occurs when two miners mine the same block sequence number in close time proximity to one another. The two miners will propogate the block through the network, but ultimately one miner will win out over the other. The winner will be determined by some combination of which miner is better connected to the overall network, and which miner gets a head start on the other one. Peers that are not miners will recieve and verify both forks of the blockchain, but miners will start their next round of mining on the blockchain which arrived first. The next block mined will be assigned to whichever block arrived first at that particular miner, and assuming another race condition doesn't occur (in which case the branching will continue), eventually one block will become part of the longest chain and the other will be abandoned by the nodes still carrying it.
# * [This set of bitcoin network visualizations](https://dsn.tm.kit.edu/bitcoin/videos.html) has a great side-by-side showing this in action. Amazingly, the winning bitcoin was mined 0.5 seconds *after* the losing one, but won out due to better network interconnectiveness on the part of its miner.
# 
# 
# ## Wallets
# * **Wallets** are relatively thin wrappers around your private and public key.
# * Wallets let you propogate your public key to the network, and they store references to the blocks with transactions you are involved in, as these are necessary to form your own transactions.
# 
# 
# ## 51% attack
# * In theory any mining pool that accrues 51% of the total mining power can defeat the security mechanisms in place.
# * This briefly happened in 2014 with one miner, but users voluntarily left the pool to decrease its percentage.
