#!/usr/bin/env python
# coding: utf-8

# # Proof of work systems
# 
# * **Proof of work** is a consensus mechanism that guards against denial of service and other spam form attacks by requiring clients to perform some work.
# * The key property of a proof of work system is asymmetry: the challenge must be easier to generate than it is to solve. By requiring that client systems expend these CPU cycles, the system incorporating proof of work into its flow aims to make volumetric attacks less computationally feasible.
# 
# ## Basics of hashcash
# This section of the notes taken from [the Wikipedia article on hashcash](https://en.wikipedia.org/wiki/Hashcash).
# 
# * Hashcash is a proof-of-concept system proposed in 1997 that demonstrates an implementation of this idea.
# * Hashcash is designed to combat email spam. Hashcash adds the following header to the email:
# 
#   ```
#   X-Hashcash: 1:20:1303030600:adam@cypherspace.org::McMybZIhxKXu57jd:ckvi
#   ```
#   
# > The header contains:
# > 
# > * ver: Hashcash format version, 1 (which supersedes version 0).
# > * bits: Number of "partial pre-image" (zero) bits in the hashed code.
# > * date: The time that the message was sent, in the format YYMMDD[hhmm[ss]].
# > * resource: Resource data string being transmitted, e.g., an IP address or email address.
# > * ext: Extension (optional; ignored in version 1).
# > * rand: String of random characters, encoded in base-64 format.
# > * counter: Binary counter, encoded in base-64 format.
#   
# * The last field is a base 64 encoded binary counter (e.g. `base64('0100101')`). The second-to-last field is a binary-encoded random string. The email in this case is the resource ID&mdash;since this is an email-based system, it's the receipient's email address.
# * The header body (colons included) must hash to a SHA-1 hash with `bits` many leading zero digits (20 in this case; or equivalently 5 hex digits). A valid header is solvable only using brute force (e.g. no faster method is known for generating it) by generating a `rand` string and then incrementing the binary counter value until the zero-ness property is satisfied. This took approximately a second of wall clock time at the time of the algorithm's original implementation.
# * The resulting value is included in the email's header payload. To verify it, the recipient checks the validity of the date (which is checked within a two-day window to account for network delay and/or clock skew), checks that the resource ID payload is correct (e.g. the email address is correct) and then checks that the hashed zero-ness property is satisfied. This operation is cheap.
# * Header payload re-use is prevented by checking the random portion of the header into a database, and rejecting messages with the same random portion payload. Presumably these database entries are evicted after a certain amount of point, to prevent legitimate messages from failing to go through simply because they generated a previously-used random hash. Or perhaps some other mechanism is used to fix this issue, I don't know, point is, it's solvable.
# * A second of compute time to send an email was expected to be reasonable for a regular user, but unreasonable for a spammer.
# * The proof of work required is exponential with the number of leading zero digits: every additional zero added doubles the cost.
# 
# ## High-level overview of the use of hashcash in bitcoin
# * The hashcash algorithm is used in Bitcoin as a proof-of-work mechanism for enabling competitive bitcoin mining.
# * Recall that in bitcoin a large variety of kinds of hardware, up to and including custom ASICs, are used to mine bitcoins, which have monetary value.
# * Miners do useful work for the bitcoin ecosystem by collecting unconfirmed transactions on the network, forming a block. However, these blocks are only accepted when the miner discovers (via trial and error) a nonce which, when combined with the block structure, hashes to a member of a certain target set.
# * Instead of SHA-1, Bitcoin uses SHA-256 (in fact it uses double SHA-256, e.g. it uses the hash of the hash). SHA-1 is weaker than SHA-256, but is still considered cryptographically secure. This decision is considered to be an arbitrary but conservative one on the part of Bitcoin's creator.
# * The target used is periodically adjusted to keep bitcoin mining difficulty constant in the face of increasing computational resources. It was original 30 leading zero bits, but as of January 2019 has been increased to 74 leading zero bits (and may be even higher now).
# * Each successful block mined counts as a vote towards the allotment of a newly minted piece of a bitcoin.
# 
# ## Some implementation details of bitcoin hashcash
# The following section draws from the [Hashcash](https://en.bitcoin.it/wiki/Hashcash) article on the Bitcoin wiki.
# 
# * Let us define the hashcash algorithm more formally.
# * We state that the proof of work presented by hashcash is formulaicly:
# 
#   $$H(s,c)//2^{(n-k)}=0$$
# 
# * Where:
#   * $s$&mdash;service string
#     * In the original hashcash implementation this is the email address of the recipient.
#     * In Bitcoin this is a block struct.
#     * The service string is the *purpose* component to the proof-of-work. In the reference implementation this is "send an email here". In Bitcoin this is "add this verifiable proof of transactions tendered to the blockchain ledger".
#     * In Bitcoin this includes the reward address, e.g. the Bitcoin address to which the bitcoin will be sent.
#   * $c$&mdash;random counter
#     * This is expected to be incremented until a correct proof is found.
#   * $H(s, c)$&mdash;hash output (as whole number)
#   * $n$&mdash;size of the hash
#     * E.g. 256 in SHA-256
#   * $k$&mdash;work factor
#     * Higher numbers are base-2 exponentially harder.
# * For example assume $H=\text{SHA-4}$ (an imaginary 4-digit hash function), so $n=4$. $k=2$. $c=0$. $s=\text{foo}$. Suppose $H(s, c) = 0010$. Then we have:
#    
#   $$H(s, c) // 2^{(4 - 2)} = \text{num}(0010) \: // \: 2^{2} = 2 \: // \: 4 = 0$$
#   $$\therefore H(s, c)\text{ is valid}$$
#   
# * Bitcoin targets 10-minute block intervals. This cannot be achieved with the reference implementation, as difficulty can be scaled up only exponentially, by adding extra digits of zero-ness. So Bitcoin modifies the proof of work to:
# 
#   $$H(s,x,c) < 2^(n-k)$$
#   
#   Where $k$ is no longer constrained to be an integer, but can be fractional.
#   
# * Since the second value is controlled by the network we may alias it to $\text{target} = 2^(n - k)$.
# * Luck causes variance in block time but the mean value is periodically adjusted over a prior time period to be 10 minutes.
# 
# 
# * In order to dig deeper into the Bitcoin network architecture we have to learn a bit of cryptoanalysis notation as well as some terminology specific to but widely used in the Bitcoin community.
# * The so-called **security margin** of a cryptographic hash function is the factor $k$ in the $O(2^k)$ time it takes to brute-force a hash function. Brute-forcing a hash function here means the difficulty of finding an object with hashes to a specific value under hashing from scratch via the only known method, random search.
# * For example, DES offers 56 bits of security. ECDSA-256 offers 128 bits of security.
# * The concept of the security margin can be extended to a proof of work algorithm, as the proof of work algorithm is (effectively) a (partial) cryptographic function inversion via random brute-force search.
# * The security margin of the Bitcoin network is continuously adjusted to meet the "10 minutes interval between blocks mean interval" design requirement.
# * The Bitcoin network's total rate of work across all miners is known as the **hash rate**. The standard unit of measurement of this value is gigahash/second, or GH.
# * We can therefore reverse engineer the target security margin of the network as a function of this hash rate: $S = \log_2{(\text{hashrate} * 600)}$. In November 2013 the network hashrate was 4 petagigahashes. Peta is $10^{15}$, and giga is $10^9$, so that's $10^{24}$ hashes per second! Achieving the desired block discovery rate requires `math.log(4 * 10**9 * 10**15 * 600)` of hardness, which works out to 63 bits. So we can logically deduce that as of November 2013 the bitcoin network proof of work has a security margin of 63 bits. Though the article states that it's actually 62 for some reason?
# * Knowing this security margin allows us to make comparisons to other hardware hash projects. For example, the EFF built a custom DES cracker designed to crack 56-bit DES in 1998 at the cost of 250,000 dollars in 1998 money. This took them 56 hours. By comparison, the security margin of the bitcoin network in November 2013, with its 62 bit security margin, is 537,000 times harder. The 2013 bitcoin mining network could crack DES in 9 seconds! `2**56 / (2**62 / 600) = 9.375`
# * By January 2019 the security margin was 70 bits. In other words, the network enumerated approximately $2^{70}$ hashes in 10 minutes. `2**56 / (2**70 / 600) = 0.0366`. This hash rate could break DES in just two seconds.
# 
# ## Miner privacy
# This section continued from the previous one, using the same reference article.
# 
# * To ensure privacy miners should in principle use a different reward address for every block they mine and reset the counter to 0 each time they start a new mining run. Given a sufficiently powerful miner that does not reset the counter, the counter can bleed information about a block's miner's identity. Additionally you can technically infer from the counter whether the block's miner is an unlucky powerful miner or a lucky weak one.
# * Bitcoin achieves this by incrementing the counter by a random amount by using two different nonces: nonce, and random nonce. The latter is randomly incremented.
# * As the cost to mine a bitcoin increased, **mining pools** were introduced. A mining pool has miners use the same reward address for every user in the pool. A naive stateless implementation of this system can have miners potentially duplicate work. To avoid this, early implementations of the mining pool protocol had the pool assign each miner a block to mine instead. This meant that miners were not validating their own block (what does this mean? I don't know just yet), which reduced the security of the network as a whole.
#   
#   With the introduction of extra nonce current best practice is to have the pool's miners choose which parts of the blocks to work on, work on them, and submit them to the pool. Sufficiently random iteration prevents work collisions whilst retaining the statelessness of the system.
# 
# ## Proof of stake
# This section taken from the Ethereum wiki article on proof of stake: https://github.com/ethereum/wiki/wiki/Proof-of-Stake-FAQ.
# 
# * Many newer cryptocurrencies use a **proof of stake** system instead of a proof of work one. Quoting the definition in the wiki:
# 
#   > Proof of Stake (PoS) is a category of consensus algorithms for public blockchains that depend on a validator's economic stake in the network.
# * TODO: go through this article in more detail later.
# 
# ## Why is proof of work necessary
# * Where does money come from? In traditional (centralized) monetary systems it is printed by a central bank. Printing money is one of the central functions of such banks, which print new money out of thin air and disbursing it to the public, generally by lending it to the country's banks.
# * However, one of the explicit goals of Bitcoin and other cryptocurrencies is decentralization. The Bitcoin network is designed with two major goals in mind.
#   * The first is trustlessness: peers can participate in transactions with one another without necessarily knowing anything about one another beyond their respective bitcoin addressses.
#     * Consider the alternative, paper money. With paper money it is the responsibility of the banks that store money and the lenders and vendors that transact with it to verify that you actually have enough of it to perform the transaction. This system is explicitly built up on trust.
#   * The second is decentralization: all of the monetary policies followed by the network were designed in advance and are verifiable on-chain; there is no need central bank like authority charged with currency management.
#     * Again, consider the alternative. In a paper money currency the central bank has enormous power.
# * We still need to create money and disburse it somehow, but since we are decentralized, we have to be able to do so without any central authority controlling the process. Additionally, we need this disbursement to be both trustless and fair.
# * Proof of work is the technique that Bitcoin uses to achieve this.
# 
# a proof of work containing a transaction, the version of the transaction that is included in that block is cleared. It becomes impossible to clear the other version of that transaction. See [my notes bitcoin](https://www.kaggle.com/residentmario/notes-on-bitcoin) for more deets.
# * Proof of work allows coins to be 
