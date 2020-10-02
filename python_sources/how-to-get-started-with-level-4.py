#!/usr/bin/env python
# coding: utf-8

# In the folllowing I assume the first three ciphers are completely known so you can encrypt your plaintext to get the input to the cipher at level 4.
# 
# Now we want to find a pair of ciphertext and plaintext - but is this possible and how can we do it.
# 

# First we make a statistic of the groups in all the ciphertexts
#     Here are the 20 most frequent groups
# 
#     422957 different groups found
# 
#      group    count
#        411:  585378
#        418:  139858
#     193268:   41924
#     551143:   41883
#     198005:   38612
#     545596:   38407
#     532384:   38387
#     318123:   38333
#     313272:   38314
#     481569:   38303
#     444660:   38295
#     555198:   38275
#     151505:   38253
#     195482:   38247
#      88645:   38241
#     223156:   38232
#     380102:   38220
#     198014:   38220
#     539530:   38201
#     374397:   38201
# 
#         We observe that 411 and 418 may be special in some way.
# 

# 

#     Make a count of how many groups there are in each ciphertext
# 
#     11195 ciphertext of difficulty 4 found
# 
#      group  text
#      count  count
#       100:     6
#       200:    50
#       300:   319
#       400:   458
#       500:   414
#       600:   477
#       700:  1353
#       800:  1292
#       900:   958
#      1000:   798
#      1100:   636
#      1200:   502
#      1300:   449
#      1400:   405
#      1500:   313
#      1600:   298
#      1700:   262
#      1800:   227
#      1900:   211
#      2000:   184
#      2100:   148
#      2200:   134
#      2300:   121
#      2400:   102
#      2500:    92
#      2600:    96
#      2700:    73
#      2800:    59
#      2900:    69
#      3000:    62
#      3100:    58
#      3200:    42
#      3300:    44
#      3400:    46
#      3500:    39
#      3600:    30
#      3700:    31
#      3800:    31
#      3900:    29
#      4000:    25
#      4100:    22
#      4200:    20
#      4300:    15
#      4400:    13
#      4500:    20
#      4600:    18
#      4700:    10
#      4800:    15
#      4900:    14
#      5000:     7
#      5100:    12
#      5200:     8
#      5300:    14
#      5400:     9
#      5500:    14
#      5600:    11
#      5700:     6
#      5800:     5
#      5900:     4
#      6000:     7
#      6100:     2
#      6200:     2
#      6300:     2
#      6500:     1
#      7000:     1
#      
#      Because we get length that are multiples of 100 (like our padded plaintext)
#      this indicates that each group could be a substitution for one character.
# 

#      Let's make a statistic of the plaintext length of plaintexts that have not yet been paired in the three previous solved levels and have a length a multiple of 100 to avoid the unknown padding characters.
# 
#        95 plaintexts found with the following lengths
# 
#       200:     1
#       300:     1
#       400:     2
#       500:     4
#       600:     3
#       700:    15
#       800:     8
#       900:    12
#      1000:     2
#      1100:     3
#      1200:     3
#      1300:     7
#      1400:     5
#      1500:     4
#      1600:     5
#      1700:     1
#      1800:     1
#      2000:     3
#      2100:     1
#      2200:     1
#      2300:     2
#      2400:     2
#      2600:     1
#      2800:     3
#      3100:     1
#      3300:     1
#      3700:     1
#      3900:     1
#      5300:     1
# 

#   We see that the longest unused (in previous levels) plaintext a multiple of length 100 has length 5300 and there is only one.
#   We have 14 ciphertexts with 5300 groups so one of them must pair up with our plaintext.

# We encrypt the known plaintext with our 3 previous ciphers to get a known input to the fourth cipher.

# We align each character in the encrypted plaintext to the cipher groups of each ciphertext

#     ID_6e4650438      S      o             o      n      e             w      o      u      l      d             t      h      i      n ...
#     ID_Encrypted      b      k      J      !      i      7      7      E      L      7      [      C      e      m      7      c      k ...
#     ID_03a11529f 426943 405719    418 107342 391215 401616 539969 216435 139378 212536 511964 315592 250470 552264 378015 157087 206388 ...
#     ID_97d0db384 139410  73508 336351 300731 444660 404843 312628   1798 318123 213521 191319  31260 313957 400377 547366 473641  70031 ...
#     ID_1bf09d38f 335808 522127 146389 509533  16591 398122 212590 539406 521765 545596    411  53489 127416 145298 198014 343980 351908 ...
#     ID_b41fce299  99256 241297 492905    423 112539 380515 537756  18979 394487 197206 545926  31855 553081 546227 123287 372429    411 ...
#     ID_11237c3ba 105294 324116  89422 418126 555412 484690 378159 540865 109919 203777 539665 198151  38448 151505 219354 548778 353740 ...
#     ID_2a0294e5f 547793 252614  88645 501567 419912  51718 395612 315657 380102 226612 290700  87523  76427 223156  70031 315809 540249 ...
#     ID_9fb4824a6 548439  59242 413755 151505  52466 343151 365429 239793 432002 526188 384446 419725 292430 309700 364772    411 380102 ...
#     ID_61bf51731   1143 447500    411 312656 364757   2615    244  13183 484528 551143 343123 521850 530564 430524 364772 360371 213521 ...
#     ID_e14152446 539549 139402  23259 535797 511328 525161  93464 539530 109252 362475 161541 350502 444263 524165 122869 407141 427702 ...
#     ID_e6648e11e 544531 556153  84752 377017 209572  13041 240684 549090 474447 271873  14853  80845 123108 351091 117557 279809 170916 ...
#     ID_d019c03ca 275207 460376 152210 539684 279306    418 476128 396839 155487 555231 294651 237636    418    411 377717    411  81631 ...
#     ID_d9406fff0 157140 383442    411  84493 549364 424198 365404 359967 318123    411 292812    411 162421 496410 509492 377717 418126 ...
#     ID_3e24c6e7c 554517 356535 129447 291516 349863 370036 467723 280694 285329 527603 174606 463405 375681 336637 463405    418 312194 ...
#     ID_5a6358252 291748 292282  94863  13757 494519 223156 437079    418 224585 458811 532369 141273 376776 193268 410762    411 116571 ...
# 

#     Now we look for the special groups 411 and 418 that we observed in a previous step and see if we can find something useful
# 
#     After some time we see that when looking at ciphertext ID_1bf09d38f we notice that
# 
#         411 always come from [ in encrypted plaintext
#         418 always come from # in encrypted plaintext
# 
#     looking a little deeper gives us that
# 
#         423 always come from ] in encrypted plaintext
# 
#     So now we have our first plaintext/ciphertext pair = (ID_6e4650438, ID_1bf09d38f)  -> index = 9188
# 
#     Now there is only 11194 more ciphertext to pair up in some way and I will leave that up to you ;-)

# 
