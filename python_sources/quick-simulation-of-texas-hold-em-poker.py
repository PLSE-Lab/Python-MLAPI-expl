
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import random

def shuffle(deck):
    random.seed()
    sortdeck = sorted(list(deck.keys()))
    playdeck = []

    while len(sortdeck)>0:
        i = random.randint(0,len(sortdeck)-1)
        playdeck.append(sortdeck.pop(i))

    return playdeck

def deal(pdeck,players):
    for p in players.keys():
        players[p]['Hand'] = []

    for ci in [1,2]:
        for p in players.keys():
            players[p]['Hand'].append(pdeck.pop(0))

    river = []
    for ci in range(5):
        river.append(pdeck.pop(0))

    for p in players.keys():
        players[p]['Cards'] = players[p]['Hand']+river

    return players, river


def calc_points(cards,verbose=0):  ##Takes 7 cards and returns the points
    hands = ['HC','OP','TP','TK','ST','FL','FH','FK','SF']
    handrank = {}
    for i in range(len(hands)):
        handrank[hands[i]] = 100*i

    sameval = {}
    samecol = {}
    st = []
    for c in cards:
        val = deck[c]['Value']
        col = deck[c]['Suit']
        try:
            sameval[val]+=1
        except:
            sameval[val]=1
        try:
            samecol[col]+=1
        except:
            samecol[col]=1
        if not val in st:
            st.append(val)
        st.sort()

    if verbose==1:print(cards)
    #print(sameval)
    #print(samecol)
    #print(st)

    flush = 0
    flushcol = ''
    for c in samecol.keys():
        if samecol[c]>=flush:
            flush = samecol[c]
            flushcol = c

    kind = 0
    kinds = []
    for v in sameval.keys():
        if sameval[v] > 1:
            kinds.append((sameval[v],v))
        if sameval[v] > kind:
            kind = sameval[v]

    straight = 1
    run = 1
    maxst = 0
    for i in range(len(st)-1):
        if st[i+1] == st[i] + 1:
            run+=1
        else:
            if run>straight:
                straight = run
                maxst = st[i]
            run = 1
    if run>straight:
        straight = run
        maxst = st[i]

    if verbose==1:print(flush,kind,kinds,straight,maxst)

    points = 0

    ##Check for SF
    if straight >=5 and flush >=5:
        if verbose==1:print('possible SF')
        fc = []
        st = []
        for c in cards:
            if deck[c]['Suit']==flushcol:
                st.append(deck[c]['Value'])

        straightf = 0
        run = 1
        for i in range(len(st)-1):
            if st[i+1] == st[i] + 1:
                run+=1
            else:
                if run>straightf:
                    straightf = run
                run = 1

            if run>straightf:
                straightf = run

        if straightf >=5:
            ##Straight Flush won
            points = handrank['SF']+max(st)

                  
    ##Check for FK
    if points == 0 and kind == 4:
        if verbose==1:print('possible FK')
        for k in kinds:
            if k[0] == 4:
                points = handrank['FK']+k[1]

        cardrank = sorted(list(sameval.keys()))
        if k[1] == cardrank[-1]:
            hc = cardrank[-2]
        else:
            hc = cardrank[-1]
        points = points + float(hc)/100
        

    ##Check for FH
    if points == 0 and kind == 3:
        if verbose==1:print('possible FH')
        fh = {'3':0,'2':0}
        for k in kinds:
            if k[0] == 3:
                fh['3'] = k[1]
            elif k[0] == 2:
                if k[1] > fh['2']:
                    fh['2']=k[1]

        if fh['3'] > 0 and fh['2'] > 0:
            points = points + handrank['FH']+fh['3']+float(fh['2'])/100

    ##Check for Flush
    if points == 0 and flush>=5:
        if verbose==1:print('possible FL')
        fc = []
        for c in cards:
            if deck[c]['Suit']==flushcol:
                st.append(deck[c]['Value'])

        points = handrank['FL']+max(st)

    ##Check for ST
    if points == 0 and straight >=5:
        if verbose==1:print('possible ST')
        points = points + handrank['ST']+maxst


    ##Check for TK
    if points == 0 and kind ==3:
        if verbose==1:print('possible TK')
        maxval = 0
        for k in kinds:
            if k[0] == 3 and k[1] > maxval:
                maxval = k[1]

        points = handrank['TK']+maxval

        cardrank = sorted(list(sameval.keys()))
        if maxval == cardrank[-1]:
            hc = cardrank[-2]
        else:
            hc = cardrank[-1]
        points = points + float(hc)/100


    ##Check for TP
    if points == 0 and kind ==2:
        if verbose==1:print('possible TP or OP')
        vals = []
        for k in kinds:
            if k[0] == 2:
                vals.append(k[1])

        vals.sort()
        if len(vals) >= 2:
            points = handrank['TP'] + vals[-1] + float(vals[-2])/100
        else:
            points = handrank['OP'] + vals[0]

        cardrank = sorted(list(sameval.keys()))
        hc = cardrank.pop(-1)
        while hc in vals[-2:]:
            hc = cardrank.pop(-1)
        points = points + float(hc)/10000
        
        
    ##Check for HC
    if points == 0:
        if verbose==1:print('Just HC')
        points = max(list(sameval.keys()))

    if verbose == 1: print(points)
        
    return points

def writehands(hands,fname):
    outf = open(fname,'w')
    outf.write('Hand\tCard1\tCard2\tSame\tWin\tDraw\tPlayed\tValue\n')
    for h in sorted(hands.keys()):
        hand = h.split(' ')
        if hand[1] == 'diff':
            same = '0'
        else:
            same = '1'

        hand = hand[0].split('/')
        outf.write(h+'\t')
        outf.write(hand[0]+'\t'+hand[1]+'\t'+same+'\t')
        outf.write(str(hands[h]['Win'])+'\t')
        outf.write(str(hands[h]['Draw'])+'\t')
        outf.write(str(hands[h]['Played'])+'\t')
        try:
            score = (hands[h]['Win']+0.5*hands[h]['Draw'])/hands[h]['Played']
        except:
            score = ''
        outf.write(str(score)+'\n')

    outf.close()
        

def abstr_hand(hand,deck):
    
    cols = ' diff'
    if deck[hand[0]]['Suit'] == deck[hand[1]]['Suit']:
        cols = ' same'

    if deck[hand[0]]['Value'] > deck[hand[1]]['Value']:
        vals = deck[hand[0]]['Name'] + '/'+deck[hand[1]]['Name']
    else:
        vals = deck[hand[1]]['Name'] + '/'+deck[hand[0]]['Name']

    name = vals + cols
    return name
    

vals = ['2','3','4','5','6','7','8','9','10','J','Q','K','A']
cols = ['Hearts','Diamonds','Clubs','Spades']

deck = {}
for i in range(len(vals)):
    for c in cols:
        card = vals[i] + ' of '+c
        deck[card] = {}
        deck[card]['Value']=i
        deck[card]['Name']=vals[i]
        deck[card]['Suit']=c
        

hands = {}
for c1 in deck.keys():
    for c2 in deck.keys():
        if not c1==c2:
            hand = [c1,c2]
            hands[abstr_hand(hand,deck)] = {'Win':0,'Draw':0,'Played':0}
            

numplayers = 6
numgames = 100
players = {}
for i in range(numplayers):
    players[i] = {}
    players[i]['Win'] = 0
    players[i]['Hand'] = []

for game in range(numgames):
    
    p = shuffle(deck)
    players,river = deal(p,players)

    ranking = []
    for p in players.keys():
        players[p]['Points'] = calc_points(players[p]['Cards'])
        ranking.append((players[p]['Points'],p))
        hands[abstr_hand(players[p]['Hand'],deck)]['Played']+=1
    ranking.sort(reverse=True)

    hp = ranking[0][0]
    winners = []
    for r in ranking:
        if r[0] == hp:
            winners.append(r[1])

    if len(winners) == 1:
        print('Game '+str(game)+ ' - Winner: Player '+str(winners[0])+' with '+str(hp))
        hands[abstr_hand(players[winners[0]]['Hand'],deck)]['Win']+=1
    else:
        print('Game '+str(game)+ ' - Multiple Winners: '+str(winners)+' with '+str(hp))
        for p in winners:
            hands[abstr_hand(players[p]['Hand'],deck)]['Draw']+=1


writehands(hands,'Hands_6p.txt')



##Testing

##c1 = ['2 of Hearts', '3 of Hearts', '4 of Hearts', '5 of Hearts', '6 of Hearts', 'Q of Clubs', 'Q of Spades']  
##c2 = ['2 of Hearts', '3 of Hearts', '4 of Hearts', 'Q of Diamonds', 'Q of Hearts', 'Q of Clubs', 'Q of Spades']  
##c3 = ['2 of Hearts', '3 of Hearts', '4 of Hearts', '7 of Hearts', 'Q of Hearts', 'Q of Clubs', 'Q of Spades']  
##c4 = ['2 of Hearts', '3 of Hearts', '4 of Clubs', '5 of Hearts', 'Q of Hearts', 'Q of Clubs', 'Q of Spades']  
##c5 = ['2 of Hearts', '3 of Hearts', '2 of Clubs', '5 of Hearts', 'Q of Hearts', '5 of Clubs', 'Q of Spades']  
##c6 = ['2 of Hearts', '3 of Hearts', '9 of Clubs', '5 of Hearts', 'Q of Hearts', '5 of Clubs', 'Q of Spades']  
##c7 = ['2 of Hearts', '3 of Hearts', '9 of Clubs', '5 of Hearts', 'Q of Hearts', '5 of Clubs', 'K of Spades']  
##c8 = ['2 of Hearts', 'A of Hearts', '9 of Clubs', '5 of Hearts', 'Q of Hearts', '7 of Clubs', 'K of Spades']  
##
##print(calc_points(c8))
