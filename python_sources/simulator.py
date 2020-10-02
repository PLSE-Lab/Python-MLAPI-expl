# 

def simulate(hitters):
    inning = 1
    runs = 0
    currentHitter = 0
   
    while inning <= 9:
#         print('\nInning: {}'.format(inning))
        # Start the inning with no outs and the  bases clear
        outs = 0
        runner = ['empty','empty','empty'] # Runners on first, second, and third
        
        while outs < 3:
#             print(' ')
#             print(hitters[currentHitter])
#             print('is hitting')
            pa = hitters[currentHitter].PlateAppearance()
            error = 1 if (np.random.random() > .984) else 0 # Fielding pct for league is .984 according to Baseball Reference
            if pa > 0:
                # Hitter got on base, need to move runners and put the hitter on base
                extraBase = pa+1 if (np.random.random() < .7) else 0
                for x in [2,1,0]: # Move runner from third, then from second, then from first.
                    if runner[x] != 'empty':
#                         print(runner[x])
                        result = runner[x].Run(x,pa) + extraBase + error
                        if (result % 1 > 0) & (error == 0): # Runner is out
                            outs += 1
#                             print('is out. {} outs'.format(outs))
                        elif result >= 4.0: # Runner scored
                            runs += 1
#                             print('scored!!!!!!!!!!!!!!!!!!!!!!!!')
                        else:
                            if runner[int(result)-1] != 'empty': # Base is occupied, runner cannot advance
                                result -= 1
                                runner[int(result)-1] = runner[x]
#                                 print('is out, base occupied')
                            else:
                                runner[int(result)-1] = runner[x]
#                                 print('runner reached {}'.format(result))
                        runner[x] = 'empty'
                if pa >= 4:
#                     print(hitters[currentHitter])
#                     print('Hits a HOME RUN!!!!!!!!!!!')
                    runs += 1
                else:
#                     print(hitters[currentHitter])
#                     print('is on {}'.format(pa))
                    runner[int(pa)-1] = hitters[currentHitter]
#                 print(hitters[currentHitter])
#                 print('is on {}'.format(int(pa)))
            else:
                if error > 0: 
                    if runner[2] != 'empty':
                        runs +=1
                        runner[2] = 'empty'
                    if runner[1] != 'empty':
                        runner[2] = runner[1]
                        runner[1] = 'empty'
                    if runner[0] != 'empty':
                        runner[1] = runner[0]
                    runner[0] = hitters[currentHitter]
                else:
                    outs += 1
#                 print(hitters[currentHitter])
#                 print('is out. {} outs'.format(outs))
              
            # Move on to the next hitter, going to the top of the lineup after the last
            currentHitter += 1
            if currentHitter > (len(hitters)-1):
                currentHitter = 0
                
        inning += 1 # Inning over, go to the next inning
    
    return(runs)
                        
            