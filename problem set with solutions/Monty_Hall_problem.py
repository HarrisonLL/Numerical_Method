import numpy as np
choices = ['g','g','c']
swap_wins = 0
not_swap_wins = 0
experients = 100000

## senario 
## choose one door out of three
## host will show one door having a goat
## contest then decide whether to swap the door he chosed 
## 'g' stands for goat, 'c' stands for car(won)
##  the probability for swaping will be around 2/3
##  the probability for not swaping will be around 1/3

for i in range(experients):
    contest_c = np.random.choice([0,1,2],1)
    if contest_c == 0 or contest_c == 1:
        swap_wins += 1
    elif contest_c == 2 :
        not_swap_wins += 1
swap_win_prob = float(swap_wins)/experients
not_swap_win_prob = float(not_swap_wins)/experients
print(swap_win_prob)
print(not_swap_win_prob)
    
        
        


