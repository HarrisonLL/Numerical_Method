import numpy as np
import scipy.linalg as la

### view test cases and plot in ipynb file 
### only functions in this file

def adj_matrix(home_team, away_team, home_score, away_score):
    team_names = np.union1d(home_team, away_team)
    team_names_dict = {}
    for i,v in enumerate(team_names) :
        team_names_dict[v] = i

    n = len(team_names)
    A = np.zeros((n,n))
    margins = home_score - away_score
    
    for i,v in enumerate(margins):
        if v > 0:
            x_cord = team_names_dict[home_team[i]]
            y_cord = team_names_dict[away_team[i]]
            A[x_cord, y_cord] += v
        else:
            x_cord = team_names_dict[away_team[i]]
            y_cord = team_names_dict[home_team[i]]
            A[x_cord, y_cord] += abs(v)
    return team_names, A


def transition(A):
    # construct the transition matrix M
    M = A.copy()
    col_sum = A.sum(axis=0)
    for i,v in enumerate(col_sum) :
        if v != 0 :
            M[:, i] = A[:, i]/v
        else:
            M[:, i] = 1.0/A.shape[0]
    return M


def power_iteration(M,tol):
    # compute things!
    n = M.shape[1]
    x = np.random.randint(1,100,n)
    x = x/x.sum(axis=0)
    x_new = M@x
    x_new = x_new/la.norm(x_new,1)
    
    while (la.norm(abs(x_new-x),2)>=tol) :
        x = x_new
        x_new = M@x_new
        x_new = x_new/la.norm(x_new, 1)
        
    return x_new



