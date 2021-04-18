import numpy as np
from sympy import *
import mystic.symbolic as mys
import matplotlib.pyplot as plt
# from symengine import solve

'''
This is our mu_s(theta) function
Hard-coded in the return values for now because this is a simple model
Gets a value of s (the type)
Returns the posterior distribution of s given theta
'''
def mu(s):
	if s==1:
		return 1-theta
	elif s==2:
		return theta

'''
This is the expectation of theta given s
Gets a value of s
Returns the expectation value according to the formula given in the paper
'''
def thetaBar(s):
	bar = integrate(theta*mu(s), (theta, 0, 1))/integrate(mu(s), (theta, 0, 1))
	return bar

'''
This is our persuasion menu
Since we're using sympy, we can simply reference the bounds as they are (x1 etc.) instead of having to give them specific values here
Gets a value of s
Returns the symbolic bounds for that type of receiver
'''
def p_s(s):
	if s==1:
		return x1, y1
	elif s==2:
		return x2, y2

'''
This is our f_s(theta) function
Gets a value of s
Returns the posterior distribution of theta given s
'''
def f_s(s):
	return 2*mu(s)

'''
This is the receiver's payoff
Gets the receiver's true type (s) and the type of the persuasion mechanism they chose (sHat), need not be the same
Returns the payoff according to the formula from the paper
'''
def u_p(sHat, s):
	xLim, yLim = p_s(sHat)
	up = integrate((theta - c)*f_s(s), (theta, xLim, yLim))
	return up

'''
This is the sender's reward function
For now, it's hard-coded as 1 for reasons explained in the paper, but allows for flexibility in the future
Gets value of theta
Returns reward for sender as a function of theta
'''
def r(theta):
	return 1

'''
This is the sender's payoff
Gets the bounds of the persuasion menu (could be passed in as an array in the future for more types of receiver)
Returns the payoff according to the formula from the paper`
'''
def vSend(x1, y1, x2, y2):
	return integrate(r(theta)*mu(1), (theta, x1, y1)) + integrate(r(theta)*mu(2), (theta, x2, y2))

'''
Defining our symbols
theta is the standard state variable
c is the cost of the policy
k is our cutoff in the cutoff model
x1, y1, x2, and y2 are the bounds for our persuasion menu
'''
theta = symbols('\u03B8')
c = symbols('c', real=True)
k = symbols('k', real=True)
x1, y1, x2, y2 = symbols('x1 y1 x2 y2', real=True) #, nonnegative=True)

'''
This next block was a first attempt at solving the locally incentive compatible conditions one-by-one
Each one was individually successful, but when combining them together, it didn't work
'''
# y2 = 1
# lic1 = solve(u_p(1, 1) >=0, c)
# lic2 = solve(u_p(2, 2) >= u_p(1, 2), c)
# lic3 = solve(u_p(1, 1) >= u_p(2, 1), c)
# print(lic1)
# print(lic2)
# print(lic3)
#print(solve([lic1, lic2, lic3], (x1, x2, y1)))
# print(solve(u_p(1, 1) >=0, (x1, x2, y1)))
# print(solveset(u_p(1, 1) >=0, x1, domain=S.Reals))
# print(solveset(2*c*x1 - 2*c*y1 + 2*x1**3/3 - x1**2*(c + 1) - 2*y1**3/3 + y1**2*(c + 1) >=0, y1, domain=S.Reals))
# y1=solve(u_p(1,1),y1)[1]
# x1=c
# print(expand(u_p(1,2))) # Save this line
# print(solve(u_p(1,2)-(2/3-c),x1))
# print(solve(3*c**2 + 4*c*x1 - 10*c - 4*x1**2 + 4*x1 + 3, x1))
# print(mu(1))

'''
Next tried the nonlinsolve method to deal with the nonlinearity of these equations, was also unsuccessful
'''
varisStr = ('x1', 'y1', 'x2')
varis = [x1, y1, x2]
print(nonlinsolve([x1-k, c -1/2*(x1+x2)], [x1,x2]))
# print(u_p(2, 2)-u_p(1, 2))
#equalSol = nonlinsolve([u_p(1, 1), u_p(2, 2) - u_p(1, 2), u_p(1, 1) - u_p(2, 1)], varis)
# optSol = nonlinsolve([u_p(1, 1), u_p(2, 2) - u_p(1, 2)], varis)
# numSol = nsolve((u_p(1, 1), u_p(2, 2) - u_p(1, 2), u_p(1, 1) - u_p(2, 1)), varis, (0, 0, 0))
# numSol = nsolve((u_p(1, 1), u_p(2, 2) - u_p(1, 2), u_p(1, 1) - u_p(2, 1)), varis, (0, 0, 0))
# numSol = nsolve(u_p(1, 1), x1, 0)
# print(optSol)
#print(nonlinsolve([2*c*x1 - 2*c*y1 + 2*x1**3/3 - x1**2*(c + 1) - 2*y1**3/3 + y1**2*(c + 1),-c*x1**2 + c*x2**2 + c*y1**2 - c*y2**2 + 2*x1**3/3 - 2*x2**3/3 - 2*y1**3/3 + 2*y2**3/3], [x1, y1]))
# print(u_p(2,2)-u_p(1,2))
y2=1
# c=0.5
# y1 = solve(u_p(1,1), y1)[1]
# print(y1)
# x2 = solve(u_p(2,2)-u_p(1,2),x2)[0]
# print(u_p(2,2))
# print(solve(u_p(1,1), y1))

'''
Tried to numerically solve the system of equations, unsuccessfully
'''
# print(nsolve([u_p(1, 1)>=0, u_p(2, 2) >= u_p(1, 2), u_p(1, 1) >= u_p(2, 1)], [x1, y1, x2], (0, 0, 0)))
# print(solve([2*c*x1 - 2*c*y1 + 2*x1**3/3 - x1**2*(c + 1) - 2*y1**3/3 + y1**2*(c + 1) = 0], [x1]))#, domain=S.Reals))

'''
Tried to find pooling solution and compare to the discriminatory one
Was successful, but uninformative
'''
# poolSol = solve([y1-y2,x2-x1,u_p(1,1),u_p(2,2)-u_p(1,2)], [y1, y2, x2])[2]
# # discSol = solve([u_p(1,1),u_p(2,2)-u_p(1,2)], [x1,x2, y1, y2])
# y1 = poolSol[0]
# y2 = poolSol[1]
# x2 = poolSol[2]
# print(poolSol)

'''
Tried to optimize the sender's payoff numerically, did not work (program ran for too long)
'''
# print(solve(diff(vSend(x1, y1, x2, y2),y2),y2))
# print(nsolve(diff(vSend(x1, y1, x2, y2),x1),x1,.2))
# print(vSend(x1, y1, x2, y2))
# testdiff = diff(vSend(x1, y1, x2, y2),x1)
# # print(testdiff)
# sendPay = vSend(x1, y1, x2, y2)
# print(sendPay)
# print(nsolve(testdiff, y2, .7))
# y2=0
# y2 - ((6*c - 12*y2 + 6)/(2*sqrt(9*c**2 + 12*c*y2 - 30*c - 12*y2**2 + 12*y2 + 9)) - 1)*(3*c/4 - y2/2 + sqrt(9*c**2 + 12*c*y2 - 30*c - 12*y2**2 + 12*y2 + 9)/4 + 3/4)/2
# print(sendPay)

'''
Produced implicit plots to try and shed some light on the relationship between x1 and c, and x1 and y1
Chose not to include in paper because I was unsure of how to analyze them, but this did work
Worthy of future study
'''
# plot_implicit(testdiff, (c, 0, 1), (x1, 0, 1)) 
# plot_implicit(y1=-0.5*x1 - 0.21650635094611*sqrt(-16.0*x1**2 + 24.0*x1 - 5.0) + 1.125, (x1, 0, 1), (y1, 0, 1))
# x1 = np.arange(0,1,1/1000)
# plt.plot(x1, x1**2)
# x1 = np.arange(0, 1, .001)
# y1=-0.5*x1 - 0.21650635094611*np.sqrt(-16.0*x1**2 + 24.0*x1 - 5.0) + 1.125
# z1 = -0.5*x1 + 0.21650635094611*np.sqrt(-16.0*x1**2 + 24.0*x1 - 5.0) + 1.125
# plt.plot(x1, y1, 'r-', x1, x1, 'g-', x1, z1, 'b-')
# plt.show()


'''
This block was an attempt to use the symbolic class in the mystic library
To solve equations in this, I first had to set out a string of the constraints
Tried different ways of doing it, first by doing string replacement, then by copying and pasting in the actual equations
Even tried making the inequalities into equalities
Kept producing an EOF (end of file) error in the mys.simplify class
'''

# constraints = '%s = 0 \n' \
# 			'%s = %s \n' \
# 			'%s = %s' \
# 			% (str(u_p(1, 1)), str(u_p(2, 2)), str(u_p(1, 2)), str(u_p(1, 1)), str(u_p(2, 1)))
constraints = '%s >= 0' \
			% (str(u_p(1, 1)))
# constraints = '%s >= 0 \n' \
# 			'%s >= %s \n' \
# 			'%s >= %s' \
# 			% (str(u_p(1, 1)), str(u_p(2, 2)), str(u_p(1, 2)), str(u_p(1, 1)), str(u_p(2, 1)))
# 
# constraints = '''2*c*x1 - 2*c*y1 + 2*x1**3/3 - x1**2*(c + 1) - 2*y1**3/3 + y1**2*(c + 1) = 0
# c*x2**2 - c - 2*x2**3/3 + 2/3 = c*x1**2 - c*y1**2 - 2*x1**3/3 + 2*y1**3/3
# 2*c*x1 - 2*c*y1 + 2*x1**3/3 - x1**2*(c + 1) - 2*y1**3/3 + y1**2*(c + 1) = 2*c*x2 - c + 2*x2**3/3 - x2**2*(c + 1) + 1/3'''
# print(constraints)
#constraints = '2*c*x1 - 2*c*y1 + 2*x1**3/3 - x1**2*(c + 1) - 2*y1**3/3 + y1**2*(c + 1) >= 0'
# print(u_p(2, 2))	
# varis = ['x1', 'y1', 'x2', 'y2', 'c']

# print(mys.simplify(constraints, variables=['x1', 'y1', 'x2', 'c']))




