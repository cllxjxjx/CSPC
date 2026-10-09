"""
PW2 Lab B Part 2 -- three routes to a minimum.

Compare gradient descent, Newton, and SLSQP on two functions:
  2A: f(x) = (x-3)**2 + 1          (easy, one minimum at x=3)
  2B: g(x) = x**4 - 3*x**2 + x + 5 (harder, several stationary points)
Run:  python warmup.py
"""
import numpy as np
from scipy.optimize import newton, minimize

# ---------- 2A: easy convex function ----------
def f(x):   return (x-3)**2 + 1
def df(x):  return 2*(x-3)
def d2f(x): return 2.0

x_grad = 0
lr = 0.1
for _ in range(1000):
  step = lr*df(x_grad)
  x_grad-=step
  if abs(step) < 1e-6:
    break
print(f"Gradient Descent: {x_grad}")

x_N = newton(df,0.0,fprime = d2f)
print(f"Newton Method: {x_N}")

x_SLSQP = minimize(f,0.0,method = "SLSQP")
print(f"SLSQP method:{x_SLSQP.x[0]}")


# ---------- 2B: harder landscape ----------
def g(x):   return x**4 - 3*x**2 + x + 5
def dg(x):  return 4*x**3 - 6*x + 1
def d2g(x): return 12*x**2 - 6


def run2b(x0):
  x_grad = x0
  lr = 0.1
  for _ in range(1000):
    step = lr*dg(x_grad)
    x_grad-=step
    if abs(step) < 1e-6:
      break
  print(f"Gradient Descent: {x_grad}")

  x_N = newton(dg,x0,fprime = d2g)
  second_derivative = d2g(x_N)
  print(f"Newton Method: {x_N} Second derivative:{second_derivative}")
  if second_derivative < 0:
      print("Max")
  elif(second_derivative > 0):
      print("Minimum")
  x_SLSQP = minimize(g,x0,method = "SLSQP")
  print(f"SLSQP method:{x_SLSQP.x[0]}")
run2b(0.0)
run2b(2.0)

# TODO 2B: run the same three methods on g, from x0=0 AND from x0=2.
#   For Newton (which solves dg(x)=0), also check the sign of d2g at the answer:
#   d2g > 0 means a minimum, d2g < 0 means a maximum.
#   In your README note: do the methods agree? did Newton find a minimum or
#   another stationary point? how did the starting point change the result?
