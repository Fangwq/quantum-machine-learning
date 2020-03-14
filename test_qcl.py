# _*_ coding:utf-8 _*_
# refer to the paper: Quantum circuit learning-Phy.Rev.A 98, 032309 (2018)
import numpy as np
import itertools
import scipy as sp
from scipy.optimize import minimize
import matplotlib.pyplot as plt


np.random.seed(12345)
#sigma z
Z = np.array([[1.0, 0.0],[0.0, -1.0]])
#sigma x
X = np.array([[0.0, 1.0],[1.0, 0.0]])
#identity matrix
I = np.array([[1.0, 0.0],[0.0, 1.0]])
#|0>
zero = np.array([[1.0],
                 [0.0]])
#|1>
one = np.array([[0.0],
                [1.0]])

#data preparation
m = 8
Xdata = np.linspace(-0.95,0.95,m)
Ydata = Xdata**2
n_qubits = 3
print Ydata

print("===========calculate initial state=============")
#initial state
initial_state = np.kron(np.kron(zero, zero), zero)
print(initial_state)

##==================calculate input state=================
# RZ(arccos(x))RY(arcsin(x))
print("===========calculate input state=============")
# three rotation matrix
RX = lambda theta: np.array([[np.cos(theta/2.0),-1.0j*np.sin(theta/2.0)],
                             [-1.0j*np.sin(theta/2.0),np.cos(theta/2.0)]])
RY = lambda theta: np.array([[np.cos(theta/2.0),-np.sin(theta/2.0)],
                             [np.sin(theta/2.0),np.cos(theta/2.0)]])
RZ = lambda theta: np.array([[np.exp(-1.0j*theta/2.0),0],
                             [0,np.exp(1.0j*theta/2.0)]])

# qubit one 
RX0 = lambda x: np.kron(np.kron(RX(x), I), I)
RY0 = lambda x: np.kron(np.kron(RY(x), I), I)
RZ0 = lambda x: np.kron(np.kron(RZ(x), I), I)

# qubit two
RX1 = lambda x: np.kron(np.kron(I, RX(x)), I)
RY1 = lambda x: np.kron(np.kron(I, RY(x)), I)
RZ1 = lambda x: np.kron(np.kron(I, RZ(x)), I)

# qubit three
RX2 = lambda x: np.kron(np.kron(I, I), RX(x))
RY2 = lambda x: np.kron(np.kron(I, I), RY(x))
RZ2 = lambda x: np.kron(np.kron(I, I), RZ(x))

input_result = np.zeros((m, 2**n_qubits, 2**n_qubits)) + 1.0j*np.zeros((m, 2**n_qubits, 2**n_qubits))
for i in range(m):
    U0 = np.dot(RZ0(np.arccos(Xdata[i]**2)), RY0(np.arcsin(Xdata[i])))
    U1 = np.dot(RZ1(np.arccos(Xdata[i]**2)), RY1(np.arcsin(Xdata[i])))
    U2 = np.dot(RZ2(np.arccos(Xdata[i]**2)), RY2(np.arcsin(Xdata[i])))
    input_result[i] = np.dot(np.dot(U0,U1),U2)
print(input_result[0])

##=================calculate hamiltonian================
# evolution operator with the Hamiltonian of connected transverse Ising model
print("===========calculate Hamiltonian=============")
J_coeff = dict()
h_coeff = np.random.uniform(-1.0, 1.0, size=n_qubits)
for val in itertools.combinations(range(n_qubits), 2):
    J_coeff[val] = np.random.uniform(-1.0, 1.0)
T=10
trotter_steps = 1000

Z0 = np.kron(np.kron(Z,I),I)
Z1 = np.kron(np.kron(I,Z),I)
Z2 = np.kron(np.kron(I,I),Z)
Inter1 = J_coeff[(0,1)]*np.dot(Z1, Z0)
Inter2 = J_coeff[(0,2)]*np.dot(Z2, Z0)
Inter3 = J_coeff[(1,2)]*np.dot(Z2, Z1)
X0 = np.kron(np.kron(X, I), I)
X1 = np.kron(np.kron(I, X), I)
X2 = np.kron(np.kron(I, I), X)
nonInter = h_coeff[0]*X0 + h_coeff[1]*X1 + h_coeff[2]*X2
Evolution = np.eye(2**n_qubits)
# Trotter Suzuki approximation
for i in range(trotter_steps):
    Evolution = np.dot(np.dot(sp.linalg.expm(-(1.0j)*T/trotter_steps*nonInter), sp.linalg.expm(-(1.0j)*T/trotter_steps*(Inter1+Inter2+Inter3))), Evolution)
print(Evolution)

##==================calculate output=================
# Qubit: n_qubits, Depth: depth, evolution with QCL framework
print("===========calculate output=============")
depth = 3
initial_theta = np.random.uniform(0.0, 2*np.pi, size=3*n_qubits*depth)
theta = initial_theta.reshape(3,n_qubits,depth)
output = np.eye(2**n_qubits)
for rj in range(depth):
    U0_theta = np.dot(RX0(theta[2,2,rj]), np.dot(RZ0(theta[1,2,rj]), RX0(theta[0,2,rj])))
    U1_theta = np.dot(RX1(theta[2,1,rj]), np.dot(RZ1(theta[1,1,rj]), RX1(theta[0,1,rj])))
    U2_theta = np.dot(RX2(theta[2,0,rj]), np.dot(RZ2(theta[1,0,rj]), RX2(theta[0,0,rj])))
    output = np.dot(np.dot(np.dot(np.dot(U0_theta,U1_theta),U2_theta), Evolution), output)
print(output)

##==================calculate expectation operator=================
print("===========calculate expectation operator=============")
ZZ = np.kron(np.kron(Z, I), I)
init_predict=np.zeros(m)
for i in range(m):
    wav = np.dot(output, np.dot(input_result[i], initial_state))
    init_predict[i] = np.dot(np.conj(wav).T,np.dot(ZZ, wav)).real[0][0]
print(init_predict)

##==================calculate gradient=================
print("===========calculate gradient=============")
def loss(theta):
    theta = theta.reshape(3,n_qubits,depth)
    output = np.eye(2**n_qubits)
    expectation = np.zeros(m)
    for rj in range(depth):
        U0_theta = np.dot(RX0(theta[2,2,rj]), np.dot(RZ0(theta[1,2,rj]), RX0(theta[0,2,rj])))
        U1_theta = np.dot(RX1(theta[2,1,rj]), np.dot(RZ1(theta[1,1,rj]), RX1(theta[0,1,rj])))
        U2_theta = np.dot(RX2(theta[2,0,rj]), np.dot(RZ2(theta[1,0,rj]), RX2(theta[0,0,rj])))
        output = np.dot(np.dot(np.dot(np.dot(U0_theta,U1_theta),U2_theta), Evolution), output)
    for i in range(m):
        wav = np.dot(output, np.dot(input_result[i], initial_state))
        expectation[i] = np.dot(np.conj(wav).T,np.dot(ZZ, wav)).real[0][0]
    loss_function = sum((expectation-Ydata)**2)
    return loss_function

##==================optimize theta=================
print("===========optimize theta=============")
objective = lambda parameters: loss(parameters)
opt_params = minimize(objective, initial_theta, method='l-bfgs-b',     \
                            tol=1e-12, options={'disp':True,'eps':1.0e-6,'maxiter': 10000})
best_theta = opt_params['x']
print(best_theta)

##==================prediction=================
print("===========prediction=============")
best_theta = best_theta.reshape(3,n_qubits,depth)
best_predict = np.zeros(m)
best_output = np.eye(2**n_qubits)
for rj in range(depth):
    U0_theta = np.dot(RX0(best_theta[2,2,rj]), np.dot(RZ0(best_theta[1,2,rj]), RX0(best_theta[0,2,rj])))
    U1_theta = np.dot(RX1(best_theta[2,1,rj]), np.dot(RZ1(best_theta[1,1,rj]), RX1(best_theta[0,1,rj])))
    U2_theta = np.dot(RX2(best_theta[2,0,rj]), np.dot(RZ2(best_theta[1,0,rj]), RX2(best_theta[0,0,rj])))
    best_output = np.dot(np.dot(np.dot(np.dot(U0_theta,U1_theta),U2_theta), Evolution), best_output)
for i in range(m):
    wav = np.dot(best_output, np.dot(input_result[i], initial_state))
    best_predict[i] = np.dot(np.conj(wav).T,np.dot(ZZ, wav)).real[0][0]

plt.figure()
plt.plot(Xdata,  init_predict, 'g-.')
plt.plot(Xdata, Ydata, 'bs')
plt.plot(Xdata, best_predict, 'r-')
plt.legend(("init_predict", "target","predict"), loc='best')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()


