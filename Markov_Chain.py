{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "8f5acf0f",
   "metadata": {},
   "source": [
    "# Markov Chain\n",
    "## Calculating equilibrium states"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2896942e",
   "metadata": {},
   "source": [
    "#### Transition Matrix"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "18c3fbba",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[0.2, 0.6, 0.2],\n",
       "       [0.3, 0. , 0.7],\n",
       "       [0.5, 0. , 0.5]])"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import numpy as np\n",
    "\n",
    "#Transition Matrix\n",
    "A = np.array([[0.2, 0.6, 0.2], [0.3, 0, 0.7], [0.5, 0, 0.5]])\n",
    "\n",
    "A"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "4d7002f0",
   "metadata": {},
   "outputs": [],
   "source": [
    "state = {0:\"Burger\", 1:\"Pizza\", 2: \"Hot dog\"}"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6c0737ed",
   "metadata": {},
   "source": [
    "#### Random Walk on Markov Chain"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "8ee966d3",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Burger ---> Pizza ---> Hot dog ---> Burger ---> Burger ---> Burger ---> Burger ---> Burger ---> Pizza ---> Burger ---> Burger ---> Pizza ---> Hot dog ---> Burger ---> Pizza ---> "
     ]
    }
   ],
   "source": [
    "n = 15\n",
    "start_state = 0\n",
    "print(state[start_state], \"--->\", end = \" \")\n",
    "prev_state = start_state\n",
    "\n",
    "while n-1:\n",
    "    curr_state = np.random.choice([0,1,2], p=A[prev_state])\n",
    "    print(state[curr_state], \"--->\", end = \" \")\n",
    "    prev_state = curr_state\n",
    "    n -= 1\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "7e122bc0",
   "metadata": {},
   "source": [
    "#### Calculating Equilibrium state"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "54aae519",
   "metadata": {},
   "source": [
    "**Approach 1: Monte Carlo** : Idea is to simulate the random walk for very high number of states and count no of times each state appearing in random walk and divide it by total number of steps to get stationary probabilites"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "06e7eb8c",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Burger ---> "
     ]
    }
   ],
   "source": [
    "n = 10**6\n",
    "start_state = 0\n",
    "print(state[start_state], \"--->\", end = \" \")\n",
    "prev_state = start_state\n",
    "#counter for how many times each state appears\n",
    "pi = np.array([0, 0, 0])\n",
    "pi[start_state] = 1\n",
    "\n",
    "i = 0\n",
    "while i < n:\n",
    "    curr_state = np.random.choice([0,1,2], p=A[prev_state])\n",
    "    pi[curr_state] += 1\n",
    "#     print(state[curr_state], \"--->\", end = \" \")\n",
    "    prev_state = curr_state\n",
    "    i += 1\n",
    "    \n",
    "static_proab = pi/n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "f9c3746c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0.352308, 0.211247, 0.436446])"
      ]
     },
     "execution_count": 24,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "static_proab"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e66d5218",
   "metadata": {},
   "source": [
    "#### 2. Repeated Matrix Multiplication"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "738632fe",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "pi =  [0.35211268 0.21126761 0.43661972]\n"
     ]
    }
   ],
   "source": [
    "steps = 10**3\n",
    "A_n = A\n",
    "i = 0\n",
    "while i<steps:\n",
    "    A_n = np.matmul(A_n,A)\n",
    "    i += 1\n",
    "\n",
    "print(\"pi = \", A_n[0])"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b361e1fc",
   "metadata": {},
   "source": [
    "### 3. Finding Left Eigen Vectors"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "74e853e1",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "left eigen vector [[-0.58746336+0.j         -0.16984156-0.35355339j -0.16984156+0.35355339j]\n",
      " [-0.35247801+0.j          0.67936622+0.j          0.67936622-0.j        ]\n",
      " [-0.72845456+0.j         -0.50952467+0.35355339j -0.50952467-0.35355339j]]\n",
      "left eigen values [ 1.  +0.j        -0.15+0.3122499j -0.15-0.3122499j]\n"
     ]
    }
   ],
   "source": [
    "import scipy.linalg\n",
    "values, left = scipy.linalg.eig(A, right = False, left = True)\n",
    "print(\"left eigen vector\", left)\n",
    "print(\"left eigen values\", values)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6860c26e",
   "metadata": {},
   "source": [
    "**Here, we can see only first eigen vector (in the first column) has real values and the eigen value is also 1 for only first one**.\n",
    "* But, we need the eigen vector whose sum equates to 1. So, we normalize the values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "4006e1db",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[0.352112676056338, 0.21126760563380279, 0.43661971830985913]"
      ]
     },
     "execution_count": 31,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "pi = left[:, 0]\n",
    "pi_normalized = [(x/np.sum(pi)).real for x in pi]\n",
    "pi_normalized"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e69ec667",
   "metadata": {},
   "source": [
    "### What's the probability that the following Chain will be produced?\n",
    "\n",
    "**P(Pizza -> Hotdog -> Hotdog -> Burger) = ?**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "id": "ab2380a6",
   "metadata": {},
   "outputs": [],
   "source": [
    "def find_prob(seq,A,pi):\n",
    "    start_state = seq[0]\n",
    "    prob = pi[start_state]\n",
    "    prev_state = start_state\n",
    "    for i in range(1,len(seq)):\n",
    "        curr_state = seq[i]\n",
    "        prob *= A[prev_state][curr_state]\n",
    "        prev_state = curr_state\n",
    "    return prob       "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "id": "9c76bae5",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.036971830985915485"
      ]
     },
     "execution_count": 34,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "find_prob([1,2,2,0],A,pi_normalized)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c720db18",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9505b71b",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
