# Fill in the respective function to implement the LQR/EKF SLAM controller

# Import libraries
import numpy as np
from base_controller import BaseController
from scipy import signal, linalg
from scipy.spatial.transform import Rotation
from util import *
from ekf_slam import EKF_SLAM

from scipy.signal import cont2discrete

# CustomController class (inherits from BaseController)
class CustomController(BaseController):

    def __init__(self, trajectory):

        super().__init__(trajectory)

        # Define constants

        self.lr = 0.82
        self.lf = 1.18
        self.Ca = 20000
        self.Iz = 3004.5
        self.m = 1000
        self.g = 9.81

        self.e1 = 0 # error1: distance error
        self.e2 = 0 # error2: yawing angle (heading angle) error
        self.e1dot = 0 # derivative of error 1
        self.e2dot = 0 # derivative of error 2
        self.t = 0 # postion
        
        self.counter = 0
        np.random.seed(99)

        # Add additional member variables according to your need here.

    def getStates(self, timestep, use_slam=False):

        delT, X, Y, xdot, ydot, psi, psidot = super().getStates(timestep)

        # Initialize the EKF SLAM estimation
        if self.counter == 0:
            # Load the map
            minX, maxX, minY, maxY = -120., 450., -500., 50.
            map_x = np.linspace(minX, maxX, 7)
            map_y = np.linspace(minY, maxY, 7)
            map_X, map_Y = np.meshgrid(map_x, map_y)
            map_X = map_X.reshape(-1,1)
            map_Y = map_Y.reshape(-1,1)
            self.map = np.hstack((map_X, map_Y)).reshape((-1))
            
            # Parameters for EKF SLAM
            self.n = int(len(self.map)/2)             
            X_est = X + 0.5
            Y_est = Y - 0.5
            psi_est = psi - 0.02
            mu_est = np.zeros(3+2*self.n)
            mu_est[0:3] = np.array([X_est, Y_est, psi_est])
            mu_est[3:] = np.array(self.map)
            init_P = 1*np.eye(3+2*self.n)
            W = np.zeros((3+2*self.n, 3+2*self.n))
            W[0:3, 0:3] = delT**2 * 0.1 * np.eye(3)
            V = 0.1*np.eye(2*self.n)
            V[self.n:, self.n:] = 0.01*np.eye(self.n)
            # V[self.n:] = 0.01
            print(V)
            
            # Create a SLAM
            self.slam = EKF_SLAM(mu_est, init_P, delT, W, V, self.n)
            self.counter += 1
        else:
            mu = np.zeros(3+2*self.n)
            mu[0:3] = np.array([X, 
                                Y, 
                                psi])
            mu[3:] = self.map
            y = self._compute_measurements(X, Y, psi)
            mu_est, _ = self.slam.predict_and_correct(y, self.previous_u)

        self.previous_u = np.array([xdot, ydot, psidot])

        print("True      X, Y, psi:", X, Y, psi)
        print("Estimated X, Y, psi:", mu_est[0], mu_est[1], mu_est[2])
        print("-------------------------------------------------------")
        
        if use_slam == True:
            return delT, mu_est[0], mu_est[1], xdot, ydot, mu_est[2], psidot
        else:
            return delT, X, Y, xdot, ydot, psi, psidot

    def _compute_measurements(self, X, Y, psi):
        x = np.zeros(3+2*self.n)
        x[0:3] = np.array([X, Y, psi])
        x[3:] = self.map
        
        p = x[0:2]
        psi = x[2]
        m = x[3:].reshape((-1,2))

        y = np.zeros(2*self.n)

        for i in range(self.n):
            y[i] = np.linalg.norm(m[i, :] - p)
            y[self.n+i] = wrapToPi(np.arctan2(m[i,1]-p[1], m[i,0]-p[0]) - psi)
            
        y = y + np.random.multivariate_normal(np.zeros(2*self.n), self.slam.V)
        # print(np.random.multivariate_normal(np.zeros(2*self.n), self.slam.V))
        return y

    def update(self, timestep):

        trajectory = self.trajectory

        lr = self.lr
        lf = self.lf
        Ca = self.Ca
        Iz = self.Iz
        m = self.m
        g = self.g
        self.t += 1

        # Fetch the states from the newly defined getStates method
        delT, X, Y, xdot, ydot, psi, psidot = self.getStates(timestep, use_slam=True)
        # You must not use true_X, true_Y and true_psi since they are for plotting purpose
        _, true_X, true_Y, _, _, true_psi, _ = self.getStates(timestep, use_slam=False)

        # You are free to reuse or refine your code from P3 in the spaces below.
# Variables:
        disError, nearIdx = closestNode(X, Y, trajectory) # distance error & nearest waypoint's index
        future_steps = 80 # fixed point of time (time step) in the future (a.k.a time horizon)
        idx_angle = nearIdx + future_steps
        if idx_angle >= 8203:
            idx_angle = 8202
        X_target = (trajectory[idx_angle][0]) # target X position
        Y_target = (trajectory[idx_angle][1]) # target y position
        num = Y_target - Y # numerator
        den = X_target - X # denominator
        psi_target = np.arctan2(num, den) # target vehicle yawing angle
        
        future_steps4vel = 400
        idx4vel_angle = nearIdx + future_steps4vel
        if idx4vel_angle >= 8203:
            idx4vel_angle = 8202
        X_target = (trajectory[idx4vel_angle][0]) # target X position
        Y_target = (trajectory[idx4vel_angle][1]) # target y position
        num = Y_target - Y # numerator
        den = X_target - X # denominator
        psi_target4vel = np.arctan2(num, den) # target vehicle yawing angle
        

        # target yawing (angular) velocity:
        #  target yawing velocity = 0 b/c it is time-varying distrubance in the state space equation and is safely assumed to be 0.
        psidot_target = 0 # target yawing (angular) velocity

        # Instructions
        # You will have to implement two controllers: LQR and MPC for this assignment.
        # You can either write both controllers in the same file and use a switch statement.
        # Alternatively, you can duplicate this file and write the controllers in two files.

        # ---------------|Lateral Controller|-------------------------

        ## lateral vehicle dynamics error states: #############################################################
        self.e2 = wrapToPi(psi - psi_target) # error 2: Yawing angle error with respect to road
        self.e1dot = ydot + xdot*(wrapToPi(psi - psi_target)) # derivative of error 1
        self.e1 = self.e1dot * delT + self.e1 # error 1: lateral position error with respect to road
        self.e2dot = psidot - psidot_target # derivative of error 2
        e_states = np.array([[self.e1], [self.e1dot], [self.e2], [self.e2dot]]) # error states
        
        ## Continous Time: ####################################################################################
        # A matrix
        A = np.array(
            [[0,1,0,0], 
            [0,-4*Ca/(m*xdot),4*Ca/m,-2*Ca*(lf-lr)/(m*xdot)], 
            [0,0,0,1], 
            [0, -2*Ca*(lf-lr)/(Iz*xdot), 2*Ca*(lf-lr)/Iz, -2*Ca*(lf**2+lr**2)/(Iz*xdot)]]
        )
        
        # B matrix
        B = np.array([[0],[2*Ca/m],[0],[2*Ca*lf/Iz]])

        # C matrix - We want to output e1, e1dot, e2, e2dot
        C = np.eye(4) 

        # D matrix
        D = 0

        ## Discrete Time: ######################################################################################
        dt = delT # time interval
        d_sys = cont2discrete((A,B,C,D), dt, method='zoh') # discretized dynamic system
        A = d_sys[0]
        B = d_sys[1]
        C = d_sys[2]
        D = d_sys[3]

        ## Infinite Horizon Discrete Time LQR Controller #########################################################
        R = 10
        Q = np.array([[100,0,0,0],[0,100,0,0],[0,0,0.5,0],[0,0,0,0.5]]) # state cost matrix
        
        # S = (A-B*K).T * S * (A-B*K) + Q + K.T * R * K # discrete time Algebraic Ricatti equation
        S = np.matrix(linalg.solve_discrete_are(A, B, Q, R)) # discrete time Algebraic Ricatti equation
        K = np.matrix(linalg.inv(R + B.T * S * B) * B.T * S * A) # gain matrix (at infinite horizon, K becomes a constant)
        u = np.ndarray.item(-np.dot(K, e_states)) # input u = -K*x (x is the orientation error e2)
        
        delta = u # yaw angle (heading angle) correction

     
        # ---------------|Longitudinal Controller|-------------------------
        """
        Please design your longitudinal controller below.
        - Longitudinal PID controllers will take errors of velocity.
            - error: velocity difference between the vhicle's target (desired) velocity and vehicle's current velocity
        """
 
        kp = 80 # P gain
        ki = 0.001 # I gain
        kd = 0.2 # D gain

        # Reference value for PID to tune to
        if abs(wrapToPi(psi - psi_target4vel)) > 0.2:
            if self.t * dt < 4: # starting booster
                desiredVelocity = 80
            else:
                desiredVelocity = 7 # brake (slows down)
        else:
            if self.t * dt >= 4 and self.t * dt < 6:
                desiredVelocity = 80 # starting booster
            else:
                desiredVelocity = 16 # desired velocity


        xdotError = (desiredVelocity - xdot)
        self.integralXdotError += xdotError
        derivativeXdotError = xdotError - self.previousXdotError
        self.previousXdotError = xdotError

        F = kp*xdotError + ki*self.integralXdotError*delT + kd*derivativeXdotError/delT # force [N]


        # Return all states and calculated control inputs (F, delta)
        return true_X, true_Y, xdot, ydot, true_psi, psidot, F, delta
