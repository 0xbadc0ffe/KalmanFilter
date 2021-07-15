import numpy as np
import os
import platform
import matplotlib.pyplot as plt



if platform.system() == 'Windows':
    CLEAR_STR = "cls" 
else:
    CLEAR_STR = "clear"


class KalmanFilter:

    # xs(k+1) = A(k)xs(k) + F(k)*N(k)
    # ys(k)   = C(k)xs(k) + G(k)*N(k)
    
    def __init__(self, P0, xsp0=np.array([0]), xs0=np.array([0]), k=0, Gain=None, Delp=None):
        # Step 1 -  Initialization
        self.xsp = xsp0             # Optimal prediction xs(k| k-1)  xs(k| k-1) = A(k-1)*xs(k-1)
        self.xs = xs0               # Optimal estimate xs(k) = A(k-1)*xs(k-1) + Pi_k(k)[Ys(k) - C(k)*A(k-1)*xs(k-1)]
                                    # initialization => xs(0|-1) = xs(0) = 0 mandatory for optimality                             
        self.P = P0                 # Error covariance P(k) = [I - Pi_k(k)*C(k)]*Delp(k) => P(0) = covariance of xs(0)
        self.k = k                  # Step k
        
        self.Gain = Gain            # Gain Matrix Pi_k(k) = Delp(k)*C(k)^T*[C(k)*Delp(k)*C(k)^T +G(k)*G(k)^T]^-1
        self.Delp = Delp            # Error prediction covariance Delp(k) = A(k-1)*P(k-1)*A(k-1)^T + F(k-1)*F^T(k-1)
                                    # Delp(k) = P(k| k-1)
        
    def step(self, ysk, A, F, Ck, Gk):    # Ys(k), A(k-1), F(k-1), C(k), G(k)
    
        # Step 2 - Error Prediction Covariance Delp(k+1)
        self.Delp = A @ self.P @ A.T + F @ F.T
        
        # Step 3 - Gain Matrix
        Cov_out_inn =  Ck @ self.Delp @ Ck.T + Gk @ Gk.T
        self.Gain = self.Delp @ Ck.T @ Cov_out_inn.I
        
        # Step 4 - Error Covariance
        self.P = (np.eye(self.xs.shape[0]) - self.Gain @ Ck) @ self.Delp
        
        # Step 5 - Optimal Prediction
        self.xsp = A @ self.xs
        
        # Step 6 - Optimal Estimate
        self.xs = self.xsp + self.Gain @ (ysk - Ck @ self.xsp)
        
        # Step 7 - k -> k+1
        self.k += 1
        
        return self.xs, self.xsp, self.P, self.k
    
    #def step_predictor  # Kalman Predictor
    #def step_extended   # Extended Kalman Filter
    
    
class System:    # Example System to follow with KF 
    
    # xs(k+1) = A(k)xs(k) + F(k)*N(k)
    # ys(k)   = C(k)xs(k) + G(k)*N(k)

    def __init__(self, x0=np.matrix('0'), k=0, err_dim=1, sigma_d=0.1, mu_d=0):
        self.x = x0
        self.sigma_d = sigma_d
        self.mu_d = mu_d
        self.err_dim = err_dim
        self.y = None
        self.k = k

        
    def step(self, A, F, C, G):  # A(k), F(k), C(k), G(k)
        K = F @ np.random.normal(self.mu_d, self.sigma_d, self.err_dim)    
        self.x = A @ self.x + K.T   # For some reason F@random.. changes from 2x1 matrix to 1x2, maybe K is considered a vector
        R = G @ np.random.normal(self.mu_d, self.sigma_d, self.err_dim)
        self.y = C @ self.x + R.T
        self.k += 1
        
        return self.x, self.y, self.k, K.T, R.T
        
        

if __name__ == "__main__":

    os.system(CLEAR_STR)
    
    xs0 = np.matrix('0;0')
    #P0  = np.matrix('1 2; 3 4')
    #P0 = np.array([0])
    P0 = np.matrix('0 0; 0 0')
    
    A = np.matrix('-0.6 -0.8; 0.8 -0.6')
    F = np.matrix('0.5; 1')
    C = np.matrix('-1 1')
    G = np.matrix('1')
    
    KF  = KalmanFilter(P0,xs0=xs0,xsp0=xs0)
    Sys = System(xs0,sigma_d=1)
    
    output = []
    track_x1 = []
    track_x1s = []
    track_x2 = []
    track_x2s = []
    track_err1 = []
    track_err2 = []
    noise_x1 = []
    noise_x2 = []
    noise_y = []
    
    SIM_TIME = 50
    steps = range(1,SIM_TIME+1)
    
    
    for i in steps:
        print(f"\n\n################# STEP {i})")
        x, y, k , rx, ry = Sys.step(A,F,C,G)
        
        track_x1.append(float(x[0]))
        track_x2.append(float(x[1]))
        output.append(float(y))
        
        noise_x1.append(float(rx[0]))
        noise_x2.append(float(rx[1]))
        noise_y.append(float(ry))
        
        print(f"\nState X({k}):          \n{x}\n\nOutput Y({k}):         \n{y}")
        
        
        xs, xsp, P, k = KF.step(y,A,F,C,G)
        track_x1s.append(float(xs[0]))
        track_x2s.append(float(xs[1]))
        print(f"\n\n\nEstimate X({k}):       \n{xs}\n\nPrediction X({k}|{k-1}):     \n{xsp}")
        print(f"\n\n\nERR: \n{x-xs}")
        track_err1.append(float(x[0]-xs[0]))
        track_err2.append(float(x[1]-xs[1]))
        print(f"\n\nREL ERR: \n{(x-xs)/x}")
        
    print()
    input("Press Enter to plot results")
    
    anoised_x1 = []
    anoised_x2 = []
    anoised_y  = []
    for i in steps:
        anoised_x1.append(track_x1[i-1]-noise_x1[i-1])
        anoised_x2.append(track_x2[i-1]-noise_x2[i-1])
        anoised_y.append(output[i-1]-noise_y[i-1])
        
    
    fig, axs = plt.subplots(2)
    fig.suptitle('State Tracking')
    axs[0].set_title("X1, X1s")
    axs[0].plot(steps,track_x1, label="X1")
    axs[0].plot(steps,track_x1s, label="X1s")
    axs[0].legend(loc="upper right")
    axs[1].set_title("X2, X2s")
    axs[1].plot(steps,track_x2, label="X2")
    axs[1].plot(steps,track_x2s, label="X2s")
    axs[1].legend(loc="upper right")
    
    
    fig2, axs2 = plt.subplots(2)
    fig2.suptitle('Tracking Errors')
    axs2[0].set_title("E1 = X1 - X1s")
    axs2[0].plot(steps,track_err1)
    axs2[1].set_title("E2 = X2 - X2s")
    axs2[1].plot(steps,track_err2)
    
    
    fig3, axs3 = plt.subplots(3)
    fig3.suptitle('Noise')
    axs3[0].set_title("X1 Noise")
    axs3[0].plot(steps,noise_x1, "-r", label="noise")
    axs3[0].plot(steps,track_x1, "-y", label="X1")
    axs3[0].legend(loc="upper right")
    axs3[1].set_title("X2 Noise")
    axs3[1].plot(steps,noise_x2, "-r", label="noise")
    axs3[1].plot(steps,track_x2, "-y", label="X2")
    axs3[1].legend(loc="upper right")
    axs3[2].set_title("Y Noise")
    axs3[2].plot(steps,noise_y, "-r",label="noise")
    axs3[2].plot(steps,output, "-y", label="Y")
    axs3[2].legend(loc="upper right")
    
    
    fig4, axs4 = plt.subplots(2)
    fig4.suptitle('Output Y')
    axs4[0].plot(steps,output, label="Y")
    axs4[0].legend(loc="upper right")
    axs4[1].plot(steps,anoised_y, label="Y - noise")
    axs4[1].legend(loc="upper right")
    
    
    fig5, axs5 = plt.subplots(2)
    fig5.suptitle('State Tracking - No Noise')
    axs5[0].set_title("X1-noise, X1s")
    axs5[0].plot(steps,anoised_x1, label="X1 - noise ")
    axs5[0].plot(steps,track_x1s, label="X1s")
    axs5[0].legend(loc="upper right")
    axs5[1].set_title("X2 - noise, X2s")
    axs5[1].plot(steps,anoised_x2, label="X2 -noise")
    axs5[1].plot(steps,track_x2s, label="X2s")
    axs5[1].legend(loc="upper right")
    
    
    plt.show()
    
    
    


        
        
        
        