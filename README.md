# KalmanFilter
A direct implementation of the Kalman Filter and Kalman Predictor.

To install all needed packages:

 `  pip3 install -r requirements.txt ` 
 

## Example:
 
 Esitmates the state (X1, X2) dynamics from the output Y, with Gaussian noise on both.
 
 ![plot](https://github.com/0xbadc0ffe/KalmanFilter/blob/main/fig_1.png)
 
 Estimates (X1s, X2s) compared to the real state (X1, X2)
 
 ![plot](https://github.com/0xbadc0ffe/KalmanFilter/blob/main/fig_2.png)
 
 Predictions (X1p, X2p) compared to the real state (X1, X2). Predictions are avaible a step before the actual state.
 
 ![plot](https://github.com/0xbadc0ffe/KalmanFilter/blob/main/fig_3.png)

## To Implement

`Extended Kalman Filter`
