import zmq
import numpy as np
import control

# ZMQ
context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")

print("Controller running...")

# LQR
A = np.array([
    [0, 1],
    [9.81, 0]
])

B = np.array([
    [0],
    [1]
])

Q = np.diag([10, 1])
R = np.array([[1]])

K, _, _ = control.lqr(A, B, Q, R)
K = np.asarray(K)

while True:

    message = socket.recv_json()

    theta = message["theta"]
    theta_dot = message["theta_dot"]

    x = np.array([
        [theta],
        [theta_dot]
    ])

    torque = -(K @ x).item()

    socket.send_json({
        "torque": torque
    })