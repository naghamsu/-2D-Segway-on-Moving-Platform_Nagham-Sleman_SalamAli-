import mujoco
import mujoco.viewer
import numpy as np
import control
import zmq
import threading
import time

# ============================================
# CONTROLLER THREAD
# ============================================

def controller_thread():

    # ZMQ SERVER
    context = zmq.Context()

    socket = context.socket(zmq.REP)

    socket.bind("tcp://*:5555")

    print("Controller started...")

    # ========================================
    # LQR SETUP
    # ========================================

    A = np.array([
        [0, 1],
        [9.81, 0]
    ])

    B = np.array([
        [0],
        [1]
    ])

    # weak controller
    # -> larger visible oscillations
    Q = np.diag([0.5, 0.05])

    R = np.array([[1]])

    K, _, _ = control.lqr(A, B, Q, R)

    K = np.asarray(K)

    print("LQR Gain:")
    print(K)

    # ========================================
    # CONTROL LOOP
    # ========================================

    while True:

        # receive state
        msg = socket.recv_json()

        theta = msg["theta"]

        theta_dot = msg["theta_dot"]

        x = np.array([
            [theta],
            [theta_dot]
        ])

        # LQR torque
        torque = -(K @ x).item()

        # disturbance for visible motion
        torque += 25 * np.sin(6 * time.time())

        # send torque
        socket.send_json({
            "torque": torque
        })

# ============================================
# START CONTROLLER THREAD
# ============================================

thread = threading.Thread(target=controller_thread)

thread.daemon = True

thread.start()

# ============================================
# ZMQ CLIENT
# ============================================

context = zmq.Context()

socket = context.socket(zmq.REQ)

socket.connect("tcp://localhost:5555")

# ============================================
# LOAD MUJOCO MODEL
# ============================================

model = mujoco.MjModel.from_xml_path(
    r"C:\Users\Nagham\Desktop\robot prog\segway2d.xml"
)

data = mujoco.MjData(model)

# ============================================
# INITIAL TILT
# ============================================

data.qpos[1] = 0.3

# ============================================
# VIEWER
# ============================================

viewer = mujoco.viewer.launch_passive(model, data)

viewer.cam.distance = 12

viewer.cam.azimuth = 90

viewer.cam.elevation = -10

viewer.cam.lookat[:] = [0, 0, 1]

# ============================================
# SIMULATION LOOP
# ============================================

t = 0

while viewer.is_running():

    # ========================================
    # LIMITED PLATFORM VELOCITY
    # ========================================

    platform_velocity = np.clip(
         5 * np.cos(2 * t),
        -5.0,
        5.0
    )

    # apply limited velocity
    data.qvel[0] = platform_velocity

    # ========================================
    # STATES
    # ========================================

    theta = float(data.qpos[1])

    theta_dot = float(data.qvel[1])

    # ========================================
    # SEND STATE TO CONTROLLER
    # ========================================

    socket.send_json({
        "theta": theta,
        "theta_dot": theta_dot
    })

    # ========================================
    # RECEIVE TORQUE
    # ========================================

    response = socket.recv_json()

    torque = response["torque"]

    # ========================================
    # APPLY CONTROL
    # ========================================

    data.ctrl[0] = torque

    # ========================================
    # STEP SIMULATION
    # ========================================

    mujoco.mj_step(model, data)

    viewer.sync()

    time.sleep(0.01)

    t += 0.01