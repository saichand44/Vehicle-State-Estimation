# Starter code for the Coursera SDC Course 2 final project.
#
# Author: Trevor Ablett and Jonathan Kelly
# University of Toronto Institute for Aerospace Studies
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from rotations import angle_normalize, rpy_jacobian_axis_angle, skew_symmetric, Quaternion, Q_del_theta

#### 1. Data ###################################################################################

################################################################################################
# This is where you will load the data from the pickle files. For parts 1 and 2, you will use
# p1_data.pkl. For Part 3, you will use pt3_data.pkl.
################################################################################################
with open('data/pt1_data.pkl', 'rb') as file:
    data = pickle.load(file)

################################################################################################
# Each element of the data dictionary is stored as an item from the data dictionary, which we
# will store in local variables, described by the following:
#   gt: Data object containing ground truth. with the following fields:
#     a: Acceleration of the vehicle, in the inertial frame
#     v: Velocity of the vehicle, in the inertial frame
#     p: Position of the vehicle, in the inertial frame
#     alpha: Rotational acceleration of the vehicle, in the inertial frame
#     w: Rotational velocity of the vehicle, in the inertial frame
#     r: Rotational position of the vehicle, in Euler (XYZ) angles in the inertial frame
#     _t: Timestamp in ms.
#   imu_f: StampedData object with the imu specific force data (given in vehicle frame).
#     data: The actual data
#     t: Timestamps in ms.
#   imu_w: StampedData object with the imu rotational velocity (given in the vehicle frame).
#     data: The actual data
#     t: Timestamps in ms.
#   gnss: StampedData object with the GNSS data.
#     data: The actual data
#     t: Timestamps in ms.
#   lidar: StampedData object with the LIDAR data (positions only).
#     data: The actual data
#     t: Timestamps in ms.
################################################################################################
gt = data['gt']
imu_f = data['imu_f']
imu_w = data['imu_w']
gnss = data['gnss']
lidar = data['lidar']

# print(f'imu_f data: \n{imu_f.data}')
# print(f'imu_w data: \n{imu_w.data}')
# print(f'GNSS data: \n{gnss.data}')
# print(f'GNSS time: \n{gnss.t}')
# print(f'LIDAR data: \n{lidar.data}')
# print(f'LIDAR time: \n{lidar.t}')
# print(f'gt.p data: \n{gt.p}')

################################################################################################
# Let's plot the ground truth trajectory to see what it looks like. When you're testing your
# code later, feel free to comment this out.
################################################################################################
# gt_fig = plt.figure()
# ax = gt_fig.add_subplot(111, projection='3d')
# ax.plot(gt.p[:,0], gt.p[:,1], gt.p[:,2])
# ax.set_xlabel('x [m]')
# ax.set_ylabel('y [m]')
# ax.set_zlabel('z [m]')
# ax.set_title('Ground Truth trajectory')
# ax.set_zlim(-1, 5)
# plt.show()

################################################################################################
# Remember that our LIDAR data is actually just a set of positions estimated from a separate
# scan-matching system, so we can insert it into our solver as another position measurement,
# just as we do for GNSS. However, the LIDAR frame is not the same as the frame shared by the
# IMU and the GNSS. To remedy this, we transform the LIDAR data to the IMU frame using our 
# known extrinsic calibration rotation matrix C_li and translation vector t_i_li.
#
# THIS IS THE CODE YOU WILL MODIFY FOR PART 2 OF THE ASSIGNMENT.
################################################################################################
# Correct calibration rotation matrix, corresponding to Euler RPY angles (0.05, 0.05, 0.1).
C_li = np.array([
   [ 0.99376, -0.09722,  0.05466],
   [ 0.09971,  0.99401, -0.04475],
   [-0.04998,  0.04992,  0.9975 ]
])

# Incorrect calibration rotation matrix, corresponding to Euler RPY angles (0.05, 0.05, 0.05).
# C_li = np.array([
#      [ 0.9975 , -0.04742,  0.05235],
#      [ 0.04992,  0.99763, -0.04742],
#      [-0.04998,  0.04992,  0.9975 ]
# ])

t_i_li = np.array([0.5, 0.1, 0.5])

# Transform from the LIDAR frame to the vehicle (IMU) frame.
lidar.data = (C_li @ lidar.data.T).T + t_i_li

#### 2. Constants ##############################################################################

################################################################################################
# Now that our data is set up, we can start getting things ready for our solver. One of the
# most important aspects of a filter is setting the estimated sensor variances correctly.
# We set the values here.
################################################################################################
var_imu_f = 0.10 * 100
var_imu_f_bias = 0.10 * 10
var_imu_w = 0.10 * 100
var_imu_w_bias = 0.10 * 10
var_gnss  = 0.10 * 10
var_lidar = 2.00 * 10

################################################################################################
# We can also set up some constants that won't change for any iteration of our solver.
################################################################################################
g = np.array([0, 0, -9.81])  # gravity

Fi_jac = np.zeros([15, 12])
Fi_jac[3:, :] = np.eye(12)  # motion model noise jacobian

H_x = np.zeros((3, 16))
H_x[:, :3] = np.eye(3) # measurement model jacobian (only predicts the position measurement)

X_dq = np.zeros((16, 15))
X_dq[:6, :6] = np.eye(6)
X_dq[10:, 9:] = np.eye(6)

#### 3. Initial Values #########################################################################

################################################################################################
# Let's set up some initial values for our ES-EKF solver.
################################################################################################
p_est = np.zeros([imu_f.data.shape[0], 3])  # position estimates
v_est = np.zeros([imu_f.data.shape[0], 3])  # velocity estimates
q_est = np.zeros([imu_f.data.shape[0], 4])  # orientation estimates as quaternions
ab_est = np.zeros([imu_f.data.shape[0], 3])  # acc. bias estimates of imu acceleration
wb_est = np.zeros([imu_f.data.shape[0], 3])  # ang. vel. bias estimates of imu angular vel.

p_cov = np.zeros([imu_f.data.shape[0], 15, 15])  # covariance matrices at each timestep

# Set initial values.
p_est[0] = gt.p[0]
v_est[0] = gt.v[0]
q_est[0] = Quaternion(euler=gt.r[0]).to_numpy()
ab_est[0] = np.array([0.0, 0.0, 0.0]).reshape(1, 3)
wb_est[0] = np.array([0.0, 0.0, 0.0]).reshape(1, 3)

p_cov[0] = np.zeros(15)  # covariance of estimate

gnss_i  = 0
lidar_i = 0
gnss_t = list(gnss.t)
lidar_t = list(lidar.t)

#### 4. Measurement Update #####################################################################

################################################################################################
# Since we'll need a measurement update for both the GNSS and the LIDAR data, let's make
# a function for it.
################################################################################################
def measurement_update(sensor_var, p_cov_check, y_k, p_check, v_check, q_check, ab_check, wb_check, H_jac):
    # 3.1 Compute Kalman Gain
    r_cov = np.eye(3)*sensor_var
    k_gain = p_cov_check @ H_jac.T @ np.linalg.inv((H_jac @ p_cov_check @ H_jac.T) + r_cov)

    # 3.2 Compute error state
    error_state = k_gain @ (y_k - p_check)

    # 3.3 Correct predicted state
    p_hat = p_check + error_state[0:3]
    v_hat = v_check + error_state[3:6]
    # q_hat = Quaternion(axis_angle=error_state[6:9]).quat_mult_left(Quaternion(*q_check))
    q_hat = Quaternion(*q_check).quat_mult_left(Quaternion(axis_angle=error_state[6:9]))
    ab_hat = ab_check + error_state[9:12]
    wb_hat = wb_check + error_state[12:15]

    # 3.4 Compute corrected covariance
    p_cov_hat = (np.eye(15) - k_gain @ H_jac) @ p_cov_check

    return p_hat, v_hat, q_hat, ab_hat, wb_hat, p_cov_hat

#### 5. Main Filter Loop #######################################################################

################################################################################################
# Now that everything is set up, we can start taking in the sensor data and creating estimates
# for our state in a loop.
################################################################################################
for k in range(1, imu_f.data.shape[0]):  # start at 1 b/c we have initial prediction from gt
    delta_t = imu_f.t[k] - imu_f.t[k - 1]

    # 1. Update state with IMU inputs
    q_prev = Quaternion(*q_est[k - 1, :]) # previous orientation as a quaternion object
    ab_prev = ab_est[k - 1, :] # previous acc. bias
    wb_prev = wb_est[k - 1, :] # previous angular vel. bias

    q_curr = Quaternion(axis_angle=((imu_w.data[k - 1] - wb_prev)*delta_t)) # current IMU orientation
    c_ns = q_prev.to_mat() # previous orientation as a matrix
    f_ns = (c_ns @ (imu_f.data[k - 1] - ab_prev)) + g # calculate sum of forces
    p_check = p_est[k - 1, :] + delta_t*v_est[k - 1, :] + 0.5*(delta_t**2)*f_ns
    v_check = v_est[k - 1, :] + delta_t*f_ns
    q_check = q_prev.quat_mult_left(q_curr)  # q_prev * q_curr
    ab_check = ab_prev
    wb_check = wb_prev

    # 1.1 Linearize the motion model and compute Jacobians
    f_jac = np.eye(15) # motion model jacobian with respect to last state
    f_jac[0:3, 3:6] = np.eye(3)*delta_t
    f_jac[3:6, 6:9] = - c_ns @ skew_symmetric(imu_f.data[k - 1] - ab_prev)*delta_t
    f_jac[3:6, 9:12] = - c_ns*delta_t
    f_jac[6:9, 6:9] = np.eye(3) - skew_symmetric(imu_w.data[k - 1] - wb_prev)*delta_t
    f_jac[6:9, 12:15] = - np.eye(3)*delta_t

    # 2. Propagate uncertainty
    q_cov = np.zeros((12, 12)) # IMU noise covariance
    q_cov[0:3, 0:3] = delta_t**2 * np.eye(3)*var_imu_f
    q_cov[3:6, 3:6] = delta_t**2 * np.eye(3)*var_imu_w
    q_cov[6:9, 6:9] = delta_t * np.eye(3)*var_imu_f_bias
    q_cov[9:, 9:] = delta_t * np.eye(3)*var_imu_w_bias

    p_cov_check = f_jac @ p_cov[k - 1, :, :] @ f_jac.T + Fi_jac @ q_cov @ Fi_jac.T

    # 3. Check availability of GNSS and LIDAR measurements
    if imu_f.t[k] in gnss_t:
        gnss_i = gnss_t.index(imu_f.t[k])

        # 3.1 compute the observation jacobian
        X_dq[6:10, 6:9] = Q_del_theta(Quaternion(*q_check))
        H_jac = H_x @ X_dq

        p_check, v_check, q_check, ab_check, wb_check, p_cov_check = \
            measurement_update(var_gnss, p_cov_check, gnss.data[gnss_i], 
                               p_check, v_check, q_check, ab_check, wb_check, H_jac)
    
    if imu_f.t[k] in lidar_t:
        lidar_i = lidar_t.index(imu_f.t[k])

        # 3.1 compute the observation jacobian
        X_dq[6:10, 6:9] = Q_del_theta(Quaternion(*q_check))
        H_jac = H_x @ X_dq

        p_check, v_check, q_check, ab_check, wb_check, p_cov_check = \
            measurement_update(var_lidar, p_cov_check, lidar.data[lidar_i], 
                               p_check, v_check, q_check, ab_check, wb_check, H_jac)

    # Update states (save)
    p_est[k, :] = p_check
    v_est[k, :] = v_check
    q_est[k, :] = q_check
    ab_est[k, :] = ab_check
    wb_est[k, :] = wb_check
    p_cov[k, :, :] = p_cov_check

#### 6. Results and Analysis ###################################################################

################################################################################################
# Now that we have state estimates for all of our sensor data, let's plot the results. This plot
# will show the ground truth and the estimated trajectories on the same plot. Notice that the
# estimated trajectory continues past the ground truth. This is because we will be evaluating
# your estimated poses from the part of the trajectory where you don't have ground truth!
################################################################################################
est_traj_fig = plt.figure()
ax = est_traj_fig.add_subplot(111, projection='3d')
ax.plot(p_est[:,0], p_est[:,1], p_est[:,2], label='Estimated')
ax.plot(gt.p[:,0], gt.p[:,1], gt.p[:,2], label='Ground Truth')
ax.set_xlabel('Easting [m]')
ax.set_ylabel('Northing [m]')
ax.set_zlabel('Up [m]')
ax.set_title('Ground Truth and Estimated Trajectory')
ax.set_xlim(0, 200)
ax.set_ylim(0, 200)
ax.set_zlim(-2, 2)
ax.set_xticks([0, 50, 100, 150, 200])
ax.set_yticks([0, 50, 100, 150, 200])
ax.set_zticks([-2, -1, 0, 1, 2])
ax.legend(loc=(0.62,0.77))
ax.view_init(elev=45, azim=-50)
plt.show()

################################################################################################
# We can also plot the error for each of the 6 DOF, with estimates for our uncertainty
# included. The error estimates are in blue, and the uncertainty bounds are red and dashed.
# The uncertainty bounds are +/- 3 standard deviations based on our uncertainty (covariance).
################################################################################################
error_fig, ax = plt.subplots(2, 3)
error_fig.suptitle('Error Plots')
num_gt = gt.p.shape[0]
p_est_euler = []
p_cov_euler_std = []

# Convert estimated quaternions to euler angles
for i in range(len(q_est)):
    qc = Quaternion(*q_est[i, :])
    p_est_euler.append(qc.to_euler())

    # First-order approximation of RPY covariance
    J = rpy_jacobian_axis_angle(qc.to_axis_angle())
    p_cov_euler_std.append(np.sqrt(np.diagonal(J @ p_cov[i, 6:9, 6:9] @ J.T)))

p_est_euler = np.array(p_est_euler)
p_cov_euler_std = np.array(p_cov_euler_std)

# Get uncertainty estimates from P matrix
p_cov_std = np.sqrt(np.diagonal(p_cov[:, :3, :3], axis1=1, axis2=2))

titles = ['Easting', 'Northing', 'Up', 'Roll', 'Pitch', 'Yaw']
for i in range(3):
    ax[0, i].plot(range(num_gt), gt.p[:, i] - p_est[:num_gt, i])
    ax[0, i].plot(range(num_gt),  3 * p_cov_std[:num_gt, i], 'r--')
    ax[0, i].plot(range(num_gt), -3 * p_cov_std[:num_gt, i], 'r--')
    ax[0, i].set_title(titles[i])
ax[0,0].set_ylabel('Meters')

for i in range(3):
    ax[1, i].plot(range(num_gt), \
        angle_normalize(gt.r[:, i] - p_est_euler[:num_gt, i]))
    ax[1, i].plot(range(num_gt),  3 * p_cov_euler_std[:num_gt, i], 'r--')
    ax[1, i].plot(range(num_gt), -3 * p_cov_euler_std[:num_gt, i], 'r--')
    ax[1, i].set_title(titles[i+3])
ax[1,0].set_ylabel('Radians')
plt.show()