#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import csv
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from crater_msgs.msg import CraterInfo
from std_msgs.msg import Header
import math
import time

def rot(theta):
    return np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])

class ParticleFilterLocalizer(Node):
    def __init__(self):
        super().__init__('pf_localizer')
        # params
        self.Np = int(self.declare_parameter('num_particles', 500).value)
        self.sigma_pos = float(self.declare_parameter('sigma_pos', 0.4).value)  # measurement pos sigma (m)
        self.sigma_r = float(self.declare_parameter('sigma_r', 0.3).value)    # radius sigma (m)
        self.gating = float(self.declare_parameter('gating_radius', 2.0).value)
        self.map_csv = str(self.declare_parameter('map_csv', 'map_craters.csv').value)

        # load map craters
        self.map_craters = self.load_map(self.map_csv)
        self.get_logger().info(f'Loaded {len(self.map_craters)} map craters')

        # particles: Nx3 array (x, y, theta)
        self.particles = np.zeros((self.Np, 3))
        self.weights = np.ones(self.Np) / float(self.Np)
        self.init_particles_random()

        # subs/pubs
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_cb, 10)
        self.crater_sub = self.create_subscription(CraterInfo, '/crater_info', self.crater_cb, 10)
        self.pose_pub = self.create_publisher(PoseStamped, '/pf_pose', 10)

        # store last odom for delta
        self.last_odom = None
        self.get_logger().info('Particle filter initialized')

    def load_map(self, csvfile):
        craters = []
        try:
            with open(csvfile, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    craters.append({'x': float(row['x']), 'y': float(row['y']), 'radius': float(row['radius'])})
        except Exception as e:
            self.get_logger().warn(f'Could not open map CSV {csvfile}: {e}. Using empty map.')
        return craters

    def init_particles_random(self):
        # initialize near origin (tune as needed)
        self.particles[:,0] = np.random.uniform(-5.0, 5.0, self.Np)
        self.particles[:,1] = np.random.uniform(-5.0, 5.0, self.Np)
        self.particles[:,2] = np.random.uniform(-math.pi, math.pi, self.Np)
        self.weights[:] = 1.0 / self.Np

    def odom_cb(self, msg: Odometry):
        # simple motion update using odom delta pose
        pose = msg.pose.pose
        x = pose.position.x
        y = pose.position.y
        # extract yaw
        q = pose.orientation
        yaw = 2.0 * math.atan2(q.z, q.w)
        if self.last_odom is None:
            self.last_odom = (x, y, yaw)
            return
        dx = x - self.last_odom[0]
        dy = y - self.last_odom[1]
        dyaw = self.wrap_angle(yaw - self.last_odom[2])
        self.last_odom = (x, y, yaw)
        # simple additive motion model with noise
        for i in range(self.Np):
            # transform delta to particle frame approx by adding with noise
            nx = dx + np.random.normal(0, 0.02)
            ny = dy + np.random.normal(0, 0.02)
            ntheta = dyaw + np.random.normal(0, 0.01)
            # rotate delta by particle orientation
            th = self.particles[i,2]
            d_world = rot(th).dot(np.array([nx, ny]))
            self.particles[i,0] += d_world[0]
            self.particles[i,1] += d_world[1]
            self.particles[i,2] = self.wrap_angle(self.particles[i,2] + ntheta)

    def crater_cb(self, msg: CraterInfo):
        # received one crater observation in rover frame
        detection = {'x': msg.x, 'y': msg.y, 'radius': msg.radius, 'confidence': msg.confidence}
        # measurement update (compute weights)
        likelihoods = np.zeros(self.Np)
        for p_i in range(self.Np):
            likelihoods[p_i] = self.eval_particle_likelihood(self.particles[p_i, :], detection)
        # multiply weights and normalize
        self.weights *= likelihoods + 1e-12
        s = np.sum(self.weights)
        if s <= 0 or not np.isfinite(s):
            # bad weights -> reinit
            self.get_logger().warn('Weight collapse detected, reinitializing weights')
            self.weights[:] = 1.0 / self.Np
        else:
            self.weights /= s

        # effective sample size -> resample if needed
        ess = 1.0 / np.sum(self.weights**2)
        if ess < self.Np / 2.0:
            self.resample_particles()

        # publish estimated pose (weighted mean)
        mean = np.average(self.particles, axis=0, weights=self.weights)
        ps = PoseStamped()
        ps.header = Header()
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.header.frame_id = 'map'
        ps.pose.position.x = float(mean[0])
        ps.pose.position.y = float(mean[1])
        # convert yaw to quaternion (z,w)
        yaw = float(mean[2])
        qz = math.sin(yaw/2.0)
        qw = math.cos(yaw/2.0)
        ps.pose.orientation.z = qz
        ps.pose.orientation.w = qw
        self.pose_pub.publish(ps)

    def eval_particle_likelihood(self, particle_pose, detection):
        # Evaluate detection likelihood for this particle by matching to nearby map craters
        px, py, pth = particle_pose
        best_like = 1e-12
        for mc in self.map_craters:
            # transform map crater center to particle frame: vector from particle->map
            dx = mc['x'] - px
            dy = mc['y'] - py
            # rotate to particle frame
            x_p = math.cos(-pth) * dx - math.sin(-pth) * dy
            y_p = math.sin(-pth) * dx + math.cos(-pth) * dy
            # gating
            dist = math.hypot(detection['x'] - x_p, detection['y'] - y_p)
            if dist > self.gating:
                continue
            # radius error
            r_err = abs(detection['radius'] - mc['radius'])
            # gaussian likelihood
            pos_term = math.exp(-0.5 * (dist**2) / (self.sigma_pos**2))
            r_term = math.exp(-0.5 * (r_err**2) / (self.sigma_r**2))
            like = detection['confidence'] * pos_term * r_term
            if like > best_like:
                best_like = like
        return best_like

    def resample_particles(self):
        # systematic resampling
        N = self.Np
        positions = (np.arange(N) + np.random.random()) / N
        cumulative = np.cumsum(self.weights)
        indexes = np.searchsorted(cumulative, positions)
        self.particles = self.particles[indexes, :].copy()
        self.weights.fill(1.0 / N)

    @staticmethod
    def wrap_angle(a):
        return (a + math.pi) % (2 * math.pi) - math.pi

def main(args=None):
    rclpy.init(args=args)
    node = ParticleFilterLocalizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
