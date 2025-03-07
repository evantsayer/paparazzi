import casadi as cs
import numpy as np
import matplotlib.pyplot as plt
from casadi import Opti
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class PathPlanner():
    
    def __init__(self):

        self.problem = cs.Opti()

        # Parameters
        #self.N = self.problem.parameter() # Prediction horizon TODO: this migh have to be a constant and dt a parameter
        #self.C = self.problem.parameter() # Intermediate timestamp
        self.N = 80
        #self.C = 80
        self.position_0 = self.problem.parameter(3, 1)
        self.velocity_0 = self.problem.parameter(3, 1)
        self.position_t = self.problem.parameter(3, 1)
        self.velocity_t = self.problem.parameter(3, 1) 

        self.prev_position = self.problem.parameter(3, self.N)
        self.prev_velocity = self.problem.parameter(3, self.N)

        
        
        # Decision variables
        self.input: cs.MX = self.problem.variable(3, self.N - 1)
        self.Gamma: cs.MX = self.problem.variable(1, self.N - 1) # slack variable

        #self.curr_position = self.problem.variable(3, self.N)
        #self.curr_velocity = self.problem.variable(3, self.N)

        # Constants
        self.g = 9.81

        self.t_f = 2.0

        self.dt = self.problem.parameter(1, 1)

        self.w = 10^(-5)
        self.alpha = 1.5
        self.position_tolerance = 0.05

        self.input_min = 0.6
        self.input_max = 23.2

        self.e_x = cs.MX([1, 0, 0])
        self.e_y = cs.MX([0, 1, 0])
        self.e_z = cs.MX([0, 0, 1])

        self.Theta_max = 100/180*cs.pi # TODO

        self.convergence_criteria = 0.05

        
        # self.position: cs.MX = self.problem.variable(3, N) 
        # self.velocity: cs.MX = self.problem.variable(3, N)
        self.R = 1.0
        H = np.array([[1, 0, 0], [0, 1, 0], [0, 0, .0001]])
        self.obstacles = [{"position":cs.DM([1, 2, 0]), "hessian":H, "radius": self.R}, {"position":cs.DM([2, 5, 0]), "hessian":H, "radius": 1}]

        # Setup solver
        p_opts = {"expand":True, }
        s_opts = {"max_iter": 500, "tol":1}
        self.problem.solver('ipopt', p_opts, s_opts)

        self.plan_trajectory()

    def plan_trajectory(self):
        # Setup parameters that are constant across iterations
        self.problem.set_value(self.position_0, cs.DM([0, 0, 0]))
        self.problem.set_value(self.velocity_0, cs.DM([0, 0, 0]))
        self.problem.set_value(self.position_t, cs.DM([2.5, 6, 0]))
        self.problem.set_value(self.velocity_t, cs.DM([0, 0, 0]))

        # Solve without convexification

        x_initial = np.linspace(0, 2.5, self.N)
        y_initial = (6-0)/(2.5-0) * (x_initial - 6) +2.5       

        self.set_constraints()
        
        self.set_objective()

        while True:
            try:

                self.solution = self.problem.solve()

                # Simulate dynamics
                u = self.solution.value(self.input)
                position_trajectory = np.array([[0], [0], [0]])
                velocity_trajectory = np.array([[0], [0], [0]])

                for i in range(self.N-1):
                    next_position, next_velocity = self.dynamics_2(position_trajectory[:, -1], velocity_trajectory[:, -1], u[:, i])
                    position_trajectory = np.append(position_trajectory, np.reshape(next_position, (3, 1)), 1)
                    velocity_trajectory = np.append(velocity_trajectory, np.reshape(next_velocity, (3, 1)), 1)

                # Change previous state trajectory
                self.problem.set_value(self.prev_position, position_trajectory)
                self.problem.set_value(self.prev_velocity, velocity_trajectory)
                # Give initial guess
                self.problem.set_initial(self.Gamma, self.solution.value(self.Gamma))
                self.problem.set_initial(self.input, self.solution.value(self.input))
                
                break

            except Exception as e:
                print(e)
                dt = self.t_f*self.alpha/self.N
                self.problem.set_value(self.dt, dt)
        print(u)
        plt.scatter(np.array(position_trajectory)[0, :], np.array(position_trajectory)[1, :])
        plt.plot(np.array(position_trajectory)[0, :], np.array(position_trajectory)[1, :])
        plt.show()
        # Solve with obstacles
        self.relaxation_term = self.problem.variable(1, 2) # TODO: this has to be variable length according to the number of obstacles
        self.set_obstacle_constraints()
        self.set_objective_relax()

        for i in range(10):
            try:
                self.solution = self.problem.solve()
                u = self.solution.value(self.input)

                # Simulate dynamics
                u = self.solution.value(self.input)
                position_trajectory = np.array([[0], [0], [0]])
                velocity_trajectory = np.array([[0], [0], [0]])

                for i in range(self.N-1):
                    next_position, next_velocity = self.dynamics_2(position_trajectory[:, -1], velocity_trajectory[:, -1], u[:, i])
                    position_trajectory = np.append(position_trajectory, np.reshape(next_position, (3, 1)), 1)
                    velocity_trajectory = np.append(velocity_trajectory, np.reshape(next_velocity, (3, 1)), 1)

                diff = np.linalg.norm(position_trajectory-self.solution.value(self.prev_position), np.inf)
                print(diff)
                if diff < self.convergence_criteria:
                    print(f"EXITED WITH CONV CRITERIA: {i}")
                    break

                self.problem.set_value(self.prev_position, position_trajectory)
                self.problem.set_value(self.prev_velocity, velocity_trajectory)
                # Give initial guess
                self.problem.set_initial(self.Gamma[:-1], self.solution.value(self.Gamma[1:]))
                self.problem.set_initial(self.input, self.solution.value(self.input))

            except Exception as e:
                print(e)
                dt = self.t_f*self.alpha/self.N
                self.problem.set_value(self.dt, dt)
                 
            
        #plt.gca().add_patch(plt.Circle((5, 5), self.R, color='r'))
        #plt.scatter(np.array(position_trajectory)[0, :], np.array(position_trajectory)[1, :])
        #plt.plot(np.array(position_trajectory)[0, :], np.array(position_trajectory)[1, :])
        #plt.show()
        self.plot_trajectory_with_obstacles(position_trajectory, velocity_trajectory, self.obstacles)
    
    def plot_trajectory_with_obstacles(self, position_trajectory, velocity_trajectory, obstacles):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot trajectory
        ax.plot(position_trajectory[0, :], position_trajectory[1, :], position_trajectory[2, :], label='Trajectory', color='b')
        ax.scatter(position_trajectory[0, :], position_trajectory[1, :], position_trajectory[2, :], color='b', s=5)
        
        # Plot velocity field as arrows
        for i in range(0, position_trajectory.shape[1], 3):  # Reduce density of arrows
            ax.quiver(position_trajectory[0, i], position_trajectory[1, i], position_trajectory[2, i],
                    velocity_trajectory[0, i], velocity_trajectory[1, i], velocity_trajectory[2, i],
                    color='g', length=0.5, normalize=True)
        
        # Plot obstacles (ellipsoids)
        for obstacle in obstacles:
            center = np.array(obstacle["position"]).flatten()
            hessian = np.array(obstacle["hessian"])
            radius = obstacle["radius"]
            radii, rotation = np.linalg.eigh(np.linalg.inv(hessian))  # Get radii and rotation from Hessian
            radii = np.sqrt(radii) * radius  # Scale radii by the given obstacle radius
            
            u = np.linspace(0, 2 * np.pi, 20)
            v = np.linspace(0, np.pi, 10)
            
            x = radii[0] * np.outer(np.cos(u), np.sin(v))
            y = radii[1] * np.outer(np.sin(u), np.sin(v))
            z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
            
            ellipsoid = np.stack([x, y, z], axis=0)
            ellipsoid = (rotation @ ellipsoid.reshape(3, -1)).reshape(3, *x.shape)
            
            ax.plot_surface(ellipsoid[0] + center[0], ellipsoid[1] + center[1], ellipsoid[2] + center[2], color='r', alpha=0.5)
        
         # Set equal aspect ratio
        max = np.array([position_trajectory[0, :].max(), position_trajectory[1, :].max(), position_trajectory[2, :].max()]).max()
        min = np.array([position_trajectory[0, :].min(), position_trajectory[1, :].min(), position_trajectory[2, :].min()]).min()

        ax.set_xlim(min, max)
        ax.set_ylim(min, max)
        ax.set_zlim(min, max)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Trajectory with Velocity Field and Obstacles')
        ax.legend()
        plt.show()


        ## Solve with convexification iteratively
        #self.problem.set_value(self.relaxation_term, 0.1)

    def set_objective(self) -> float:
        
        term1 = self.Gamma @ self.Gamma.T 
        self.problem.minimize(self.w * term1)

    def set_objective_relax(self):
        term1 = self.Gamma @ self.Gamma.T 
        term2 = self.relaxation_term @ self.relaxation_term.T
        self.problem.minimize(self.w * term1 + term2)

    def set_constraints(self) -> None:
        constraints = []
        # Simulate dynamics
        position_trajectory = [self.position_0]
        velocity_trajectory = [self.velocity_0]
        for i in range(self.N-1):
            next_position, next_velocity = self.dynamics(position_trajectory[-1], velocity_trajectory[-1], self.input[:, i])
            position_trajectory.append(next_position)
            velocity_trajectory.append(next_velocity)
        
        # Initial consitions
        constraints.append(self.input[:, 0] == self.g * self.e_z)

        # Intermediate constraints
        #constraints.append(position_trajectory[20 + 1] == [3, 5, 0])
        #onstraints.append(self.e_x.T @ self.velocity[:, self.c + 1] == 0)
        #onstraints.append(self.e_z.T @ self.velocity[:, self.c + 1] == 0)

        # Terminal constraints
        constraints.append(position_trajectory[-1] == self.position_t)
        constraints.append(velocity_trajectory[-1] == self.velocity_t)
        constraints.append(self.input[:, -1] == self.g * self.e_z)

        # Slack variable constraints
        # Nx3 x 3xN <= 
        constraints.append(self.input[0, :]**2 + self.input[1, :]**2 + self.input[2, :]**2  <= self.Gamma**2) # TODO: incorrect
        constraints.append(self.Gamma*np.cos(self.Theta_max) <= self.e_z.T @ self.input)
        constraints.append(self.input_min <= self.Gamma)
        constraints.append(self.input_max >= self.Gamma)
        # 1xN x const == 1x3 x 3xN
        

        # Set bounding area
        #constraints.append(-50 <= self.e_y.T @ position_trajectory[2:][1])
        #constraints.append( 50 >= self.e_y.T @ position_trajectory[2:][1])
        #constraints.append(-50 <= self.e_x.T @ position_trajectory[2:][0])
        #constraints.append( 50 >= self.e_x.T @ position_trajectory[2:][0])
        #constraints.append(-50 <= self.e_z.T @ position_trajectory[2:][2])
        #constraints.append( 50 >= self.e_z.T @ position_trajectory[2:][2])

        #TODO remove this constraint

        self.problem.subject_to(constraints)

    def set_obstacle_constraints(self) -> None:

        position_trajectory = [self.position_0]
        velocity_trajectory = [self.velocity_0]
        for i in range(self.N-1):
            next_position, next_velocity = self.dynamics(position_trajectory[-1], velocity_trajectory[-1], self.input[:, i])
            position_trajectory.append(next_position)
            velocity_trajectory.append(next_velocity)

        constraints = []
        constraints.append(self.relaxation_term >= 0)
        for j, obstacle in enumerate(self.obstacles):
            distance_from_obstacle = self.prev_position - obstacle["position"]
            print(position_trajectory)
            dposition = position_trajectory - self.prev_position

            # 3x3 x 3xN
            xi = obstacle["hessian"] @ distance_from_obstacle
            xi = (xi[0]**2 + xi[1]**2 + xi[2]**2)**(1/2)                
            # 3
            zeta = (obstacle["hessian"].T @ obstacle["hessian"] @ distance_from_obstacle) * 1/xi
        # 1xN + Nx3 x 3xN 
            constraints.append(xi + zeta.T @ dposition >= obstacle["radius"] - self.relaxation_term[0, j])

        self.problem.subject_to(constraints)

    def dynamics(self, position: cs.MX, velocity: cs.MX, input: cs.MX) -> tuple[cs.MX]:

        next_position = position + velocity * self.dt
        next_velocity = velocity + (input - self.g * self.e_z) * self.dt

        return next_position, next_velocity
    
    def dynamics_2(self, position: cs.MX, velocity: cs.MX, input: cs.MX) -> tuple[cs.MX]:
        next_position = position + velocity * self.solution.value(self.dt)
        next_velocity = velocity + (input - self.g * self.solution.value(self.e_z)) * self.solution.value(self.dt)

        return next_position, next_velocity
    
if __name__ == "__main__":
    planner = PathPlanner()