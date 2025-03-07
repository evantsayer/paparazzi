import numpy as np
import matplotlib.pyplot as plt
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
import casadi as cs
import copy

class PathPlanner(AcadosOcp):
    def __init__(self, final_position=np.array([5, 5, 5])):
        AcadosOcp.__init__(self)

        # Constants
        self.N = 50  # Prediction horizon
        self.t_f = 1  # Time step
        self.g = 9.81
        self.w = 10**(-5)
        self.alpha = 1.5
        self.input_min = 0.6
        self.input_max = 23.2
        self.Theta_max = 100 / 180 * cs.pi
        self.convergence_criteria = 0.05

        # Define static obstacles
        self.R = 1.0
        H = np.array([[1, 0, 0], [0, 1, 0], [0, 0, .0001]])
        self.obstacles = [{"position":np.array([2, 2, 0]), "hessian":H, "radius": self.R},
                          {"position":np.array([2, 5, 0]), "hessian":H, "radius": self.R},
                          {"position":np.array([3, 5, 0]), "hessian":H, "radius": self.R},
                          {"position":np.array([-1, 2, 0]), "hessian":H, "radius": self.R},
                          {"position":np.array([-0.5, 5, 0]), "hessian":H, "radius": self.R},
                          {"position":np.array([0, 3, 0]), "hessian":H, "radius": self.R}]

        self.final_position = final_position
        self.start_position = np.array([1, 1, 1])

        self.solver_options.print_level = 1

        # Dimensions
        self.nx = 6 + len(self.obstacles)
        self.nu = 4 
        self.ny = 2

        # Setup problem
        self.model = self.create_model()
        self.set_solver_options()
        self.set_cost()
        self.set_constraints()
        self.set_obstacle_constraints()
        self.ocp_solver = AcadosOcpSolver(self)
        self.plan_trajectory()

    def set_solver_options(self) -> None:

        # Define the prediction horizon and time step
        self.dims.N = self.N - 1
        self.solver_options.tf = self.t_f
        
        # Solver options
        self.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        self.solver_options.integrator_type = "ERK"
        self.solver_options.nlp_solver_type = "SQP"

    def set_cost(self) -> None:
        
        # Define cost function
        self.cost.cost_type = "LINEAR_LS"
        self.cost.cost_type_e = "LINEAR_LS"

        self.dims.ny = self.ny
        self.dims.ny_0 = self.ny

        
        # Cost weight Gamma:= w, relaxation_term := 1
        W = np.eye(self.ny)
        W[0, 0] = W[0, 0]*self.w    #Weight of gamma
        W[1, 1] = W[1, 1]*1/self.N  #Weight of relaxation terms
        
        ### Select stage outputs ###
        # Select relaxation variable
        Vx = np.zeros((self.ny, self.nx))
        Vx[1, 6:] = np.ones((1, len(self.obstacles)))
        
        # Select Gamma
        Vu = np.zeros((self.ny, self.nu))
        Vu[0, 3] = 1

        # No reference trajectory
        yref = np.zeros((self.ny, 1))

        # Set previously defined matrices
        self.cost.W = np.asfortranarray(W)
        self.cost.Vx = np.asfortranarray(Vx)
        self.cost.Vu = np.asfortranarray(Vu)
        self.cost.Vx_0 = np.asfortranarray(Vx)
        self.cost.Vu_0 = np.asfortranarray(Vu)
        self.cost.yref = np.asfortranarray(yref)
        self.cost.yref_0 = np.asfortranarray(yref)
        self.cost.ny = self.ny

    def set_external_cost(self) -> None:
        self.cost.cost_type = "EXTERNAL"
        self.cost.cost_type_0 = "EXTERNAL"
        self.cost.cost_type_e = "EXTERNAL"

        sum_relaxation_term = 0
        for i in range(len(self.obstacles)):
            sum_relaxation_term += self.model.x[6+i]

        self.model.cost_expr_ext_cost = self.model.u[3]**2 + sum_relaxation_term
        self.model.cost_expr_ext_cost_0 = self.model.u[3]**2 + sum_relaxation_term

    def set_constraints(self) -> None:

        # Define terminal position constraints TODO: Add contraint for speed
        self.constraints.idxbx_e = np.concatenate((np.array([0, 1, 2]), np.arange(6, 6+len(self.obstacles)))) # Position and relaxation term constraints 
        self.constraints.lbx_e = np.concatenate((self.final_position[:3], np.array([0]*len(self.obstacles))))
        self.constraints.ubx_e = np.concatenate((self.final_position[:3], np.array([1000]*len(self.obstacles))))
        
        # Define initial constraints
        self.constraints.idxbx_0 = np.array([0, 1, 2, 3, 4, 5])  # Position constraints
        self.constraints.lbx_0 = np.concatenate((self.start_position, np.array([0, 0, 0])))
        self.constraints.ubx_0 = np.concatenate((self.start_position, np.array([0, 0, 0])))

        # Define stage constraints #TODO: remove this if
        # moving it to end constraint proven to be better
        #self.constraints.idxbx = np.array([6, 7])  # Relaxation term constraints
        #self.constraints.lbx = np.array([0, 0])
        #self.constraints.ubx = np.array([1000, 1000])

        ## Define bounding box constraints
        self.constraints.idxbx = np.array([0, 1, 2])  # Position constraints
        self.constraints.lbx = np.array([-200, -200, -200])
        self.constraints.ubx = np.array([200, 200, 200])
        
        # Define bounds on gamma and input (to later
        # pose constraint on first and last input)
        self.constraints.idxbu = np.array([0, 1, 2, 3])
        self.constraints.lbu = np.array([-23, -23, -23, self.input_min])
        self.constraints.ubu = np.array([ 23,  23,  23, self.input_max])

        # Define nonlinear constraints
        nonlin_constraints = []
        #TODO: fix this
        #nonlin_constraints.append(
        #    self.model.u[3]**2 - (self.model.u[0]**2 + self.model.u[1]**2 + self.model.u[2]**2)
        #)
        
        nonlin_constraints.append(
            - self.model.u[3] * np.cos(self.Theta_max) + cs.MX([0, 0, 1]).T @ self.model.u[:3]
        )

        self.model.con_h_expr = cs.vertcat(*nonlin_constraints)
        self.constraints.uh = np.array([100000] * len(nonlin_constraints))
        self.constraints.lh = np.array([0] * len(nonlin_constraints))
        
    def set_obstacle_constraints(self) -> None:
        constraint_list = []

        for j, obstacle in enumerate(self.obstacles):

            d_obstacle = self.model.p[:3] - obstacle["position"]
            d_difference = self.model.x[:3] -self.model.p[:3]

            xi = obstacle["hessian"] @ d_obstacle
            xi = cs.sqrt(xi[0]**2 + xi[1]**2 + xi[2]**2)                
            # 3
            zeta = (obstacle["hessian"].T @ obstacle["hessian"] @ d_obstacle) * 1/(xi + 0.001)
            # 1xN + Nx3 x 3xN 
            constraint_list.append(xi + zeta.T @ d_difference - obstacle["radius"] + self.model.x[6+j])


        self.model.con_h_expr = cs.vertcat(*constraint_list, self.model.con_h_expr)
        self.constraints.uh = np.concatenate((np.array([10000000] * len(self.obstacles)), self.constraints.uh))
        self.constraints.lh = np.concatenate((np.zeros((len(self.obstacles), 1)), np.reshape(self.constraints.lh, (-1, 1))))

    def create_model(self) -> None:

        model = AcadosModel()
        model.name = "quadrotor"

        ### State variables ###
        position = cs.MX.sym('position', 3)
        velocity = cs.MX.sym('velocity', 3)
        v = cs.MX.sym('relaxation_term', len(self.obstacles))
        # Relaxation term is defined as a state since it is
        # constant along the horizon corresponding to dxdt = 0

        ### Inputs ###
        u = cs.MX.sym('u', 3)
        gamma = cs.MX.sym('gamma', 1) 
        # Gamma is defined as an input as it is varying along the
        # horizon and does not follow dynamics constraints
        
        ### Parameters ###
        p = cs.MX.sym('previous_position', 3)

        ### Dynamics ###
        f_expl = cs.vertcat(
            velocity,
            u - self.g * cs.MX([0, 0, 1]), 
            cs.MX([0]*len(self.obstacles)))

        f_impl = cs.vertcat(
            position - velocity,
            velocity - (u - self.g * cs.MX([0, 0, 1])),
            v)

        # Set previously defined variables
        model.x = cs.vertcat(position, velocity, v)
        model.u = cs.vertcat(u, gamma)
        model.p = p

        model.y = model.x
        model.f_expl_expr = f_expl
        model.f_impl_expr = f_impl

        self.parameter_values = np.zeros((3,))

        return model

    def plan_trajectory(self) -> None:

        total_time = 0
        previous_position_trajectory = np.zeros((3, self.N))
        position_trajectory = np.zeros((3, self.N))
        velocity_trajectory = np.zeros((3, self.N))
        u_trajectory = np.zeros((3, self.N))

        # Initial guess: Linear trajectory towards final position
        for i in range(self.N):
            position_trajectory[:, i] = (self.final_position * (i / (self.N - 1)))
            self.ocp_solver.set(i, "p", position_trajectory[:, i])
            velocity_trajectory[:, i] = np.zeros(3)  # Start with zero velocity
            u_trajectory[:, i] = np.zeros(3)  # Start with zero control input

        for j in range(10):
            try:
                status = 0

                # Pose hovering constraint on initial and final input
                self.ocp_solver.set(0, "lbu", np.array([0, 0, self.g*1, 0.6]))
                self.ocp_solver.set(0, "ubu", np.array([0, 0, self.g*1, 23.5]))
                self.ocp_solver.set(self.N-2, "lbu", np.array([0, 0, self.g*1, 0.6]))
                self.ocp_solver.set(self.N-2, "ubu", np.array([0, 0, self.g*1, 23.5]))

                # Solve for initial condition
                self.ocp_solver.solve_for_x0(np.concatenate((self.start_position, np.array([0.0, 0.0, 0.0]))))
                
                # Save solve time
                solver_time = self.ocp_solver.get_stats("time_tot")
                total_time += solver_time

                for i in range(self.N):
                    x = self.ocp_solver.get(i, "x")
                    prev_position = self.ocp_solver.get(i, "p")
                    u = self.ocp_solver.get(i, "u")
                    print(f"u: {u}")
                    #print(x[:3])
                    #print(prev_position)

                    obstacle_n = 5
                    previous_position_trajectory[:, i] = prev_position
                    d_obstacle = prev_position[:3] - self.obstacles[obstacle_n]["position"]
                    d_difference = x[:3] - prev_position[:3]

                    xi = self.obstacles[obstacle_n]["hessian"] @ d_obstacle
                    xi = cs.sqrt(xi[0]**2 + xi[1]**2 + xi[2]**2)                
                    # 3
                    zeta = (self.obstacles[obstacle_n]["hessian"].T @ self.obstacles[obstacle_n]["hessian"] @ d_obstacle) * 1/xi
                    # 1xN + Nx3 x 3xN 
                    #print(xi + zeta.T @ d_difference - self.obstacles[obstacle_n]["radius"] + x[6+obstacle_n])
                    #print(f"relaxation term {obstacle_n}: {x[6+obstacle_n]}")
                    #print(f"distance from obst: {np.linalg.norm(d_obstacle[:2])}")
                    position_trajectory[:, i] = x[:3]
                    self.ocp_solver.set(i, "p", x[:3])
                    velocity_trajectory[:, i] = x[3:6]
                
                if status != 0:
                    #TODO: Implement proper error handling
                    print(f"ACADOS solver failed at iteration {i}")
                    break
                else:
                    print(f"Acados ran successfully: {solver_time * 1000} ms")
                    if np.linalg.norm(previous_position_trajectory-position_trajectory, np.inf) < self.convergence_criteria:
                        print(f"Converged in iteration {j}, Total time: {total_time *1000} ms")
                        break

            except Exception as e:
                print(f"Error at iteration {i}: {e}")

        self.plot_trajectory(position_trajectory, self.obstacles)
        self.plot_trajectory_with_obstacles(position_trajectory, velocity_trajectory, self.obstacles)

    def plot_trajectory(self, position_trajectory, obstacles) -> None:
        plt.figure()
        plt.plot(position_trajectory[0, :], position_trajectory[1, :], label="Planned Path")
        plt.scatter(self.final_position[0], self.final_position[1], color='r', label="Final Position")
        for obs in obstacles:
            plt.scatter(obs["position"][0], obs["position"][1], color='k', marker='x', label="Obstacle")
            plt.gca().add_artist(plt.Circle((obs["position"][0], obs["position"][1]), obs["radius"], color='r', alpha=0.5))
        plt.legend()
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.title("Path Planning Trajectory")
        plt.grid()
        plt.show()

    def plot_trajectory_with_obstacles(self, position_trajectory, velocity_trajectory, obstacles) -> None:
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

if __name__ == "__main__":
    planner = PathPlanner(final_position=np.array([2.5, 6, 1]))
