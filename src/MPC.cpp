#include "MPC.h"
#include <cppad/cppad.hpp>
#include <cppad/ipopt/solve.hpp>
#include "Eigen-3.3/Eigen/Core"

using CppAD::AD;

// This is the length from front to CoG that has a similar radius.
const double Lf = 2.67;

// Useful constants
const double REF_CTE = 0;   // desired CTE, Cross Track Error
const double REF_EPSI = 0;  // desired EPsi
const double REF_V  = 85;   // max car velocity, mph

// Start variable positions, since the Solver takes State and Actuator variables in a single vector
size_t  x_start     = 0;
size_t  y_start     = x_start + N;
size_t  psi_start   = y_start + N;
size_t  v_start     = psi_start + N;
size_t  cte_start   = v_start + N;
size_t  epsi_start  = cte_start + N;
size_t  delta_start = epsi_start + N;
size_t  a_start     = delta_start + N - 1;

class FG_eval {
 public:
  // Fitted polynomial coefficients
  Eigen::VectorXd coeffs;
  vector<double> actuation_history;

  FG_eval(Eigen::VectorXd coeffs, vector<double> history ) {
    this->coeffs = coeffs;
    this->actuation_history = history;
  }

  typedef CPPAD_TESTVECTOR(AD<double>) ADvector;
  void operator()(ADvector& fg, const ADvector& vars) {
    // The cost is stored is the first element of `fg`.
    // Any additions to the cost should be added to `fg[0]`.
    fg[0] = 0;

    // 1. Cost Function
    // State: [x,y,ψ,v,cte,eψ]
    // 1a - Cost based on Reference State; for Trajectory, set cost to 0
    for (int t = 0; t < N; t++) {
      fg[0] +=  1.0 * CppAD::pow(vars[cte_start + t] - REF_CTE, 2);   // Cross Track Error
      fg[0] +=  1.0 * CppAD::pow(vars[epsi_start + t] - REF_EPSI, 2); // Orientation error
      fg[0] +=  1.0 * CppAD::pow(vars[v_start + t] - REF_V, 2);       // Velocity error
    }

    // 1b - Minimise the use of Actuators
    // Add Scaling factor to smooth out
    for (int t = 0; t < N - 1; t++) {
      fg[0] +=  SCALE_DELTA * CppAD::pow(vars[delta_start + t], 2);
      fg[0] +=  SCALE_ACC   * CppAD::pow(vars[a_start + t], 2);
    }

    // 1c - Minimize the value gap between sequential actuations
    // Add Scaling factor to smooth out
    for (int t = 0; t < N - 2; t++) {
      fg[0] +=  SCALE_DELTA_D * CppAD::pow(vars[delta_start + t + 1] - vars[delta_start + t], 2);
      fg[0] +=  SCALE_ACC_D   * CppAD::pow(vars[a_start + t + 1]  - vars[a_start + t], 2);
    }

    // 2 -  Setup Constraints
    // 2a - Initial Constraints
    // We initialize the model to the initial state.
    // Recall fg[0] is reserved for the cost value, so the other indices are bumped up by 1.
    fg[1 + x_start]     = vars[x_start];
    fg[1 + y_start]     = vars[y_start];
    fg[1 + psi_start]   = vars[psi_start];
    fg[1 + v_start]     = vars[v_start];
    fg[1 + cte_start]   = vars[cte_start];
    fg[1 + epsi_start]  = vars[epsi_start];

    // 2b - Rest of the Constraints
    for (int t = 1; t < N ; t++) {
      // State at time t + 1
      AD<double>  x1    = vars[x_start + t];
      AD<double>  y1    = vars[y_start + t];
      AD<double>  psi1  = vars[psi_start + t];
      AD<double>  v1    = vars[v_start + t];
      AD<double>  cte1  = vars[cte_start + t];
      AD<double>  epsi1 = vars[epsi_start + t];

      // State at time t
      AD<double>  x0    = vars[x_start + t - 1];
      AD<double>  y0    = vars[y_start + t - 1];
      AD<double>  psi0  = vars[psi_start + t - 1];
      AD<double>  v0    = vars[v_start + t - 1];
      AD<double>  cte0  = vars[cte_start + t - 1];
      AD<double>  epsi0 = vars[epsi_start + t - 1];

      // Only consider the actuation at time t.
      AD<double>  delta0  = vars[delta_start + t - 1];
      AD<double>  a0      = vars[a_start + t - 1];

      // polynomial of degree 3
      AD<double>  f0      = coeffs[0] + coeffs[1] * x0 + coeffs[2] * x0 * x0 + coeffs[3]*x0 * x0 * x0;
      // desired epsi = arctan(coeffs[1])
      AD<double>  psides0 = CppAD::atan(coeffs[1] + 2*coeffs[2]*x0  + 3*coeffs[3]*x0*x0);

      // 2c - Constrain these values to 0
      fg[1 + x_start + t]   = x1    - (x0 + v0 * CppAD::cos(psi0) * dt);
      fg[1 + y_start + t]   = y1    - (y0 + v0 * CppAD::sin(psi0) * dt);
      fg[1 + psi_start + t] = psi1  - (psi0 + v0 * delta0 / Lf * dt);
      fg[1 + v_start + t]   = v1    - (v0 + a0 * dt);
      fg[1 + cte_start + t] = cte1  - ((f0 - y0) + (v0 * CppAD::sin(epsi0) * dt));
      fg[1 + epsi_start + t]= epsi1 - ((psi0 - psides0) + v0 * delta0 / Lf * dt);
    }

  }
};

//
// MPC class definition implementation.
//

MPC::MPC() {}
MPC::~MPC() {}

Resultant MPC::Solve(Eigen::VectorXd x0, Eigen::VectorXd coeffs) {
  // Note: x0 is the current State
  bool ok = true;
  // size_t i;
  typedef CPPAD_TESTVECTOR(double) Dvector;

  // 1- Set the number of model variables
  size_t n_vars = N * 6 + (N - 1) * 2;
  size_t n_constraints = N * 6;

  double x    = x0[0];
  double y    = x0[1];
  double psi  = x0[2];
  double v    = x0[3];
  double cte  = x0[4];
  double epsi = x0[5];

  // 2- Initial value of the independent variables.
  Dvector vars(n_vars);
  for (int i = 0; i < n_vars; i++) {
    vars[i] = 0.0;
  }

  // 3- Set initial var
  vars[x_start]   = x;
  vars[y_start]   = y;
  vars[psi_start] = psi;
  vars[v_start]   = v;
  vars[cte_start] = cte;
  vars[epsi_start]= epsi;

  // 4- Set lower and upper limits for variables.
  Dvector vars_lowerbound(n_vars);
  Dvector vars_upperbound(n_vars);

  // 4a- Set non-actuator upper and lower-limits
  for (int i = 0; i < delta_start; i++) {
    vars_lowerbound[i]  = -1.0e19;
    vars_upperbound[i]  = 1.0e19;
  }

  // 4b- steering angle constrains
  for (int i = delta_start; i < a_start; i++) {
    vars_lowerbound[i]  = -0.436332;
    vars_upperbound[i]  = 0.436332;
  }

  // 4c- set delta to previous:
  for (int i = delta_start; i < delta_start + Latency_dt; i++) {
    vars_lowerbound[i] = delta_previous;
    vars_upperbound[i] = delta_previous;
  }

  // 4d- acceleration constrains: +/- 1.0
  for (int i = a_start; i < n_vars; i++) {
    vars_lowerbound[i] = -1.0;
    vars_upperbound[i] = 1.0;
  }

  // 4e- acceleration previous
  for (int i = a_start; i < a_start + Latency_dt; i++) {
    vars_lowerbound[i] = acc_previous;
    vars_upperbound[i] = acc_previous;
  }

  // 5- Lower and upper limits for the constraints

  // 5a- Should be 0 besides initial state.
  Dvector constraints_lowerbound(n_constraints);
  Dvector constraints_upperbound(n_constraints);
  for (int i = 0; i < n_constraints; i++) {
    constraints_lowerbound[i] = 0;
    constraints_upperbound[i] = 0;
  }

  // 5b- lowerbound
  constraints_lowerbound[x_start]     = x;
  constraints_lowerbound[y_start]     = y;
  constraints_lowerbound[psi_start]   = psi;
  constraints_lowerbound[v_start]     = v;
  constraints_lowerbound[cte_start]   = cte;
  constraints_lowerbound[epsi_start]  = epsi;

  // 5c- uppwerbound
  constraints_upperbound[x_start]     = x;
  constraints_upperbound[y_start]     = y;
  constraints_upperbound[psi_start]   = psi;
  constraints_upperbound[v_start]     = v;
  constraints_upperbound[cte_start]   = cte;
  constraints_upperbound[epsi_start]  = epsi;

  // 6- object that computes objective and constraints
  vector<double> actuation_history = { delta_previous, acc_previous};
  FG_eval fg_eval(coeffs, actuation_history);

  // options for IPOPT solver
  std::string options;
  // Uncomment this if you'd like more print information
  options += "Integer print_level  0\n";
  options += "Sparse  true        forward\n";
  options += "Sparse  true        reverse\n";
  options += "Numeric max_cpu_time          0.5\n";

  // 7 - place to return solution
  CppAD::ipopt::solve_result<Dvector> solution;

  // 8 - solve the problem
  CppAD::ipopt::solve<Dvector, FG_eval>(
      options, vars, vars_lowerbound, vars_upperbound, constraints_lowerbound,
      constraints_upperbound, fg_eval, solution);

  // Check some of the solution values
  ok &= solution.status == CppAD::ipopt::solve_result<Dvector>::success;

  Resultant result; // struct sent to Simulator
  for (auto k = 0; k < N - 1; k++) {
    // we only need these four values for Simulator
    result.X.push_back(solution.x[x_start + k]);
    result.Y.push_back(solution.x[y_start + k]);
    result.Delta.push_back(solution.x[delta_start + k]);
    result.A.push_back(solution.x[a_start + k]);

    // values below not really sent to Simulator
    result.Psi.push_back(solution.x[psi_start + k]);
    result.V.push_back(solution.x[v_start + k]);
    result.CTE.push_back(solution.x[cte_start + k]);
    result.EPsi.push_back(solution.x[epsi_start + k]);
  }

  // Cost
  auto cost = solution.obj_value;
  std::cout << "Cost     " << cost << std::endl;

  return result;
}
