#ifndef MPC_H
#define MPC_H

#include <vector>
#include "Eigen-3.3/Eigen/Core"

using namespace std;

const size_t N  = 12;
const double dt = 0.05;   // 50 ms
const int Latency_dt = 2; // Latency in units of dt

// Scale factors for Cost
const double SCALE_DELTA  = 1.0;
const double SCALE_ACC    = 10.0;
const double SCALE_DELTA_D = 400.0;
const double SCALE_ACC_D  = 2.0;

// Result from the Solver
struct Resultant {
    vector<double>  X;      // X coordinate
    vector<double>  Y;      // Y coordinate
    vector<double>  Psi;    // Orientation angle
    vector<double>  V;      // Velocity
    vector<double>  CTE;    // Cross Track Error
    vector<double>  EPsi;   // Orientation angle Error
    vector<double>  Delta;  // Steering angle
    vector<double>  A;      // Acceleration (throttle)
};

class MPC {
 public:
  MPC();

  virtual ~MPC();

  Resultant Solve(Eigen::VectorXd state, Eigen::VectorXd coeffs);

  double delta_previous {0};
  double acc_previous {0.1};

};

#endif /* MPC_H */
