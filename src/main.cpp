#include <math.h>
#include <uWS/uWS.h>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"
#include "MPC.h"
#include "json.hpp"

// for convenience
using json = nlohmann::json;

// For converting back and forth between radians and degrees.
constexpr double pi() { return M_PI; }
double deg2rad(double x) { return x * pi() / 180; }
double rad2deg(double x) { return x * 180 / pi(); }

// Checks if the SocketIO event has JSON data.
// If there is data the JSON object in string format will be returned,
// else the empty string "" will be returned.
string hasData(string s) {
  auto found_null = s.find("null");
  auto b1 = s.find_first_of("[");
  auto b2 = s.rfind("}]");
  if (found_null != string::npos) {
    return "";
  } else if (b1 != string::npos && b2 != string::npos) {
    return s.substr(b1, b2 - b1 + 2);
  }
  return "";
}

// Evaluate a polynomial.
double polyeval(Eigen::VectorXd coeffs, double x) {
  double result = 0.0;
  for (int i = 0; i < coeffs.size(); i++) {
    result += coeffs[i] * pow(x, i);
  }
  return result;
}

// Fit a polynomial.
// Adapted from
// https://github.com/JuliaMath/Polynomials.jl/blob/master/src/Polynomials.jl#L676-L716
Eigen::VectorXd polyfit(Eigen::VectorXd xvals,
                        Eigen::VectorXd yvals,
                        int order) {
  assert(xvals.size() == yvals.size());
  assert(order >= 1 && order <= xvals.size() - 1);
  Eigen::MatrixXd A(xvals.size(), order + 1);

  for (int i = 0; i < xvals.size(); i++) {
    A(i, 0) = 1.0;
  }

  for (int j = 0; j < xvals.size(); j++) {
    for (int i = 0; i < order; i++) {
      A(j, i + 1) = A(j, i) * xvals(j);
    }
  }

  auto Q = A.householderQr();
  auto result = Q.solve(yvals);
  return result;
}

// Convert from Map Coordinates to Car's Coordinates
Eigen::MatrixXd convertToCarCoordinates(double x,
                                        double y,
                                        double psi,
                                        const vector<double>& xpoints,
                                        const vector<double>& ypoints) {
    assert(xpoints.size() == ypoints.size());
    unsigned  len = xpoints.size();

    auto points = Eigen::MatrixXd(2, len);

    for (auto k=0; k < len; k++) {
      points(0, k) =  cos(psi) * (xpoints[k] - x) + sin(psi) * (ypoints[k] - y);
      points(1, k) = -sin(psi) * (xpoints[k] - x) + cos(psi) * (ypoints[k] - y);
    }

    return points;
}

int main() {
  uWS::Hub h;

  // MPC is initialized here!
  MPC mpc;

  h.onMessage([&mpc](uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length,
                     uWS::OpCode opCode) {
    // "42" at the start of the message means there's a websocket message event.
    // The 4 signifies a websocket message
    // The 2 signifies a websocket event
    string sdata = string(data).substr(0, length);
    cout << sdata << endl;
    if (sdata.size() > 2 && sdata[0] == '4' && sdata[1] == '2') {
      string s = hasData(sdata);
      if (s != "") {
        auto j = json::parse(s);
        string event = j[0].get<string>();
        if (event == "telemetry") {
          // j[1] is the data JSON object
          vector<double> ptsx = j[1]["ptsx"];
          vector<double> ptsy = j[1]["ptsy"];
          double px           = j[1]["x"];
          double py           = j[1]["y"];
          double psi          = j[1]["psi"];
          double v            = j[1]["speed"];

          // 1 - Transform from Map coordinates to Car
          Eigen::MatrixXd waypoints = convertToCarCoordinates(px, py, psi, ptsx, ptsy);
          Eigen::VectorXd PointsX   = waypoints.row(0);
          Eigen::VectorXd PointsY   = waypoints.row(1);

          // 2 - Fit polynomial (degree 3) to the waypoints
          auto coefficients = polyfit(PointsX, PointsY, 3);

          // 3 - Determine Cross Track Error CTE
          // Note: in Car's coordinates, the cte is the intercept at x=0
          double cte = polyeval(coefficients, 0);

          // 4 - Orientation error
          // Note: in car'd coordinates, the orientation error epsi is
          // -atan(b1 + b2*x, b3* x^2),
          // but the car is always at x=0
          double epsi = -atan(coefficients[1]);

          // 5 - State (in car coordinates, x,y and orientation are 0)
          Eigen::VectorXd state(6);
          state << 0, 0, 0, v, cte, epsi;

          // 6 - Determine a Reference trajectory
          Resultant solution = mpc.Solve(state, coefficients);

          /*
          * 7 - Calculate steering angle and throttle using MPC.
          * Both are in between [-1, 1].
          */
          double steer_value      = solution.Delta.at(Latency_dt);
          double throttle_value   = solution.A.at(Latency_dt);
          mpc.delta_previous      = steer_value;
          mpc.acc_previous        = throttle_value;

          json msgJson;
          msgJson["steering_angle"] = -steer_value / 0.436332;
          msgJson["throttle"]       = throttle_value;

          cout << " x         " << px << endl;
          cout << " y         " << py << endl;
          cout << " psi       " << psi << endl;
          cout << " v         " << v << endl;
          cout << " cte       " << cte << endl;
          cout << " epsi      " << epsi << endl;
          cout << " steer     " << steer_value << endl;
          cout << " throttle  " << throttle_value << endl;

          // 8 - Display the MPC predicted trajectory
          /// vector<double> mpc_x_vals;
          /// vector<double> mpc_y_vals;

          msgJson["mpc_x"] = solution.X;
          msgJson["mpc_y"] = solution.Y;

          // 9 - Display the waypoints/reference line
          vector<double> next_x_vals;
          vector<double> next_y_vals;

          for (int j = 0; j < ptsx.size(); j++) {
            next_x_vals.push_back(PointsX(j));
            next_y_vals.push_back(PointsY(j));
          }

          msgJson["next_x"] = next_x_vals;
          msgJson["next_y"] = next_y_vals;


          auto msg = "42[\"steer\"," + msgJson.dump() + "]";
          std::cout << msg << std::endl;
          // Latency
          // The purpose is to mimic real driving conditions where
          // the car does actuate the commands instantly.
          //
          // Feel free to play around with this value but should be to drive
          // around the track with 100ms latency.
          //
          // NOTE: REMEMBER TO SET THIS TO 100 MILLISECONDS BEFORE
          // SUBMITTING.
          this_thread::sleep_for(chrono::milliseconds(100));
          ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
        } // if event == 'telemetry'
      }   // if s != ''
      else {
        // Manual driving
        std::string msg = "42[\"manual\",{}]";
        ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
      }
    } // if sdata == '42'
  });

  // We don't need this since we're not using HTTP but if it's removed the
  // program
  // doesn't compile :-(
  h.onHttpRequest([](uWS::HttpResponse *res, uWS::HttpRequest req, char *data,
                     size_t, size_t) {
    const std::string s = "<h1>Hello world!</h1>";
    if (req.getUrl().valueLength == 1) {
      res->end(s.data(), s.length());
    } else {
      // i guess this should be done more gracefully?
      res->end(nullptr, 0);
    }
  });

  h.onConnection([&h](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
    std::cout << "Connected!!!" << std::endl;
  });

  h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code,
                         char *message, size_t length) {
    ws.close();
    std::cout << "Disconnected" << std::endl;
  });

  int port = 4567;
  if (h.listen(port)) {
    std::cout << "Listening to port " << port << std::endl;
  } else {
    std::cerr << "Failed to listen to port" << std::endl;
    return -1;
  }
  h.run();
}
