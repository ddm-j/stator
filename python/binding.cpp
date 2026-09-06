#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include <stator/core/types.h>
#include <stator/core/numeric_result.h>
#include <stator/physics/constants.h>
#include <stator/physics/ball.h>
#include <stator/physics/departure.h>
#include <stator/physics/rim.h>
#include <stator/physics/predict.h>

namespace py = pybind11;

PYBIND11_MODULE(_stator, m)
{
    using namespace stator::core;
    using namespace stator::physics;

    m.doc() = "Python bindings for stator";

    // Linear Regression Result (what fitmed returns)
    py::class_<LinRegResult>(m, "LinRegResult")
            .def(py::init<>())
            .def(py::init<real, real, real>(),
                 py::arg("a"), py::arg("b"), py::arg("abdev"))
            .def_readonly("a", &LinRegResult::a)
            .def_readonly("b", &LinRegResult::b)
            .def_readonly("abdev", &LinRegResult::abdev)
            .def("__repr__", [](const LinRegResult& r) {
                return "LinRegResult(a=" + std::to_string(r.a)
                     + ", b=" + std::to_string(r.b)
                     + ", abdev=" + std::to_string(r.abdev) + ")";
            });

    // Ball Params
    py::class_<BallParams>(m, "BallParams")
            .def(py::init<>())
            .def(py::init<real, real>(), py::arg("a"), py::arg("b"))
            .def_readonly("a", &BallParams::a)
            .def_readonly("b", &BallParams::b)
            .def("__repr__", [](const BallParams& p) {
                return "BallParams(a=" + std::to_string(p.a)
                     + ", b=" + std::to_string(p.b) + ")";
            });

    // Departure Params
    py::class_<DepartureParams>(m, "DepartureParams")
            .def(py::init<>())
            .def(py::init<real, real, real>(),
                 py::arg("phi"), py::arg("eta"), py::arg("omega_sq"))
            .def_readonly("phi", &DepartureParams::phi)
            .def_readonly("eta", &DepartureParams::eta)
            .def_readonly("omega_sq", &DepartureParams::omega_sq)
            .def("__repr__", [](const DepartureParams& p) {
                return "DepartureParams(phi=" + std::to_string(p.phi)
                     + ", eta=" + std::to_string(p.eta)
                     + ", omega_sq=" + std::to_string(p.omega_sq) + ")";
            });

    // FitParams
    py::class_<FitParams>(m, "FitParams")
            .def(py::init<>())
            .def(py::init<BallParams, DepartureParams>(),
                 py::arg("ball_params"), py::arg("dep_params"))
            .def_readonly("ball_params", &FitParams::ball_params)
            .def_readonly("dep_params", &FitParams::dep_params)
            .def("__repr__", [](const FitParams& p) {
                return "FitParams(a=" + std::to_string(p.ball_params.a)
                     + ", b=" + std::to_string(p.ball_params.b)
                     + ", phi=" + std::to_string(p.dep_params.phi)
                     + ", eta=" + std::to_string(p.dep_params.eta)
                     + ", omega_sq=" + std::to_string(p.dep_params.omega_sq) + ")";
            });
    
    // A B Parameter Fitter
    m.def("fit_ab", &fit_ab,
            py::arg("tk")
        );

    // A B Parameter Refinement
    m.def("refine_ab", &refine_ab,
            py::arg("ball_params"),
            py::arg("tk")
        );

    // Departure Parameter Fitter
    m.def("fit_departure", &fit_departure,
            py::arg("To"),
            py::arg("theta"),
            py::arg("ball_params")
        );
    m.def("fit_departure_perspin", &fit_departure_perspin,
            py::arg("tks"),
            py::arg("To"),
            py::arg("theta"),
            py::arg("ball_params")
        );

    // Departure Objective Factory
    // Returns a callable phi -> LinRegResult. The factory copies everything the
    // closure needs, so the vectors pybind11 materialises for To/theta_f do not
    // need to outlive this call.
    m.def("make_departure_objective",
            [](real a, real b, const std::vector<real>& To,
               const std::vector<real>& theta_f)
            {
                return std::function<LinRegResult(real)>(
                        stator::physics::make_departure_objective(
                                a, b, To, theta_f));
            },
            py::arg("a"),
            py::arg("b"),
            py::arg("To"),
            py::arg("theta_f")
        );

    // Per-spin overload of the same factory: one (a, b) per spin rather than
    // one for the set. This is the objective fit_departure_perspin searches;
    // binding it is what lets the phi landscape be observed point by point,
    // the way the scalar one already can be. Registered second, so a scalar
    // call still matches the scalar overload first.
    m.def("make_departure_objective",
            [](const std::vector<real>& a, const std::vector<real>& b,
               const std::vector<real>& To, const std::vector<real>& theta_f)
            {
                if (a.size() != To.size() || b.size() != To.size())
                    throw std::invalid_argument(
                            "make_departure_objective(): a, b, To and theta_f "
                            "must all be the same length");
                return std::function<LinRegResult(real)>(
                        stator::physics::make_departure_objective(
                                a, b, To, theta_f));
            },
            py::arg("a"),
            py::arg("b"),
            py::arg("To"),
            py::arg("theta_f")
        );

    // Theta Predictor
    m.def("predict_theta", &predict_theta,
            py::arg("To"),
            py::arg("params")
        );

    // // // Ball Timing Model
    // // m.def("ball_timing_model", &ball_timing_model,
    // //   py::arg("k"),
    // //   py::arg("params")
    // // );
    
    // // Ball Timing Fitter
    // m.def("fit_ball_timings", &fit_ball_timings,
    //   py::arg("t_k"),
    //   py::arg("sig"),
    //   py::arg("a") = a_i,
    //   py::arg("b") = b_i
    // );
}