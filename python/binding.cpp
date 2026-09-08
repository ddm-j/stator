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
#include <stator/physics/wheel.h>
#include <stator/physics/predict.h>
#include <stator/physics/predictor.h>
#include <stator/physics/timing.h>

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
                 py::arg("delta"), py::arg("eta"), py::arg("omega_sq"))
            .def_readonly("delta", &DepartureParams::delta)
            .def_readonly("eta", &DepartureParams::eta)
            .def_readonly("omega_sq", &DepartureParams::omega_sq)
            .def("__repr__", [](const DepartureParams& p) {
                return "DepartureParams(delta=" + std::to_string(p.delta)
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
                     + ", delta=" + std::to_string(p.dep_params.delta)
                     + ", eta=" + std::to_string(p.dep_params.eta)
                     + ", omega_sq=" + std::to_string(p.dep_params.omega_sq) + ")";
            });
    
    // Ball Prediction
    py::class_<BallPrediction>(m, "BallPrediction")
            .def(py::init<>())
            .def(py::init<real, real>(), py::arg("theta"), py::arg("t_f"))
            .def_readonly("theta", &BallPrediction::theta)
            .def_readonly("t_f", &BallPrediction::t_f)
            .def("__repr__", [](const BallPrediction& p) {
                return "BallPrediction(theta=" + std::to_string(p.theta)
                     + ", t_f=" + std::to_string(p.t_f) + ")";
            });

    // Prediction (what Predictor returns: the full stitched result)
    py::class_<Prediction>(m, "Prediction")
            .def(py::init<>())
            .def_readonly("departure_time", &Prediction::departure_time)
            .def_readonly("ball_travel", &Prediction::ball_travel)
            .def_readonly("wheel_travel", &Prediction::wheel_travel)
            .def_readonly("wheel_angle", &Prediction::wheel_angle)
            // pocket is a string_view into the static layout table; hand python
            // an owned str rather than a view.
            .def_property_readonly("pocket",
                 [](const Prediction& p) { return std::string(p.pocket); })
            .def("__repr__", [](const Prediction& p) {
                return "Prediction(pocket=" + std::string(p.pocket)
                     + ", departure_time=" + std::to_string(p.departure_time)
                     + ", ball_travel=" + std::to_string(p.ball_travel)
                     + ", wheel_travel=" + std::to_string(p.wheel_travel)
                     + ", wheel_angle=" + std::to_string(p.wheel_angle) + ")";
            });

    // Rim
    py::class_<Rim>(m, "Rim")
            .def(py::init<>())
            .def(py::init<idx, idx>(), py::arg("M"), py::arg("Y"))
            .def(py::init<real, real, real, real, real>(),
                 py::arg("a"), py::arg("b"), py::arg("delta"),
                 py::arg("eta"), py::arg("omega_sq"))
            .def("add_timing",
                 [](Rim& rim, std::string_view id, std::vector<real> timestamps, real theta, real s) {
                     rim.add_timing(id, timestamps, theta, s);
                 },
                 py::arg("id"), py::arg("timestamps"), py::arg("theta"), py::arg("s") = 1.0)
            .def("add_timing",
                 [](Rim& rim, std::vector<std::string> ids,
                    std::vector<std::vector<real>> timestamps, std::vector<real> thetas,
                    std::vector<real> ss) {
                     rim.add_timing(ids, timestamps, thetas, ss);
                 },
                 py::arg("ids"), py::arg("timestamps"), py::arg("thetas"),
                 py::arg("ss") = std::vector<real>{})
            .def("fit", &Rim::fit)
            .def("predict",
                 static_cast<std::optional<BallPrediction> (Rim::*)(const std::vector<real>&, const real) const>(&Rim::predict),
                 py::arg("timestamps"), py::arg("s") = 1.0)
            .def("predict",
                 static_cast<std::vector<std::optional<BallPrediction>> (Rim::*)(const std::vector<std::vector<real>>&, const std::vector<real>&) const>(&Rim::predict),
                 py::arg("timestamps"), py::arg("ss") = std::vector<real>{})
            .def_property_readonly("params", &Rim::params)
            .def_property_readonly("lap_floor", &Rim::lap_floor)
            .def_property_readonly("a_slope", &Rim::a_slope)
            .def("__repr__", [](const Rim& rim) {
                const FitParams& p { rim.params() };
                return "Rim(a=" + std::to_string(p.ball_params.a)
                     + ", b=" + std::to_string(p.ball_params.b)
                     + ", delta=" + std::to_string(p.dep_params.delta)
                     + ", eta=" + std::to_string(p.dep_params.eta)
                     + ", omega_sq=" + std::to_string(p.dep_params.omega_sq)
                     + ", lap_floor=" + std::to_string(rim.lap_floor()) + ")";
            });

    // Wheel
    // No public constructor: build one with Wheel.American() / Wheel.European().
    // predict/add_timing take non-const vector refs in C++, so the lambdas take
    // their vectors by value and pass those lvalues through.
    py::class_<Wheel>(m, "Wheel")
            .def_static("American", &Wheel::American)
            .def_static("European", &Wheel::European)
            .def("add_timing",
                 [](Wheel& wheel, std::string_view id, std::vector<real> timestamps) {
                     wheel.add_timing(id, timestamps);
                 },
                 py::arg("id"), py::arg("timestamps"))
            .def("add_timing",
                 [](Wheel& wheel, std::vector<std::string> ids,
                    std::vector<std::vector<real>> timestamps) {
                     wheel.add_timing(ids, timestamps);
                 },
                 py::arg("ids"), py::arg("timestamps"))
            .def("fit", &Wheel::fit)
            .def("predict",
                 [](const Wheel& wheel, std::vector<real> timestamps, real t_drop) {
                     return wheel.predict(timestamps, t_drop);
                 },
                 py::arg("timestamps"), py::arg("t_drop"))
            .def("get_pkt_index", &Wheel::get_pkt_index, py::arg("pocket"))
            .def("get_pkt_angle", &Wheel::get_pkt_angle, py::arg("pocket"))
            .def("get_pkt_from_angle", &Wheel::get_pkt_from_angle, py::arg("angle"))
            .def_property_readonly("n", &Wheel::get_n)
            .def_property_readonly("decay_k", &Wheel::get_decay_k)
            .def_property_readonly("pkt_ang", &Wheel::get_pkt_ang)
            .def("__repr__", [](const Wheel& wheel) {
                return "Wheel(pockets=" + std::to_string(wheel.get_n())
                     + ", decay_k=" + std::to_string(wheel.get_decay_k()) + ")";
            });

    // Predictor
    // Takes a fitted Rim and Wheel by value, so the python objects stay usable
    // and independent afterwards. Both senses are required: nothing assumes the
    // ball and rotor counter-rotate.
    py::class_<Predictor>(m, "Predictor")
            .def(py::init<Rim, Wheel>(), py::arg("rim"), py::arg("wheel"))
            .def("predict",
                 [](const Predictor& pred, std::vector<real> ball_ts,
                    std::vector<real> wheel_ts, real ball_sense, real wheel_sense) {
                     return pred.predict(ball_ts, wheel_ts, ball_sense, wheel_sense);
                 },
                 py::arg("ball_ts"), py::arg("wheel_ts"),
                 py::arg("ball_sense"), py::arg("wheel_sense"))
            .def("__repr__", [](const Predictor&) {
                return std::string("Predictor()");
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
            py::arg("s"),
            py::arg("ball_params")
        );
    m.def("fit_departure_perspin", &fit_departure_perspin,
            py::arg("tks"),
            py::arg("To"),
            py::arg("theta"),
            py::arg("s"),
            py::arg("ball_params")
        );

    // Departure Objective Factory
    // Returns a callable phi -> LinRegResult. The factory copies everything the
    // closure needs, so the vectors pybind11 materialises for To/theta_f do not
    // need to outlive this call.
    m.def("make_departure_objective",
            [](real a, real b, const std::vector<real>& To,
               const std::vector<real>& theta_f, const std::vector<real>& s)
            {
                return std::function<LinRegResult(real)>(
                        stator::physics::make_departure_objective(
                                a, b, To, theta_f, s));
            },
            py::arg("a"),
            py::arg("b"),
            py::arg("To"),
            py::arg("theta_f"),
            py::arg("s")
        );

    // Per-spin overload of the same factory: one (a, b) per spin rather than
    // one for the set. This is the objective fit_departure_perspin searches;
    // binding it is what lets the phi landscape be observed point by point,
    // the way the scalar one already can be. Registered second, so a scalar
    // call still matches the scalar overload first.
    m.def("make_departure_objective",
            [](const std::vector<real>& a, const std::vector<real>& b,
               const std::vector<real>& To, const std::vector<real>& theta_f,
               const std::vector<real>& s)
            {
                if (a.size() != To.size() || b.size() != To.size())
                    throw std::invalid_argument(
                            "make_departure_objective(): a, b, To and theta_f "
                            "must all be the same length");
                return std::function<LinRegResult(real)>(
                        stator::physics::make_departure_objective(
                                a, b, To, theta_f, s));
            },
            py::arg("a"),
            py::arg("b"),
            py::arg("To"),
            py::arg("theta_f"),
            py::arg("s")
        );

    // Theta Predictor
    m.def("predict_theta", &predict_theta,
            py::arg("To"),
            py::arg("s"),
            py::arg("params")
        );

    // Time of Fall Predictor
    m.def("predict_tf", &predict_tf,
            py::arg("To"),
            py::arg("theta"),
            py::arg("s"),
            py::arg("params")
        );

    // To Estimator and its calibration. Bound so the anchor/window sweeps can
    // be driven from python without going through Rim.
    m.def("lap_durations", &lap_durations,
            py::arg("tk")
        );
    m.def("lap_curvature", &lap_curvature,
            py::arg("dT"),
            py::arg("lap_floor")
        );
    m.def("estimate_To", &estimate_To,
            py::arg("tk"),
            py::arg("m"),
            py::arg("lap_floor"),
            py::arg("a_slope"),
            py::arg("N")
        );
    m.def("fit_lap_floor", &fit_lap_floor,
            py::arg("tks"),
            py::arg("N"),
            py::arg("N_scan") = 60
        );

    // Rotor Decay Fitter
    // Takes a list of per-spin timestamp lists. WheelTiming is an internal type,
    // so the ids it carries are synthesised here rather than exposed; fit_rotor
    // never reads them. Returns (k [rad/s^2], relative sigma of k).
    m.def("fit_rotor",
            [](const std::vector<std::vector<real>>& timestamps) {
                std::vector<WheelTiming> timings;
                timings.reserve(timestamps.size());
                for (idx j {}; j < timestamps.size(); j++)
                    timings.emplace_back(std::to_string(j), timestamps[j]);
                return stator::physics::fit_rotor(timings);
            },
            py::arg("timestamps")
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