#pragma once

#include <vector>
#include <cmath>
#include <optional>
#include <string>

#include <stator/core/types.h>
#include <stator/core/errors.h>

#include <stator/physics/types.h>
#include <stator/physics/rim.h>
#include <stator/physics/wheel.h>

using namespace stator::core;

namespace stator::physics {

// Predictor Class
class Predictor {
public:
    Predictor(Rim rim, Wheel wheel)
        : m_rim { std::move(rim) }
        , m_wheel { std::move(wheel) }
    {}

    std::optional<Prediction> predict(std::vector<real>& ball_ts, std::vector<real>& wheel_ts, const real ball_sense, const real wheel_sense) const
    {
        // Makes a roulette prediction based on ball timings and wheel timings
        // if (ball_ts.front() < wheel_ts.back())
        //     throw InvalidArgument("Predictor.predict(): ball timestamps cannot come before wheel timestamps.");
        
        // Predict Ball Position
        std::optional<BallPrediction> ball_pred { m_rim.predict(ball_ts, ball_sense) };
        if (!ball_pred.has_value())
            return std::nullopt;
        const real ball_travel { ball_pred.value().theta };
        const real t_drop { ball_pred.value().t_f + ball_ts.back() };

        // Predict Wheel Position
        const real wheel_travel { m_wheel.predict(wheel_ts, t_drop) };

        // Get Ball Position in Wheel Frame
        real W { std::copysign(ball_travel, ball_sense) - std::copysign(wheel_travel, wheel_sense) + m_wheel.get_pkt_ang()/2.0 };

        // Pocket Prediction
        std::string_view pocket { m_wheel.get_pkt_from_angle(W) };

        return Prediction(t_drop, ball_travel, wheel_travel, W, pocket);
    }

private:
    Rim m_rim;
    Wheel m_wheel;
};

}
