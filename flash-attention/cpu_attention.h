#pragma once

#include <iostream>
#include <vector>
#include <chrono>

void cpu_attention(float const* Q, float const* K, float const* V,
    int n, int d, float* output) {
    for (int i = 0; i < n; ++i) {
        std::vector<float> scores(n, 0.0f);
        for (int j = 0; j < n; ++j) {
            float score = 0.0f;
            for (int k = 0; k < d; ++k) {
                score += Q[i * d + k] * K[j * d + k];
            }
            scores[j] = score;
        }

        float max_score = *std::max_element(scores.begin(), scores.end());
        float sum_exp = 0.0f;
        for (int j = 0; j < n; ++j) {
            scores[j] = std::exp(scores[j] - max_score);
            sum_exp += scores[j];
        }
        for (int j = 0; j < n; ++j) {
            scores[j] /= sum_exp;
        }

        for (int k = 0; k < d; ++k) {
            output[i * d + k] = 0.0f;
            for (int j = 0; j < n; ++j) {
                output[i * d + k] += scores[j] * V[j * d + k];
            }
        }
    }
}
