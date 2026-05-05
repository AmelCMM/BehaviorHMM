#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <cmath>
#include <random>
#include <algorithm>
#include <limits>
#include <ctime>
#include <iomanip>

const double NEG_INF = -1e100;
const double EPS = 1e-9;

// ------------------------------------------------------
// RAW OBSERVATION -> FEATURE BUCKET ENGINE
// ------------------------------------------------------
struct RawObservation {
    std::string type;   // positive_message, negative_message, neutral_message, no_reply
    int hour;
    int reply_delay_min;
    int msg_length;
    double sentiment;
    bool she_initiated;

    std::string to_bucket() const {
        std::string t =
            (type == "positive_message") ? "pos" :
            (type == "negative_message") ? "neg" :
            (type == "neutral_message") ? "neu" : "none";

        std::string h =
            (hour < 6) ? "night" :
            (hour < 12) ? "morning" :
            (hour < 18) ? "afternoon" : "evening";

        std::string d =
            (reply_delay_min < 5) ? "instant" :
            (reply_delay_min < 60) ? "fast" :
            (reply_delay_min < 360) ? "slow" : "very_slow";

        std::string l =
            (msg_length < 10) ? "very_short" :
            (msg_length < 50) ? "short" :
            (msg_length < 200) ? "medium" : "long";

        std::string s =
            (sentiment < -0.33) ? "neg" :
            (sentiment > 0.33) ? "pos" : "neu";

        std::string i = she_initiated ? "init" : "reply";

        return t + "_" + h + "_" + d + "_" + l + "_" + s + "_" + i;
    }
};

// ------------------------------------------------------
// FEATURE-ENHANCED HMM 
// ------------------------------------------------------
class BehaviorHMM {
private:
    std::vector<std::string> states;
    std::vector<std::string> observations;

    std::map<std::string, int> state_idx;
    std::map<std::string, int> obs_idx;

    std::vector<std::vector<double>> trans;
    std::vector<std::vector<double>> emit;
    std::vector<double> start;

    // Counters (use double for smoothing, but could be int)
    std::vector<std::vector<double>> trans_count;
    std::vector<std::vector<double>> emit_count;
    std::vector<double> start_count;

    std::mt19937 rng{std::random_device{}()};

    double log_safe(double x) const {
        return (x <= 0.0) ? NEG_INF : std::log(x);
    }

public:
    BehaviorHMM(const std::vector<std::string>& s)
        : states(s) {
        int S = states.size();

        for (int i = 0; i < S; i++)
            state_idx[states[i]] = i;

        // Probability matrices
        start.assign(S, 0.0);
        trans.assign(S, std::vector<double>(S, 0.0));
        emit.assign(S, std::vector<double>());      // rows exist, columns added later

        // Count matrices
        start_count.assign(S, 0.0);
        trans_count.assign(S, std::vector<double>(S, 0.0));
        emit_count.assign(S, std::vector<double>()); // rows exist, columns added later
    }

    // --------------------------------------------------
    // Dynamically add new observation bucket
    // --------------------------------------------------
    int get_obs(const std::string& o) {
        if (obs_idx.count(o))
            return obs_idx[o];

        int id = observations.size();
        observations.push_back(o);
        obs_idx[o] = id;

        int S = states.size();

        // Add a new column for every state
        for (int i = 0; i < S; i++) {
            emit[i].push_back(0.0);
            emit_count[i].push_back(0.0);
        }

        return id;
    }

    // --------------------------------------------------
    // TRAINING (count accumulation)
    // --------------------------------------------------
    void learn(const std::vector<std::pair<std::string, RawObservation>>& seq) {
        for (size_t t = 0; t < seq.size(); t++) {
            if (!state_idx.count(seq[t].first))
                continue;

            int s = state_idx[seq[t].first];
            int o = get_obs(seq[t].second.to_bucket());

            if (t == 0)
                start_count[s] += 1.0;

            if (t > 0 && state_idx.count(seq[t-1].first)) {
                int ps = state_idx[seq[t-1].first];
                trans_count[ps][s] += 1.0;
            }

            emit_count[s][o] += 1.0;
        }

        rebuild();
    }

    // --------------------------------------------------
    // NORMALISATION (with Laplace smoothing)
    // --------------------------------------------------
    void rebuild() {
        int S = states.size();
        int O = observations.size();

        // Start probabilities
        double st_sum = 0.0;
        for (double x : start_count) st_sum += x + 1.0;   // +1 smoothing
        for (int i = 0; i < S; i++)
            start[i] = (start_count[i] + 1.0) / st_sum;

        // Transition probabilities
        for (int i = 0; i < S; i++) {
            double row_sum = 0.0;
            for (int j = 0; j < S; j++)
                row_sum += trans_count[i][j] + 1.0;
            for (int j = 0; j < S; j++)
                trans[i][j] = (trans_count[i][j] + 1.0) / row_sum;
        }

        // Emission probabilities
        if (O > 0) {
            for (int i = 0; i < S; i++) {
                double row_sum = 0.0;
                for (int j = 0; j < O; j++)
                    row_sum += emit_count[i][j] + 1.0;
                for (int j = 0; j < O; j++)
                    emit[i][j] = (emit_count[i][j] + 1.0) / row_sum;
            }
        }
    }

    // --------------------------------------------------
    // VITERBI (log‑space, stable)
    // --------------------------------------------------
    std::vector<std::string> decode(const std::vector<RawObservation>& obs) {
        int T = obs.size();
        int S = states.size();
        if (T == 0) return {};

        // Convert observations to indices
        std::vector<int> oidx(T);
        for (int t = 0; t < T; t++) {
            std::string b = obs[t].to_bucket();
            if (!obs_idx.count(b)) return {};   // unknown observation – cannot decode
            oidx[t] = obs_idx[b];
        }

        // log‑delta and backpointers
        std::vector<std::vector<double>> dp(T, std::vector<double>(S, NEG_INF));
        std::vector<std::vector<int>> back(T, std::vector<int>(S, -1));

        // Initialisation
        for (int i = 0; i < S; i++) {
            dp[0][i] = log_safe(start[i] + 1e-12) +
                       log_safe(emit[i][oidx[0]] + 1e-12);
        }

        // Recursion
        for (int t = 1; t < T; t++) {
            for (int j = 0; j < S; j++) {
                double best = NEG_INF;
                int best_i = -1;
                for (int i = 0; i < S; i++) {
                    double val = dp[t-1][i] + log_safe(trans[i][j] + 1e-12);
                    if (val > best) {
                        best = val;
                        best_i = i;
                    }
                }
                dp[t][j] = best + log_safe(emit[j][oidx[t]] + 1e-12);
                back[t][j] = best_i;
            }
        }

        // Termination
        double best_prob = NEG_INF;
        int last = -1;
        for (int i = 0; i < S; i++) {
            if (dp[T-1][i] > best_prob) {
                best_prob = dp[T-1][i];
                last = i;
            }
        }

        // Backtrack
        std::vector<int> path(T);
        path[T-1] = last;
        for (int t = T-2; t >= 0; t--) {
            path[t] = back[t+1][path[t+1]];
        }

        // Convert to state names
        std::vector<std::string> result(T);
        for (int t = 0; t < T; t++)
            result[t] = states[path[t]];

        return result;
    }

    // --------------------------------------------------
    // Prediction of next hidden state distribution
    // --------------------------------------------------
    std::vector<double> predict_next_state(const std::vector<RawObservation>& obs) {
        auto decoded = decode(obs);
        if (decoded.empty() || !state_idx.count(decoded.back()))
            return std::vector<double>(states.size(), 1.0 / states.size());
        int cur = state_idx[decoded.back()];
        return trans[cur];
    }

    // --------------------------------------------------
    // Predict next observation (what will she do?)
    // --------------------------------------------------
    std::vector<std::pair<std::string, double>> predict_next_observation(const std::vector<RawObservation>& obs) {
        std::vector<std::pair<std::string, double>> result;
        if (observations.empty()) return result;

        auto hidden_dist = predict_next_state(obs);
        std::vector<double> obs_dist(observations.size(), 0.0);

        for (size_t h = 0; h < states.size(); h++) {
            for (size_t o = 0; o < observations.size(); o++) {
                obs_dist[o] += hidden_dist[h] * emit[h][o];
            }
        }

        for (size_t o = 0; o < observations.size(); o++)
            result.push_back({observations[o], obs_dist[o]});

        std::sort(result.begin(), result.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });
        return result;
    }

    // --------------------------------------------------
    // Debug print
    // --------------------------------------------------
    void print() const {
        std::cout << "Hidden states: " << states.size() << "\n";
        std::cout << "Observation bucket types: " << observations.size() << "\n";
        if (!observations.empty()) {
            std::cout << "Example buckets:\n";
            for (size_t i = 0; i < std::min((size_t)5, observations.size()); i++)
                std::cout << "  " << observations[i] << "\n";
        }
    }
};

// ------------------------------------------------------
// MAIN
// ------------------------------------------------------
int main() {
    std::cout << "=== Feature-Enhanced HMM (C++17, no external libs) ===\n\n";

    BehaviorHMM hmm({"happy", "neutral", "frustrated"});

    // Training data (labeled hidden states + raw observations)
    std::vector<std::pair<std::string, RawObservation>> data = {
        {"happy",      {"positive_message", 14, 2, 120, 0.9, true}},
        {"happy",      {"positive_message", 15, 5,  90, 0.8, false}},
        {"neutral",    {"neutral_message",  10, 40, 20, 0.1, false}},
        {"frustrated", {"negative_message", 22, 180, 5, -0.7, false}},
        {"frustrated", {"no_reply",         23, 700, 0,  0.0, false}},
        {"neutral",    {"positive_message",  9, 60, 30, 0.3, true}}
    };

    std::cout << "Training on " << data.size() << " labeled steps...\n";
    hmm.learn(data);
    hmm.print();

    // Test sequence (only raw observations)
    std::vector<RawObservation> test = {
        {"positive_message", 14, 3, 110, 0.85, true},
        {"positive_message", 15, 4, 100, 0.8, false}
    };

    // Decode hidden states
    auto decoded = hmm.decode(test);
    std::cout << "\nDecoded hidden states: ";
    for (auto& s : decoded) std::cout << s << " ";
    std::cout << "\n";

    // Predict next observation
    auto next_obs = hmm.predict_next_observation(test);
    std::cout << "\nPredicted next observation (top 3):\n";
    for (size_t i = 0; i < std::min((size_t)3, next_obs.size()); i++) {
        std::cout << "  " << std::fixed << std::setprecision(3) 
                  << next_obs[i].second << " : " << next_obs[i].first << "\n";
    }

    std::cout << "\nEXECUTION COMPLETE, LOL :-) \n";
    return 0;
}
