#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <cmath>
#include <random>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iomanip>
#include <limits>
#include <ctime>

const double INF = std::numeric_limits<double>::infinity();
const double NEG_INF = -INF;

// --------------------------------------------------------------
// Helper: automatic observation bucketing
// --------------------------------------------------------------
struct RawObservation {
    std::string type;           // "positive_message", "negative_message", "neutral_message", "no_reply"
    int hour;                   // 0-23
    int reply_delay_min;        // minutes since last message
    int msg_length;             // character count
    double sentiment;           // -1 to +1
    bool she_initiated;         // true if she sent first

    std::string to_bucket() const {
        std::string type_abbr;
        if (type == "positive_message") type_abbr = "pos";
        else if (type == "negative_message") type_abbr = "neg";
        else if (type == "neutral_message") type_abbr = "neu";
        else if (type == "no_reply") type_abbr = "none";
        else type_abbr = type.substr(0, 3);

        std::string hour_bucket;
        if (hour < 6) hour_bucket = "night";
        else if (hour < 12) hour_bucket = "morning";
        else if (hour < 18) hour_bucket = "afternoon";
        else hour_bucket = "evening";

        std::string delay_bucket;
        if (reply_delay_min < 5) delay_bucket = "instant";
        else if (reply_delay_min < 60) delay_bucket = "fast";
        else if (reply_delay_min < 360) delay_bucket = "slow";
        else delay_bucket = "very_slow";

        std::string len_bucket;
        if (msg_length < 10) len_bucket = "very_short";
        else if (msg_length < 50) len_bucket = "short";
        else if (msg_length < 200) len_bucket = "medium";
        else len_bucket = "long";

        std::string sent_bucket;
        if (sentiment < -0.33) sent_bucket = "neg";
        else if (sentiment > 0.33) sent_bucket = "pos";
        else sent_bucket = "neu";

        std::string init_bucket = she_initiated ? "init" : "reply";

        return type_abbr + "_" + hour_bucket + "_" + delay_bucket + "_" +
               len_bucket + "_" + sent_bucket + "_" + init_bucket;
    }
};

// --------------------------------------------------------------
// Hidden Markov Model with automatic bucketing
// --------------------------------------------------------------
class BehaviorHMM {
private:
    std::vector<std::string> hidden_states;
    std::vector<std::string> observations;
    std::map<std::string, int> obs_to_idx;

    std::vector<std::vector<double>> trans_prob;
    std::vector<std::vector<double>> emiss_prob;
    std::vector<double> init_prob;

    std::vector<std::vector<int>> trans_counts;
    std::vector<std::vector<int>> emiss_counts;
    std::vector<int> init_counts;

    std::vector<std::vector<std::time_t>> trans_last_time;
    std::vector<std::vector<std::time_t>> emiss_last_time;
    std::vector<std::time_t> init_last_time;

    double decay_halflife_days = 30.0;
    std::mt19937 rng;
    std::map<std::string, int> state_to_idx;

    void apply_decay() {
        std::time_t now = std::time(nullptr);
        int S = hidden_states.size();
        int O = observations.size();

        for (int i = 0; i < S; ++i) {
            for (int j = 0; j < S; ++j) {
                double days_old = std::difftime(now, trans_last_time[i][j]) / 86400.0;
                double weight = std::pow(0.5, days_old / decay_halflife_days);
                if (weight < 0.01) weight = 0.0;
                trans_counts[i][j] = static_cast<int>(trans_counts[i][j] * weight);
                // FIX: always reset timestamp to prevent stale decay on next call
                trans_last_time[i][j] = now;
            }
        }

        for (int i = 0; i < S; ++i) {
            for (int j = 0; j < O; ++j) {
                if (j >= (int)emiss_counts[i].size()) continue;
                double days_old = std::difftime(now, emiss_last_time[i][j]) / 86400.0;
                double weight = std::pow(0.5, days_old / decay_halflife_days);
                if (weight < 0.01) weight = 0.0;
                emiss_counts[i][j] = static_cast<int>(emiss_counts[i][j] * weight);
                // FIX: always reset timestamp
                emiss_last_time[i][j] = now;
            }
        }

        for (int i = 0; i < S; ++i) {
            double days_old = std::difftime(now, init_last_time[i]) / 86400.0;
            double weight = std::pow(0.5, days_old / decay_halflife_days);
            if (weight < 0.01) weight = 0.0;
            init_counts[i] = static_cast<int>(init_counts[i] * weight);
            // FIX: always reset timestamp
            init_last_time[i] = now;
        }

        recompute_all_probs();
    }

    void recompute_all_probs() {
        int S = hidden_states.size();
        int O = observations.size();

        for (int i = 0; i < S; ++i) {
            double total = 0.0;
            for (int j = 0; j < S; ++j) total += trans_counts[i][j] + 1.0;
            for (int j = 0; j < S; ++j) trans_prob[i][j] = (trans_counts[i][j] + 1.0) / total;
        }

        if (O > 0) {
            for (int i = 0; i < S; ++i) {
                double total = 0.0;
                for (int j = 0; j < (int)emiss_counts[i].size(); ++j)
                    total += emiss_counts[i][j] + 1.0;
                for (int j = 0; j < (int)emiss_counts[i].size(); ++j)
                    emiss_prob[i][j] = (emiss_counts[i][j] + 1.0) / total;
            }
        }

        double init_total = 0.0;
        for (int i = 0; i < S; ++i) init_total += init_counts[i] + 1.0;
        for (int i = 0; i < S; ++i) init_prob[i] = (init_counts[i] + 1.0) / init_total;
    }

public:
    BehaviorHMM(const std::vector<std::string>& states)
        : hidden_states(states), rng(std::random_device{}())
    {
        int S = states.size();
        for (int i = 0; i < S; ++i) state_to_idx[states[i]] = i;

        observations.clear();
        obs_to_idx.clear();

        trans_counts.assign(S, std::vector<int>(S, 0));
        init_counts.assign(S, 0);
        trans_last_time.assign(S, std::vector<std::time_t>(S, std::time(nullptr)));
        init_last_time.assign(S, std::time(nullptr));

        trans_prob.assign(S, std::vector<double>(S, 0.0));
        init_prob.assign(S, 0.0);

        emiss_counts.assign(S, std::vector<int>());
        emiss_last_time.assign(S, std::vector<std::time_t>());
        emiss_prob.assign(S, std::vector<double>());

        recompute_all_probs();
    }

    // Add a new observation bucket to vocabulary; grow emission matrices for all states
    int get_or_add_observation(const std::string& bucket) {
        if (obs_to_idx.count(bucket))
            return obs_to_idx[bucket];

        int idx = observations.size();
        observations.push_back(bucket);
        obs_to_idx[bucket] = idx;

        int S = hidden_states.size();
        for (int i = 0; i < S; ++i) {
            emiss_counts[i].push_back(0);
            emiss_last_time[i].push_back(std::time(nullptr));
            emiss_prob[i].push_back(0.0);
        }

        recompute_all_probs();
        return idx;
    }

    void learn_sequence(const std::vector<std::pair<std::string, RawObservation>>& seq) {
        apply_decay();
        for (size_t t = 0; t < seq.size(); ++t) {
            const std::string& h = seq[t].first;
            std::string bucket = seq[t].second.to_bucket();
            if (h.empty() || !state_to_idx.count(h)) continue;

            int hidx = state_to_idx[h];
            int oidx = get_or_add_observation(bucket);

            if (t == 0) {
                init_counts[hidx]++;
                init_last_time[hidx] = std::time(nullptr);
            }
            if (t > 0 && !seq[t-1].first.empty() && state_to_idx.count(seq[t-1].first)) {
                int prev = state_to_idx[seq[t-1].first];
                trans_counts[prev][hidx]++;
                trans_last_time[prev][hidx] = std::time(nullptr);
            }

            emiss_counts[hidx][oidx]++;
            emiss_last_time[hidx][oidx] = std::time(nullptr);
        }
        recompute_all_probs();
    }

    std::vector<std::string> decode(const std::vector<RawObservation>& obs_seq) {
        if (obs_seq.empty()) return {};

        int T = obs_seq.size();
        int S = hidden_states.size();

        std::vector<int> obs_idx(T, -1);
        for (int t = 0; t < T; ++t) {
            std::string bucket = obs_seq[t].to_bucket();
            if (obs_to_idx.count(bucket))
                obs_idx[t] = obs_to_idx[bucket];
        }

        // Log-space Viterbi
        std::vector<std::vector<double>> log_delta(T, std::vector<double>(S, NEG_INF));
        std::vector<std::vector<int>> psi(T, std::vector<int>(S, -1));

        // Initialization
        for (int i = 0; i < S; ++i) {
            double log_emit = 0.0;
            if (obs_idx[0] >= 0 && obs_idx[0] < (int)emiss_prob[i].size())
                log_emit = log(emiss_prob[i][obs_idx[0]]);
            log_delta[0][i] = log(init_prob[i]) + log_emit;
        }

        // Recursion
        for (int t = 1; t < T; ++t) {
            for (int j = 0; j < S; ++j) {
                double log_emit = 0.0;
                if (obs_idx[t] >= 0 && obs_idx[t] < (int)emiss_prob[j].size())
                    log_emit = log(emiss_prob[j][obs_idx[t]]);

                double best = NEG_INF;
                int best_i = -1;
                for (int i = 0; i < S; ++i) {
                    double cand = log_delta[t-1][i] + log(trans_prob[i][j]);
                    if (cand > best) { best = cand; best_i = i; }
                }
                log_delta[t][j] = best + log_emit;
                psi[t][j] = best_i;
            }
        }

        // Termination
        double best_prob = NEG_INF;
        int best_last = -1;
        for (int i = 0; i < S; ++i) {
            if (log_delta[T-1][i] > best_prob) {
                best_prob = log_delta[T-1][i];
                best_last = i;
            }
        }

        // Backtrack
        std::vector<int> best_path(T);
        best_path[T-1] = best_last;
        for (int t = T-2; t >= 0; --t)
            best_path[t] = psi[t+1][best_path[t+1]];

        std::vector<std::string> result(T);
        for (int t = 0; t < T; ++t)
            result[t] = (best_path[t] >= 0) ? hidden_states[best_path[t]] : "unknown";
        return result;
    }

    // FIX: marginalize through transition matrix — P(next_obs=o) = sum_j trans[cur->j] * emiss[j][o]
    std::vector<std::pair<std::string, double>> predict_next_observation(const std::vector<RawObservation>& obs_seq) {
        std::vector<std::pair<std::string, double>> result;
        if (obs_seq.empty() || observations.empty()) return result;

        auto decoded = decode(obs_seq);
        if (decoded.empty()) return result;

        const std::string& cur_hidden = decoded.back();
        if (!state_to_idx.count(cur_hidden)) return result;

        int cur_idx = state_to_idx[cur_hidden];
        int S = hidden_states.size();

        // Marginalize: P(o) = sum over next states j of P(cur->j) * P(o|j)
        std::vector<double> obs_dist(observations.size(), 0.0);
        for (int j = 0; j < S; ++j) {
            for (size_t o = 0; o < observations.size(); ++o) {
                if (o < emiss_prob[j].size())
                    obs_dist[o] += trans_prob[cur_idx][j] * emiss_prob[j][o];
            }
        }

        for (size_t o = 0; o < observations.size(); ++o)
            result.push_back({observations[o], obs_dist[o]});

        std::sort(result.begin(), result.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });
        return result;
    }

    void print_info() const {
        std::cout << "Hidden states: " << hidden_states.size() << "\n";
        std::cout << "Observation buckets: " << observations.size() << "\n";
        for (size_t i = 0; i < std::min((size_t)5, observations.size()); ++i)
            std::cout << "  " << observations[i] << "\n";
        if (observations.size() > 5) std::cout << "  ...\n";
    }
};

// --------------------------------------------------------------
// Main
// --------------------------------------------------------------
int main() {
    std::cout << "=== HMM with Automatic Bucketing ===\n\n";

    std::vector<std::string> hidden = {"happy", "neutral", "frustrated"};
    BehaviorHMM hmm(hidden);

    std::vector<std::pair<std::string, RawObservation>> training = {
        {"happy",      {"positive_message", 14,   2, 120,  0.9, true}},
        {"happy",      {"positive_message", 15,   5,  95,  0.8, false}},
        {"neutral",    {"neutral_message",  10,  45,  20,  0.1, false}},
        {"frustrated", {"negative_message", 22, 180,   5, -0.7, false}},
        {"frustrated", {"no_reply",         23, 720,   0,  0.0, false}},
        {"neutral",    {"positive_message",  9,  60,  30,  0.4, true}}
    };

    std::cout << "Training on " << training.size() << " examples...\n";
    hmm.learn_sequence(training);
    hmm.print_info();

    std::vector<RawObservation> recent = {
        {"positive_message", 14, 3, 110, 0.85, true},
        {"positive_message", 15, 4, 105, 0.82, false}
    };

    std::cout << "\n--- Next Observation Prediction ---\n";
    auto pred = hmm.predict_next_observation(recent);
    std::cout << "Top 5 predictions:\n";
    for (size_t i = 0; i < std::min((size_t)5, pred.size()); ++i)
        std::cout << "  " << std::fixed << std::setprecision(3)
                  << pred[i].second << " : " << pred[i].first << "\n";

    std::cout << "\n--- Decoding Hidden States ---\n";
    auto hidden_seq = hmm.decode(recent);
    std::cout << "Inferred: ";
    for (auto& h : hidden_seq) std::cout << h << " ";
    std::cout << "\n\n✅ Model ready!\n";

    return 0;
}