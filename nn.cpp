#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <regex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

using namespace std::chrono_literals;

// --- Utilities ---

template<typename T>
class TSQueue {
public:
    explicit TSQueue(std::size_t cap = 32) : cap_(cap) {}

    void push(T v) {
        std::unique_lock lk(mu_);
        cv_full_.wait(lk, [&] { return closed_ || q_.size() < cap_; });
        if (closed_) return;
        q_.push(std::move(v));
        cv_empty_.notify_one();
    }

    std::optional<T> pop() {
        std::unique_lock lk(mu_);
        cv_empty_.wait(lk, [&] { return closed_ || !q_.empty(); });
        if (q_.empty()) return std::nullopt;
        T v = std::move(q_.front()); q_.pop();
        cv_full_.notify_one();
        return v;
    }

    void close() {
        { std::lock_guard lk(mu_); closed_ = true; }
        cv_empty_.notify_all();
        cv_full_.notify_all();
    }

private:
    std::mutex mu_;
    std::condition_variable cv_empty_, cv_full_;
    std::queue<T> q_;
    std::size_t cap_;
    bool closed_{false};
};

// --- Reactive Core ---

template<typename T>
class Observable : public std::enable_shared_from_this<Observable<T>> {
public:
    using Callback = std::function<void(const T&)>;
    using Unsubscribe = std::function<void()>;

    Unsubscribe subscribe(Callback cb) {
        std::lock_guard lk(mu_);
        auto id = next_id_++;
        subs_[id] = std::move(cb);
        return [self = this->weak_from_this(), id] {
            if (auto s = self.lock()) {
                std::lock_guard lk_sub(s->mu_);
                s->subs_.erase(id);
            }
        };
    }

    void emit(const T& v) {
        std::vector<Callback> active_subs;
        {
            std::lock_guard lk(mu_);
            for (auto& [id, cb] : subs_) active_subs.push_back(cb);
        }
        for (auto& cb : active_subs) cb(v);
    }

    // High-order operators
    template<typename U>
    auto map(std::function<U(const T&)> f) {
        auto out = std::make_shared<Observable<U>>();
        subscribe([w = std::weak_ptr(out), f](const T& v) {
            if (auto s = w.lock()) s->emit(f(v));
        });
        return out;
    }

    auto filter(std::function<bool(const T&)> p) {
        auto out = std::make_shared<Observable<T>>();
        subscribe([w = std::weak_ptr(out), p](const T& v) {
            if (p(v)) if (auto s = w.lock()) s->emit(v);
        });
        return out;
    }

private:
    std::mutex mu_;
    std::unordered_map<std::size_t, Callback> subs_;
    std::size_t next_id_{0};
};

// --- Domain Models ---

enum class CmdType { Load, Describe, Quit, Unknown };
struct Command {
    CmdType type;
    std::string arg;

    static Command parse(const std::string& s) {
        static const std::regex load_re(R"(^\s*load\s+(.+)$)", std::regex::icase);
        static const std::regex quit_re(R"(^\s*quit\s*$)", std::regex::icase);
        static const std::regex desc_re(R"(^\s*describe\s*$)", std::regex::icase);

        std::smatch m;
        if (std::regex_match(s, quit_re)) return {CmdType::Quit, ""};
        if (std::regex_match(s, desc_re)) return {CmdType::Describe, ""};
        if (std::regex_match(s, m, load_re)) return {CmdType::Load, m[1].str()};
        return {CmdType::Unknown, s};
    }
};

// --- Event Bus ---

class ReactiveBus {
public:
    ReactiveBus() : src_(std::make_shared<Observable<std::string>>()) {
        worker_ = std::jthread([this](std::stop_token st) {
            while (!st.stop_requested()) {
                if (auto msg = q_.pop()) src_->emit(*msg);
                else break;
            }
        });
    }

    void post(std::string msg) { q_.push(std::move(msg)); }
    void stop() { q_.close(); worker_.request_stop(); }
    auto stream() { return src_; }

private:
    TSQueue<std::string> q_;
    std::shared_ptr<Observable<std::string>> src_;
    std::jthread worker_;
};

// --- Main Execution ---

int main() {
    ReactiveBus bus;
    std::string current_image_path;
    std::vector<Observable<Command>::Unsubscribe> tokens;

    // Build Pipeline: String -> Command
    auto commands = bus.stream()->map<Command>([](auto& s) { return Command::parse(s); });

    // Define Handlers
    tokens.push_back(commands->filter([](auto& c) { return c.type == CmdType::Load; })
        ->subscribe([&](auto& c) { 
            current_image_path = c.arg;
            std::cout << "[System] Loaded: " << c.arg << "\n"; 
        }));

    tokens.push_back(commands->filter([](auto& c) { return c.type == CmdType::Describe; })
        ->subscribe([&](auto&) {
            if (current_image_path.empty()) std::cout << "[Warn] No image loaded.\n";
            else std::cout << "[Caption] Description of " << current_image_path << "\n";
        }));

    // Simulating user input
    std::jthread producer([&](std::stop_token st) {
        std::vector<std::string> inputs = {"load dog.jpg", "describe", "load cat.png", "describe", "quit"};
        for (const auto& in : inputs) {
            if (st.stop_requested()) break;
            std::this_thread::sleep_for(500ms);
            bus.post(in);
            if (in == "quit") break;
        }
    });

    // Use a simple blocking mechanism for the "Quit" event
    std::promise<void> exit_signal;
    auto exit_future = exit_signal.get_future();

    tokens.push_back(commands->filter([](auto& c) { return c.type == CmdType::Quit; })
        ->subscribe([&](auto&) { 
            std::cout << "[System] Shutting down...\n";
            exit_signal.set_value(); 
        }));

    exit_future.wait(); // Wait for the "quit" command to process
    bus.stop();

    return 0;
}
