#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <optional>
#include <queue>
#include <regex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

// ---------- Utilities ----------
static inline std::string now_str() {
    using namespace std::chrono;
    auto tp = system_clock::now();
    auto t  = system_clock::to_time_t(tp);
    std::tm bt{};
#if defined(_WIN32)
    localtime_s(&bt, &t);
#else
    localtime_r(&t, &bt);
#endif
    std::ostringstream os;
    os << std::put_time(&bt, "%H:%M:%S");
    return os.str();
}

template<class T>
class BoundedQueue {
public:
    explicit BoundedQueue(std::size_t cap) : cap_(cap) {}

    // Blocks if full (simple backpressure)
    void push(T item) {
        std::unique_lock<std::mutex> lk(mu_);
        cv_not_full_.wait(lk, [&]{ return closed_ || q_.size() < cap_; });
        if (closed_) return;
        q_.push(std::move(item));
        cv_not_empty_.notify_one();
    }

    // Returns nullopt on closed & drained
    std::optional<T> pop() {
        std::unique_lock<std::mutex> lk(mu_);
        cv_not_empty_.wait(lk, [&]{ return closed_ || !q_.empty(); });
        if (q_.empty()) return std::nullopt;
        T v = std::move(q_.front());
        q_.pop();
        cv_not_full_.notify_one();
        return v;
    }

    void close() {
        std::lock_guard<std::mutex> lk(mu_);
        closed_ = true;
        cv_not_empty_.notify_all();
        cv_not_full_.notify_all();
    }

private:
    std::mutex mu_;
    std::condition_variable cv_not_empty_, cv_not_full_;
    std::queue<T> q_;
    std::size_t cap_;
    bool closed_{false};
};

// ---------- Reactive core (tiny Rx-like) ----------
template<typename T>
class Observable;

template<typename T>
class Subscription {
public:
    Subscription() = default;
    Subscription(std::weak_ptr<void> source, std::size_t id, std::function<void(std::size_t)> on_unsub)
        : source_(std::move(source)), id_(id), on_unsub_(std::move(on_unsub)) {}
    Subscription(const Subscription&) = delete;
    Subscription& operator=(const Subscription&) = delete;
    Subscription(Subscription&& o) noexcept { *this = std::move(o); }
    Subscription& operator=(Subscription&& o) noexcept {
        if (this != &o) {
            unsubscribe();
            source_   = std::move(o.source_);
            id_       = o.id_;
            on_unsub_ = std::move(o.on_unsub_);
            o.id_     = 0;
        }
        return *this;
    }
    ~Subscription() { unsubscribe(); }

    void unsubscribe() {
        if (id_ && !source_.expired() && on_unsub_) {
            on_unsub_(id_);
            id_ = 0;
        }
    }
private:
    std::weak_ptr<void> source_;
    std::size_t id_{0};
    std::function<void(std::size_t)> on_unsub_;
};

template<typename T>
class Observable : public std::enable_shared_from_this<Observable<T>> {
public:
    using Fn = std::function<void(const T&)>;

    Subscription<T> subscribe(Fn fn) {
        std::lock_guard<std::mutex> lk(mu_);
        std::size_t id = ++next_id_;
        subs_.emplace(id, std::move(fn));
        auto self = this->shared_from_this();
        return Subscription<T>(self, id, [this,self](std::size_t sid){
            std::lock_guard<std::mutex> lk(mu_);
            subs_.erase(sid);
        });
    }

    // Operators
    template<typename U>
    std::shared_ptr<Observable<U>> map(std::function<U(const T&)> proj) {
        auto out = std::make_shared<Observable<U>>();
        this->subscribe([weak=out, proj=std::move(proj)](const T& v){
            if (auto s = weak.lock()) s->emit(proj(v));
        });
        return out;
    }

    std::shared_ptr<Observable<T>> filter(std::function<bool(const T&)> pred) {
        auto out = std::make_shared<Observable<T>>();
        this->subscribe([weak=out, pred=std::move(pred)](const T& v){
            if (pred(v)) if (auto s = weak.lock()) s->emit(v);
        });
        return out;
    }

    std::shared_ptr<Observable<T>> throttle(std::chrono::milliseconds dur) {
        auto out = std::make_shared<Observable<T>>();
        auto last = std::make_shared<std::chrono::steady_clock::time_point>(std::chrono::steady_clock::now() - dur);
        this->subscribe([weak=out, last, dur](const T& v){
            auto now = std::chrono::steady_clock::now();
            if (now - *last >= dur) {
                *last = now;
                if (auto s = weak.lock()) s->emit(v);
            }
        });
        return out;
    }

    // For sources
    void emit(const T& v) {
        std::unordered_map<std::size_t, Fn> copy;
        {
            std::lock_guard<std::mutex> lk(mu_);
            copy.reserve(subs_.size());
            for (auto& kv : subs_) copy.emplace(kv.first, kv.second);
        }
        for (auto& kv : copy) kv.second(v);
    }

private:
    std::mutex mu_;
    std::unordered_map<std::size_t, Fn> subs_;
    std::size_t next_id_{0};
};

// ---------- Event bus over a bounded queue ----------
struct Event { std::string text; };

class ReactiveBus {
public:
    ReactiveBus(std::size_t cap=64)
        : queue_(cap), source_(std::make_shared<Observable<Event>>()) {
        worker_ = std::thread([this]{
            while (true) {
                auto item = queue_.pop();
                if (!item) break;                 // closed & drained
                source_->emit(*item);
            }
        });
    }
    ~ReactiveBus() {
        stop();
    }

    void on_next(Event e) { queue_.push(std::move(e)); }
    void stop() {
        if (!stopped_.exchange(true)) {
            queue_.close();
            if (worker_.joinable()) worker_.join();
        }
    }

    std::shared_ptr<Observable<Event>> observable() { return source_; }

private:
    std::atomic<bool> stopped_{false};
    BoundedQueue<Event> queue_;
    std::shared_ptr<Observable<Event>> source_;
    std::thread worker_;
};

// ---------- Domain: image + caption ----------
struct Image {
    std::string name; // in real life: raw pixels / cv::Mat / tensor
};

static std::string image_to_text_description(const Image& img) {
    // Placeholder for CNN/Transformer captioning
    (void)img;
    return "This is an image of a cute golden retriever playing in the park.";
}

static Image load_image(const std::string& path) {
    std::cout << "[" << now_str() << "] [Load] " << path << "\n";
    return Image{path};
}

// ---------- Command handling ----------
enum class CmdType { Load, Describe, Quit, Unknown };
struct Command {
    CmdType type{CmdType::Unknown};
    std::string arg;
    static Command parse(const std::string& s) {
        static const std::regex load_re(R"(^\s*load\s+(.+)\s*$)", std::regex::icase);
        if (std::regex_match(s, std::regex(R"(^\s*quit\s*$)", std::regex::icase))) {
            return {CmdType::Quit, {}};
        }
        if (std::regex_match(s, std::regex(R"(^\s*describe\s*$)", std::regex::icase))) {
            return {CmdType::Describe, {}};
        }
        std::smatch m;
        if (std::regex_match(s, m, load_re)) {
            return {CmdType::Load, m[1].str()};
        }
        return {CmdType::Unknown, s};
    }
};

// ---------- Demo: simulate a producer ----------
static void simulate_user_input(ReactiveBus& bus) {
    using namespace std::chrono_literals;
    std::this_thread::sleep_for(200ms);
    bus.on_next(Event{"load dog.jpg"});
    std::this_thread::sleep_for(150ms);
    bus.on_next(Event{"describe"});
    std::this_thread::sleep_for(150ms);
    bus.on_next(Event{"load cat.png"});
    std::this_thread::sleep_for(50ms);
    bus.on_next(Event{"describe"});
    std::this_thread::sleep_for(100ms);
    bus.on_next(Event{"quit"});
}

// ---------- Main ----------
int main() {
    std::atomic<bool> exit_flag{false};
    ReactiveBus bus(32);
    Image current;

    auto events = bus.observable();

    // Pipeline: Event -> string -> Command (with simple throttle)
    auto commands = events
        ->map<std::string>([](const Event& e){ return e.text; })
        ->throttle(std::chrono::milliseconds(20))
        ->map<Command>([](const std::string& s){ return Command::parse(s); });

    // Subscriptions
    auto sub_load = commands
        ->filter([](const Command& c){ return c.type == CmdType::Load; })
        ->subscribe([&](const Command& c){
            current = load_image(c.arg);
        });

    auto sub_desc = commands
        ->filter([](const Command& c){ return c.type == CmdType::Describe; })
        ->subscribe([&](const Command&){
            if (current.name.empty()) {
                std::cout << "[" << now_str() << "] [Warn] No image loaded.\n";
            } else {
                auto caption = image_to_text_description(current);
                std::cout << "[" << now_str() << "] [Caption] " << caption << "\n";
            }
        });

    auto sub_quit = commands
        ->filter([](const Command& c){ return c.type == CmdType::Quit; })
        ->subscribe([&](const Command&){
            std::cout << "[" << now_str() << "] [Exit] Program terminating.\n";
            exit_flag = true;
            bus.stop(); // stop dispatch thread cleanly
        });

    auto sub_unknown = commands
        ->filter([](const Command& c){ return c.type == CmdType::Unknown; })
        ->subscribe([](const Command& c){
            std::cout << "[" << now_str() << "] [Warn] Unknown command: " << c.arg << "\n";
        });

    // Producer thread
    std::thread producer(simulate_user_input, std::ref(bus));

    // Wait for quit (no busy loop)
    while (!exit_flag.load(std::memory_order_acquire)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    if (producer.joinable()) producer.join();

    // Unsubscribe explicitly (optional, RAII would also handle it)
    sub_load.unsubscribe();
    sub_desc.unsubscribe();
    sub_quit.unsubscribe();
    sub_unknown.unsubscribe();

    return 0;
}
