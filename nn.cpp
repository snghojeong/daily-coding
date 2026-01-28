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

/**
 * @brief Thread-Safe Queue (차단형 큐)
 * 생산자-소비자 패턴에서 데이터 전달 통로 역할을 하며, 큐의 크기 제한(Capacity)을 지원합니다.
 */
template<class T>
class TSQueue {
public:
    explicit TSQueue(std::size_t cap = 32) : cap_(cap) {}

    // 데이터를 큐에 삽입 (큐가 가득 찼으면 대기)
    void push(T v) {
        std::unique_lock<std::mutex> lk(mu_);
        cv_not_full_.wait(lk, [&] { return closed_ || q_.size() < cap_; });
        if (closed_) return;
        q_.push(std::move(v));
        cv_not_empty_.notify_one(); // 데이터를 기다리는 팝 스레드에 알림
    }

    // 데이터를 큐에서 추출 (큐가 비었으면 대기)
    std::optional<T> pop() {
        std::unique_lock<std::mutex> lk(mu_);
        cv_not_empty_.wait(lk, [&] { return closed_ || !q_.empty(); });
        if (q_.empty()) return std::nullopt; // 닫힌 상태에서 데이터가 없으면 종료
        T v = std::move(q_.front()); q_.pop();
        cv_not_full_.notify_one(); // 공간을 기다리는 푸시 스레드에 알림
        return v;
    }

    // 큐를 닫고 대기 중인 모든 스레드를 해제
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

/**
 * @brief Observable 클래스
 * 데이터 스트림을 생성하고 연산(Map, Filter, Throttle)을 통해 변형 및 전달합니다.
 */
template<typename T>
class Observable : public std::enable_shared_from_this<Observable<T>> {
public:
    using Fn = std::function<void(const T&)>;
    using Unsub = std::function<void()>;

    // 구독자 등록 및 구독 취소 함수 반환
    Unsub subscribe(Fn fn) {
        std::lock_guard<std::mutex> lk(mu_);
        auto id = ++next_id_;
        subs_[id] = std::move(fn);
        std::weak_ptr<Observable> self = this->shared_from_this();
        return [self, id] {
            if (auto s = self.lock()) {
                std::lock_guard<std::mutex> lk(s->mu_);
                s->subs_.erase(id);
            }
        };
    }

    // 데이터 타입 변환 (T -> U)
    template<typename U>
    std::shared_ptr<Observable<U>> map(std::function<U(const T&)> f) {
        auto out = std::make_shared<Observable<U>>();
        this->subscribe([w = std::weak_ptr(out), f](const T& v) {
            if (auto s = w.lock()) s->emit(f(v));
        });
        return out;
    }

    // 조건에 맞는 데이터만 통과
    std::shared_ptr<Observable<T>> filter(std::function<bool(const T&)> p) {
        auto out = std::make_shared<Observable<T>>();
        this->subscribe([w = std::weak_ptr(out), p](const T& v) {
            if (p(v)) if (auto s = w.lock()) s->emit(v);
        });
        return out;
    }

    // 지정된 시간 간격 내의 첫 번째 데이터만 통과 (과도한 이벤트 발생 방지)
    std::shared_ptr<Observable<T>> throttle(std::chrono::milliseconds d) {
        auto out = std::make_shared<Observable<T>>();
        auto last = std::make_shared<std::chrono::steady_clock::time_point>(std::chrono::steady_clock::now() - d);
        this->subscribe([w = std::weak_ptr(out), last, d](const T& v) {
            auto now = std::chrono::steady_clock::now();
            if (now - *last >= d) {
                *last = now;
                if (auto s = w.lock()) s->emit(v);
            }
        });
        return out;
    }

    // 등록된 모든 구독자에게 데이터 전파
    void emit(const T& v) {
        std::unordered_map<std::size_t, Fn> copy;
        {
            std::lock_guard<std::mutex> lk(mu_);
            copy = subs_; // 락 범위를 최소화하기 위해 복사 후 실행
        }
        for (auto& kv : copy) kv.second(v);
    }

private:
    std::mutex mu_;
    std::unordered_map<std::size_t, Fn> subs_;
    std::size_t next_id_{0};
};

// 기본 이벤트 구조체
struct Event { std::string text; };

/**
 * @brief ReactiveBus
 * 큐에 쌓인 이벤트를 백그라운드 스레드에서 꺼내 Observable 스트림으로 배포합니다.
 */
class ReactiveBus {
public:
    explicit ReactiveBus(std::size_t cap = 64) : q_(cap), src_(std::make_shared<Observable<Event>>()) {
        worker_ = std::thread([this] {
            while (auto e = q_.pop()) { src_->emit(*e); }
        });
    }

    ~ReactiveBus() { stop(); }

    void on_next(Event e) { q_.push(std::move(e)); }

    void stop() {
        if (!stopped_.exchange(true)) {
            q_.close();
            if (worker_.joinable()) worker_.join();
        }
    }

    std::shared_ptr<Observable<Event>> observable() { return src_; }

private:
    std::atomic<bool> stopped_{false};
    TSQueue<Event> q_;
    std::shared_ptr<Observable<Event>> src_;
    std::thread worker_;
};

// 도메인 모델 및 도우미 함수
struct Image { std::string path; };
static Image load_image(const std::string& p) { std::cout << "[Load] " << p << "\n"; return Image{p}; }
static std::string image_to_text(const Image&) { return "A cute golden retriever playing in the park."; }

enum class CmdType { Load, Describe, Quit, Unknown };
struct Command {
    CmdType type{CmdType::Unknown};
    std::string arg;

    // 문자열로부터 명령어 파싱 (Regex 활용)
    static Command parse(const std::string& s) {
        static const std::regex load_re(R"(^\s*load\s+(.+)\s*$)", std::regex::icase);
        static const std::regex quit_re(R"(^\s*quit\s*$)", std::regex::icase);
        static const std::regex desc_re(R"(^\s*describe\s*$)", std::regex::icase);

        if (std::regex_match(s, quit_re)) return {CmdType::Quit, {}};
        if (std::regex_match(s, desc_re)) return {CmdType::Describe, {}};
        std::smatch m;
        if (std::regex_match(s, m, load_re)) return {CmdType::Load, m[1].str()};
        return {CmdType::Unknown, s};
    }
};

// 사용자 입력을 모방하는 시뮬레이션 함수
static void simulate_input(ReactiveBus& bus) {
    using namespace std::chrono_literals;
    std::this_thread::sleep_for(1000ms); bus.on_next({"load dog.jpg"});
    std::this_thread::sleep_for(500ms);  bus.on_next({"describe"});
    std::this_thread::sleep_for(500ms);  bus.on_next({"load cat.png"});
    std::this_thread::sleep_for(200ms);  bus.on_next({"describe"});
    std::this_thread::sleep_for(200ms);  bus.on_next({"quit"});
}

int main() {
    ReactiveBus bus(128);
    Image cur; // 현재 로드된 이미지 상태
    std::atomic<bool> exit_flag{false};

    // 1. 이벤트 처리 파이프라인 구성
    auto commands = bus.observable()
        ->map<std::string>([](const Event& e) { return e.text; }) // Event -> String
        ->throttle(std::chrono::milliseconds(10))                 // 과부하 방지
        ->map<Command>([](const std::string& s) { return Command::parse(s); }); // String -> Command Object

    // 2. 각 명령어별 핸들러 등록 (Filter + Subscribe)
    auto un1 = commands->filter([](auto& c) { return c.type == CmdType::Load; })
        ->subscribe([&](auto& c) { cur = load_image(c.arg); });

    auto un2 = commands->filter([](auto& c) { return c.type == CmdType::Describe; })
        ->subscribe([&](auto&) {
            if (cur.path.empty()) std::cout << "[Warn] No image.\n"; // cur.name -> cur.path로 수정
            else std::cout << "[Caption] " << image_to_text(cur) << "\n";
        });

    auto un3 = commands->filter([](auto& c) { return c.type == CmdType::Quit; })
        ->subscribe([&](auto&) {
            std::cout << "[Exit]\n";
            exit_flag = true;
            bus.stop();
        });

    auto un4 = commands->filter([](auto& c) { return c.type == CmdType::Unknown; })
        ->subscribe([](auto& c) { std::cout << "[Warn] Unknown command: " << c.arg << "\n"; });

    // 3. 입력 스레드 시작 및 대기
    std::thread producer(simulate_input, std::ref(bus));

    while (!exit_flag.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    if (producer.joinable()) producer.join();

    // 4. 구독 해제 (리소스 정리)
    un1(); un2(); un3(); un4();

    return 0;
}
