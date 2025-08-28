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

// ---------- Tiny thread-safe queue ----------
template<class T>
class TSQueue {
public:
    explicit TSQueue(std::size_t cap=64) : cap_(cap) {}
    void push(T v){
        std::unique_lock<std::mutex> lk(mu_);
        cv_not_full_.wait(lk,[&]{return closed_||q_.size()<cap_;});
        if (closed_) return;
        q_.push(std::move(v));
        cv_not_empty_.notify_one();
    }
    std::optional<T> pop(){
        std::unique_lock<std::mutex> lk(mu_);
        cv_not_empty_.wait(lk,[&]{return closed_||!q_.empty();});
        if (q_.empty()) return std::nullopt;
        T v=std::move(q_.front()); q_.pop();
        cv_not_full_.notify_one();
        return v;
    }
    void close(){
        std::lock_guard<std::mutex> lk(mu_);
        closed_=true; cv_not_empty_.notify_all(); cv_not_full_.notify_all();
    }
private:
    std::mutex mu_; std::condition_variable cv_not_empty_, cv_not_full_;
    std::queue<T> q_; std::size_t cap_; bool closed_{false};
};

// ---------- Minimal Rx-like Observable ----------
template<typename T>
class Observable : public std::enable_shared_from_this<Observable<T>>{
public:
    using Fn=std::function<void(const T&)>;
    using Unsub=std::function<void()>;

    Unsub subscribe(Fn fn){
        std::lock_guard<std::mutex> lk(mu_);
        auto id=++next_id_;
        subs_[id]=std::move(fn);
        std::weak_ptr<Observable> self=this->shared_from_this();
        return [self,id]{
            if(auto s=self.lock()){ std::lock_guard<std::mutex> lk(s->mu_); s->subs_.erase(id); }
        };
    }

    template<typename U>
    std::shared_ptr<Observable<U>> map(std::function<U(const T&)> f){
        auto out=std::make_shared<Observable<U>>();
        this->subscribe([w=std::weak_ptr(out),f](const T& v){ if(auto s=w.lock()) s->emit(f(v)); });
        return out;
    }
    std::shared_ptr<Observable<T>> filter(std::function<bool(const T&)> p){
        auto out=std::make_shared<Observable<T>>();
        this->subscribe([w=std::weak_ptr(out),p](const T& v){ if(p(v)) if(auto s=w.lock()) s->emit(v); });
        return out;
    }
    std::shared_ptr<Observable<T>> throttle(std::chrono::milliseconds d){
        auto out=std::make_shared<Observable<T>>();
        auto last=std::make_shared<std::chrono::steady_clock::time_point>(std::chrono::steady_clock::now()-d);
        this->subscribe([w=std::weak_ptr(out),last,d](const T& v){
            auto now=std::chrono::steady_clock::now();
            if(now-*last>=d){ *last=now; if(auto s=w.lock()) s->emit(v); }
        });
        return out;
    }
    void emit(const T& v){
        std::unordered_map<std::size_t,Fn> copy;
        { std::lock_guard<std::mutex> lk(mu_); copy=subs_; }
        for(auto& kv:copy) kv.second(v);
    }
private:
    std::mutex mu_; std::unordered_map<std::size_t,Fn> subs_; std::size_t next_id_{0};
};

// ---------- Event bus ----------
struct Event{ std::string text; };

class ReactiveBus{
public:
    explicit ReactiveBus(std::size_t cap=64):q_(cap),src_(std::make_shared<Observable<Event>>()){
        worker_=std::thread([this]{ while(auto e=q_.pop()){ src_->emit(*e);} });
    }
    ~ReactiveBus(){ stop(); }
    void on_next(Event e){ q_.push(std::move(e)); }
    void stop(){
        if(!stopped_.exchange(true)){ q_.close(); if(worker_.joinable()) worker_.join(); }
    }
    std::shared_ptr<Observable<Event>> observable(){ return src_; }
private:
    std::atomic<bool> stopped_{false}; TSQueue<Event> q_; std::shared_ptr<Observable<Event>> src_; std::thread worker_;
};

// ---------- Domain stubs ----------
struct Image{ std::string name; };
static Image load_image(const std::string& p){ std::cout<<"[Load] "<<p<<"\n"; return Image{p}; }
static std::string image_to_text(const Image&){ return "A cute golden retriever playing in the park."; }

// ---------- Commands ----------
enum class CmdType{Load,Describe,Quit,Unknown};
struct Command{
    CmdType type{CmdType::Unknown}; std::string arg;
    static Command parse(const std::string& s){
        static const std::regex load_re(R"(^\s*load\s+(.+)\s*$)",std::regex::icase);
        if(std::regex_match(s,std::regex(R"(^\s*quit\s*$)",std::regex::icase))) return {CmdType::Quit,{}};
        if(std::regex_match(s,std::regex(R"(^\s*describe\s*$)",std::regex::icase))) return {CmdType::Describe,{}};
        std::smatch m; if(std::regex_match(s,m,load_re)) return {CmdType::Load,m[1].str()};
        return {CmdType::Unknown,s};
    }
};

// ---------- Demo producer ----------
static void simulate_input(ReactiveBus& bus){
    using namespace std::chrono_literals;
    std::this_thread::sleep_for(100ms); bus.on_next({"load dog.jpg"});
    std::this_thread::sleep_for(50ms);  bus.on_next({"describe"});
    std::this_thread::sleep_for(50ms);  bus.on_next({"load cat.png"});
    std::this_thread::sleep_for(20ms);  bus.on_next({"describe"});
    std::this_thread::sleep_for(20ms);  bus.on_next({"quit"});
}

// ---------- Main ----------
int main(){
    ReactiveBus bus(32); Image cur; std::atomic<bool> exit{false};
    auto commands = bus.observable()
        ->map<std::string>([](auto& e){return e.text;})
        ->throttle(std::chrono::milliseconds(20))
        ->map<Command>([](const std::string& s){return Command::parse(s);});

    auto un1 = commands->filter([](auto& c){return c.type==CmdType::Load;})
        ->subscribe([&](auto& c){ cur=load_image(c.arg); });
    auto un2 = commands->filter([](auto& c){return c.type==CmdType::Describe;})
        ->subscribe([&](auto&){ if(cur.name.empty()) std::cout<<"[Warn] No image.\n";
                                 else std::cout<<"[Caption] "<<image_to_text(cur)<<"\n"; });
    auto un3 = commands->filter([](auto& c){return c.type==CmdType::Quit;})
        ->subscribe([&](auto&){ std::cout<<"[Exit]\n"; exit=true; bus.stop(); });
    auto un4 = commands->filter([](auto& c){return c.type==CmdType::Unknown;})
        ->subscribe([](auto& c){ std::cout<<"[Warn] Unknown: "<<c.arg<<"\n"; });

    std::thread producer(simulate_input,std::ref(bus));
    while(!exit.load()) std::this_thread::sleep_for(std::chrono::milliseconds(10));
    if(producer.joinable()) producer.join();
    un1(); un2(); un3(); un4();
    return 0;
}
