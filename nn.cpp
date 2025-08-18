#include <iostream>
#include <functional>
#include <string>
#include <vector>
#include <thread>
#include <chrono>
#include <atomic>

// Simple reactive event source
class ReactiveSource {
public:
    void emit(const std::string& data) {
        for (auto& subscriber : subscribers) {
            subscriber(data);
        }
    }

    void subscribe(std::function<void(const std::string&)> fn) {
        subscribers.push_back(fn);
    }

private:
    std::vector<std::function<void(const std::string&)>> subscribers;
};

// Simulated image class
struct Image {
    std::string name;
    // Real-world: raw pixel data or cv::Mat or tensor
};

// Stub: Image-to-text neural network function
std::string image_to_text_description(const Image& img) {
    // Real-world: Use CNN + RNN or Transformer for captioning
    return "This is an image of a cute golden retriever playing in the park.";
}

// Simulate loading an image from disk
Image load_image(const std::string& path) {
    std::cout << "[Load] Loading image from: " << path << "\n";
    return Image{ path };
}

// Simulated user input (events)
void simulate_user_input(ReactiveSource& input_stream) {
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    input_stream.emit("load dog.jpg");
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    input_stream.emit("describe");
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    input_stream.emit("quit");
}

int main() {
    ReactiveSource input_stream;
    std::atomic<bool> exit_flag = false;
    Image current_image;

    // Pipe: "load <filename>" => load image
    input_stream.subscribe([&](const std::string& input) {
        if (input.rfind("load ", 0) == 0) {
            std::string filename = input.substr(5);
            current_image = load_image(filename);
        }
    });

    // Pipe: "describe" => describe current image
    input_stream.subscribe([&](const std::string& input) {
        if (input == "describe") {
            if (current_image.name.empty()) {
                std::cout << "[Warn] No image loaded.\n";
            } else {
                std::string description = image_to_text_description(current_image);
                std::cout << "[Caption] " << description << "\n";
            }
        }
    });

    // Pipe: "quit" => exit
    input_stream.subscribe([&](const std::string& input) {
        if (input == "quit") {
            std::cout << "[Exit] Program terminating.\n";
            exit_flag = true;
        }
    });

    std::thread input_thread(simulate_user_input, std::ref(input_stream));

    // Main event loop
    while (!exit_flag) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    input_thread.join();
    return 0;
}
