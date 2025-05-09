#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <atomic>
#include <crow.h>
#include <nlohmann/json.hpp>

#include "fmt/format.h"
#include "codec.h"
#include "model.h"
#include "sampler.h"
#include "time.h"
#include "tokenizer.h"

// 使用nlohmann/json简化JSON处理
using json = nlohmann::json;

// 请求结构
struct InferenceRequest {
    std::string prompt;
    int max_tokens = 256;
    float temperature = 1.0;
    int context_length = 0;
    std::string device = "cuda";
};

// 响应结构
struct InferenceResponse {
    std::string generated_text;
    int tokens_generated = 0;
    double time_taken = 0.0;
};

// 推理任务队列
class InferenceQueue {
public:
    InferenceQueue() : stop_flag(false) {}

    void add_task(InferenceRequest req, std::function<void(InferenceResponse)> callback) {
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            tasks.push({std::move(req), std::move(callback)});
        }
        cv.notify_one();
    }

    std::pair<InferenceRequest, std::function<void(InferenceResponse)>> get_task() {
        std::unique_lock<std::mutex> lock(queue_mutex);
        cv.wait(lock, [this] { return !tasks.empty() || stop_flag; });
        
        if (stop_flag && tasks.empty()) {
            return {{}, nullptr};
        }
        
        auto task = std::move(tasks.front());
        tasks.pop();
        return task;
    }

    void stop() {
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            stop_flag = true;
        }
        cv.notify_all();
    }

    bool is_stopped() const {
        return stop_flag;
    }

private:
    std::queue<std::pair<InferenceRequest, std::function<void(InferenceResponse)>>> tasks;
    std::mutex queue_mutex;
    std::condition_variable cv;
    std::atomic<bool> stop_flag;
};

// 模型管理类
class ModelManager {
public:
    ModelManager(const std::string& checkpoint_path) 
        : checkpoint_path(checkpoint_path), model_ready(false) {
        // 初始化加载模型
        load_model();
    }

    ~ModelManager() {
        inference_queue.stop();
        if (worker_thread.joinable()) {
            worker_thread.join();
        }
    }

    void start_worker() {
        worker_thread = std::thread(&ModelManager::process_queue, this);
    }

    void queue_inference(InferenceRequest req, std::function<void(InferenceResponse)> callback) {
        inference_queue.add_task(std::move(req), std::move(callback));
    }

    bool is_model_ready() const {
        return model_ready;
    }

private:
    void load_model() {
        try {
            std::cout << "Loading model from " << checkpoint_path << std::endl;
            model_data = std::make_unique<YALMData>();
            model_data->from_file(checkpoint_path);
            
            // 模型加载成功
            model_ready = true;
            
            // 启动工作线程
            start_worker();
        } 
        catch (const std::exception& e) {
            std::cerr << "Error loading model: " << e.what() << std::endl;
            model_ready = false;
        }
    }

    InferenceResponse run_inference(const InferenceRequest& req) {
        InferenceResponse response;
        uint64_t start_ms = get_timestamp_ms();
        
        // 创建模型实例（为每次推理创建新实例以确保状态隔离）
        Model model(*model_data, req.context_length);
        InferenceState state(model.config);
        Sampler sampler(model.config, get_timestamp_ms());
        Tokenizer tokenizer(*model_data);

        if (req.device == "cuda") {
            model.cuda();
            state.cuda();
        }

        // 进行一次前向推理作为预热
        model.forward(state, 0, 0);

        // 编码提示词
        std::vector<int> encoding = tokenizer.encode(req.prompt, true);
        
        // 分配输出字符串
        std::stringstream output;
        
        // 为KV缓存填充所有提示词tokens
        for (size_t pos = 0; pos < encoding.size(); pos++) {
            int token_id = encoding[pos];
            InferenceMode inferMode = pos + 1 == encoding.size() ? 
                InferenceMode::OUTPUT_LOGITS : InferenceMode::HYDRATE_KV_CACHE;
            model.forward(state, token_id, pos, inferMode);
        }

        // 进行生成
        int max_steps = req.max_tokens;
        if (max_steps == 0) {
            max_steps = model.config->max_seq_len;
        }
        
        for (int i = 0; i < max_steps; i++) {
            int token_id = sampler.sample(state, req.temperature);
            std::string token_str = tokenizer.decode_one(encoding.back(), token_id);
            output << token_str;
            encoding.push_back(token_id);
            if (token_id == tokenizer.eos_id || token_id == tokenizer.eot_id) {
                break;
            }
            model.forward(state, token_id, encoding.size() - 1);
        }
        
        uint64_t end_ms = get_timestamp_ms();
        response.generated_text = output.str();
        response.tokens_generated = encoding.size() - tokenizer.encode(req.prompt, true).size();
        response.time_taken = (end_ms - start_ms) / 1000.0;
        
        return response;
    }

    void process_queue() {
        while (!inference_queue.is_stopped()) {
            auto [request, callback] = inference_queue.get_task();
            
            if (!callback) {
                continue;  // 队列被停止
            }
            
            try {
                auto response = run_inference(request);
                callback(response);
            } 
            catch (const std::exception& e) {
                InferenceResponse error_response;
                error_response.generated_text = std::string("Error: ") + e.what();
                callback(error_response);
            }
        }
    }

    std::string checkpoint_path;
    std::unique_ptr<YALMData> model_data;
    std::atomic<bool> model_ready;
    InferenceQueue inference_queue;
    std::thread worker_thread;
};

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: server <checkpoint_path> [port]" << std::endl;
        return 1;
    }
    
    std::string checkpoint_path = argv[1];
    int port = (argc > 2) ? std::stoi(argv[2]) : 8080;
    
    // 初始化模型管理器
    ModelManager model_manager(checkpoint_path);
    
    // 如果模型加载失败，退出
    if (!model_manager.is_model_ready()) {
        std::cerr << "Failed to initialize model. Exiting." << std::endl;
        return 1;
    }
    
    // 创建Crow应用
    crow::SimpleApp app;
    
    // 健康检查端点
    CROW_ROUTE(app, "/health")
    ([]() {
        return crow::response(200, "OK");
    });
    
    // 生成文本端点
    CROW_ROUTE(app, "/v1/generate")
    .methods(crow::HTTPMethod::POST)
    ([&model_manager](const crow::request& req) {
        crow::response res;
        res.set_header("Content-Type", "application/json");
        
        try {
            // 解析请求JSON
            auto request_json = json::parse(req.body);
            InferenceRequest inf_req;
            
            // 提取参数
            if (request_json.contains("prompt")) {
                inf_req.prompt = request_json["prompt"].get<std::string>();
            } else {
                res.code = 400;
                res.write("{\"error\": \"Missing 'prompt' field\"}");
                return res;
            }
            
            if (request_json.contains("max_tokens")) {
                inf_req.max_tokens = request_json["max_tokens"].get<int>();
            }
            
            if (request_json.contains("temperature")) {
                inf_req.temperature = request_json["temperature"].get<float>();
            }
            
            if (request_json.contains("context_length")) {
                inf_req.context_length = request_json["context_length"].get<int>();
            }
            
            if (request_json.contains("device")) {
                inf_req.device = request_json["device"].get<std::string>();
                if (inf_req.device != "cpu" && inf_req.device != "cuda") {
                    res.code = 400;
                    res.write("{\"error\": \"Device must be 'cpu' or 'cuda'\"}");
                    return res;
                }
            }
            
            // 创建响应处理器
            auto response_handler = std::make_shared<crow::response>();
            auto promise = std::make_shared<std::promise<void>>();
            auto future = promise->get_future();
            
            // 添加到推理队列
            model_manager.queue_inference(std::move(inf_req), [response_handler, promise](InferenceResponse inf_res) {
                // 构造JSON响应
                json response_json = {
                    {"generated_text", inf_res.generated_text},
                    {"tokens_generated", inf_res.tokens_generated},
                    {"time_taken", inf_res.time_taken}
                };
                
                response_handler->write(response_json.dump(2));
                promise->set_value();
            });
            
            // 等待异步处理完成
            future.wait();
            return *response_handler;
        } 
        catch (const std::exception& e) {
            res.code = 500;
            res.write(fmt::format("{{\"error\": \"{}\" }}", e.what()));
            return res;
        }
    });
    
    // 启动服务器
    std::cout << fmt::format("YALM API server running on port {}", port) << std::endl;
    app.port(port).multithreaded().run();
    
    return 0;
}