
#ifndef COMMAND_LINE_PARSER_H
#define COMMAND_LINE_PARSER_H

#include <algorithm>
#include <unordered_map>
#include <vector>
#include <string>

class CommandLineParser {
public:
    CommandLineParser(int argc, char* argv[]) {
        parseArguments(argc, argv);
    }

    std::string getValue(const std::string& key) const {
        auto it = arguments_.find(key);
        return (it != arguments_.end()) ? it->second : "";
    }

    int getValueAsInt(const std::string& key) const {
        auto it = arguments_.find(key);
        return (it != arguments_.end()) ? std::stoi(it->second) : 0;
    }

    long getValueAsLong(const std::string& key) const {
        auto it = arguments_.find(key);
        return (it != arguments_.end()) ? std::stol(it->second) : 0;
    }

    std::vector<int> getValueAsListOfInt(const std::string& key) const {
        std::vector<int> result;
        auto it = arguments_.find(key);
        if (it != arguments_.end()) {
            auto values = split(it->second, ':');
            std::transform(values.begin(), values.end(), std::back_inserter(result),
                           [](const std::string& str) { return std::stoi(str); });
        }
        return result;
    }

    std::vector<std::string> getValueAsListOfString(const std::string& key) const {
        std::vector<std::string> result;
        auto it = arguments_.find(key);
        if (it != arguments_.end()) {
            if (it->second.find(':') != std::string::npos) {
                result = split(it->second, ':');
            } else {
                result.push_back(it->second);
            }
        }
        return result;
    }

private:
    void parseArguments(int argc, char* argv[]) {
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg[0] == '-') {
                std::string key = arg.substr(1);
                if (i + 1 < argc && argv[i + 1][0] != '-') {
                    arguments_[key] = argv[i + 1];
                    ++i; // Skip the next argument as it is the value for the current key
                } else {
                    arguments_[key] = ""; // Set the value as an empty string if no value is provided
                }
            }
        }
    }

    std::vector<std::string> split(const std::string& s, char delimiter) const {
        std::vector<std::string> tokens;
        size_t start = 0;
        size_t end = s.find(delimiter);
        while (end != std::string::npos) {
            tokens.push_back(s.substr(start, end - start));
            start = end + 1;
            end = s.find(delimiter, start);
        }
        tokens.push_back(s.substr(start, end));
        return tokens;
    }

    std::unordered_map<std::string, std::string> arguments_;
};

#endif // COMMAND_LINE_PARSER_H
