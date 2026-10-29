#include "server-common.h"
#include "server-chat.h"
#include "chat.h"
#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>

static void require(bool value, const char * message) {
    if (!value) throw std::runtime_error(message);
}

int main(int argc, char ** argv) {
    try {
        require(argc == 2, "template file required");
        std::ifstream file(argv[1]);
        std::ostringstream source; source << file.rdbuf();
        require(!source.str().empty(), "empty DeepSeek V4.1 template");
        server_chat_params opt{};
        opt.use_jinja = true;
        opt.enable_thinking = false;
        opt.reasoning_budget = -1;
        opt.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
        opt.chat_template_kwargs["enable_thinking"] = "false"; // --reasoning off
        opt.tmpls = common_chat_templates_init(nullptr, source.str());
        const json messages = json::array({{{"role", "user"}, {"content", "2 + 2?"}}});
        auto check = [&](json fields, bool off, const std::string & level, int budget = -1) {
            fields["messages"] = messages;
            fields["max_tokens"] = 5;
            std::vector<raw_buffer> files;
            common_chat_session session;
            const auto params = oaicompat_chat_params_parse(nullptr, fields, opt, files, session);
            const auto prompt = session.prompt();
            if (level == "High" || level == "Max") {
                require(prompt.find("Reasoning Effort: " + std::string(level == "High" ? "75" : "100")) != std::string::npos,
                        "wrong DeepSeek V4.1 effort in prompt");
            } else {
                require(prompt.find("Reasoning Effort:") == std::string::npos, "Low must use the base DeepSeek effort");
            }
            require(prompt.size() >= 7 && prompt.substr(prompt.size()-7) == "<think>", "thinking block was closed in prefill");
            require(prompt.find(DSV41F_REASONING_BUDGET_MESSAGE) == std::string::npos, "forced message leaked into prompt");
            if (params.at("reasoning_budget_tokens") != (off ? 0 : budget)) {
                throw std::runtime_error("wrong reasoning budget for " + fields.dump() + ": " + params.dump());
            }
            require(params.at("max_tokens") == 5, "completion cap changed");
            if (off) {
                require(params.at("reasoning_budget_message") == DSV41F_REASONING_BUDGET_MESSAGE, "wrong forced message");
                // Split even the closing tag across chunks, as SSE output can do.
                for (const char c : std::string(DSV41F_REASONING_BUDGET_MESSAGE) + "</think>4") {
                    session.feed(common_chat_input(std::string(1, c)));
                }
                const auto msg = session.finish();
                require(msg.reasoning_content == DSV41F_REASONING_BUDGET_MESSAGE, "budget message not parsed as reasoning");
                require(msg.content == "4", "answer not separated from reasoning");
            } else {
                require(params.at("reasoning_budget_message") == "", "forced Off message leaked into On");
            }
        };
        check(json::object(), true, "Low");
        for (const char * level : {"none", "off", "minimal", "disabled", "false", " NONE "})
            check({{"reasoning_effort", level}}, true, "Low");
        check({{"reasoning_effort", false}}, true, "Low");
        check({{"reasoning", {{"effort", "none"}}}}, true, "Low");
        check({{"chat_template_kwargs", {{"enable_thinking", false}}}}, true, "Low");
        check({{"chat_template_kwargs", {{"reasoning_effort", "none"}}}}, true, "Low");
        check({{"reasoning_effort", "none"}, {"reasoning_budget_tokens", 1024}, {"reasoning_budget_message", "override"}}, true, "Low");
        check({{"reasoning_effort", "high"}, {"chat_template_kwargs", {{"enable_thinking", false}}}}, true, "Low");
        check({{"chat_template_kwargs", {{"thinking", false}}}}, true, "Low");
        check({{"reasoning_effort", "high"}, {"chat_template_kwargs", {{"thinking", false}}}}, true, "Low");
        check({{"chat_template_kwargs", {{"thinking", false}, {"enable_thinking", false}}}}, true, "Low");
        check({{"chat_template_kwargs", {{"thinking", true}, {"enable_thinking", true}}}}, false, "Low");
        check({{"reasoning_effort", "low"}}, false, "Low");
        check({{"reasoning_effort", "high"}, {"reasoning_budget_tokens", 128}}, false, "High", 128);
        check({{"reasoning_effort", "medium"}}, false, "High");
        check({{"reasoning_effort", "max"}}, false, "Max");
        check({{"reasoning_effort", "xhigh"}}, false, "Max");
        opt.reasoning_budget = 4096;
        check({{"reasoning_effort", "high"}}, false, "High", 4096);
        check(json::object(), true, "Low");
        opt.reasoning_budget = -1;
        opt.enable_thinking = true;
        opt.chat_template_kwargs.clear();
        check({{"reasoning_effort", "none"}}, true, "Low");
        check(json::object(), false, "Low");
        auto anthropic = server_chat_convert_anthropic_to_oai({{"messages", messages}, {"thinking", {{"type", "disabled"}}}});
        check(anthropic, true, "Low");
        opt.enable_thinking = false;
        anthropic = server_chat_convert_anthropic_to_oai({{"messages", messages}, {"thinking", {{"type", "enabled"}, {"budget_tokens", 32}}}});
        check(anthropic, false, "Low", 32);
        // Off's forced prefix must not hide tools. Check the bundled spaced
        // DSML syntax and the canonical V4 syntax supported by the same parser.
        for (bool spaced : {true, false}) {
            auto template_source = source.str();
            if (!spaced) {
                for (const auto & replacement : {std::pair<std::string, std::string>{" calls>", "tool_calls>"},
                                                {" invoke", "invoke"}, {" parameter", "parameter"}}) {
                    size_t pos = 0;
                    while ((pos = template_source.find(replacement.first, pos)) != std::string::npos) {
                        template_source.replace(pos, replacement.first.size(), replacement.second);
                        pos += replacement.second.size();
                    }
                }
            }
            opt.tmpls = common_chat_templates_init(nullptr, template_source);
            json tool = json::parse(R"({"type":"function","function":{"name":"add","parameters":{
                "type":"object","properties":{"a":{"type":"integer"}},"required":["a"]}}})");
            json body = {{"messages", messages}, {"reasoning_effort", "none"},
                         {"tools", json::array({tool})}, {"tool_choice", "required"}};
            std::vector<raw_buffer> files; common_chat_session session;
            oaicompat_chat_params_parse(nullptr, body, opt, files, session);
            const std::string tag = spaced ? "｜DSML｜ " : "｜DSML｜";
            const std::string block = spaced ? "calls" : "tool_calls";
            const auto msg = session.finish(common_chat_input(std::string(DSV41F_REASONING_BUDGET_MESSAGE) +
                "</think>\n\n<" + tag + block + ">\n<" + tag + "invoke name=\"add\">\n<" + tag +
                "parameter name=\"a\" string=\"false\">2</" + tag + "parameter>\n</" + tag +
                "invoke>\n</" + tag + block + ">"));
            require(msg.reasoning_content == DSV41F_REASONING_BUDGET_MESSAGE, "tool call lost forced reasoning");
            require(msg.tool_calls.size() == 1 && msg.tool_calls[0].name == "add", "tool markup was not parsed");
            require(json::parse(msg.tool_calls[0].arguments).at("a") == 2, "tool argument was not parsed");
        }
        // An unrelated template must retain the original budget/message behavior.
        opt.tmpls = common_chat_templates_init(nullptr,
            "{% for m in messages %}{{ m.content }}{% endfor %}{% if add_generation_prompt %}<think>{% endif %}");
        json body = {{"messages", messages}, {"reasoning_effort", "none"}};
        std::vector<raw_buffer> files; common_chat_session session;
        auto params = oaicompat_chat_params_parse(nullptr, body, opt, files, session);
        require(params.value("reasoning_budget_message", "") != DSV41F_REASONING_BUDGET_MESSAGE, "other-template behavior changed");
        std::puts("PASS: DeepSeek V4.1 Off renders Low/open think, forces through decode, separates reasoning, and preserves On/other templates");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
}
