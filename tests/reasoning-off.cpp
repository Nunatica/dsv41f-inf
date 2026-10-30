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
        const json messages = json::array({{{"role", "user"}, {"content", "Explain story of the romance of three kingdoms."}}});
        auto check = [&](json fields, bool off, int effort, int budget = -1) {
            fields["messages"] = messages;
            fields["max_tokens"] = 5;
            std::vector<raw_buffer> files;
            common_chat_session session;
            const auto params = oaicompat_chat_params_parse(nullptr, fields, opt, files, session);
            const auto prompt = session.prompt();
            require(prompt.find("<｜System｜>Reasoning Effort: " + std::to_string(effort) + " (range 1-100") != std::string::npos,
                    "wrong DeepSeek V4.1 effort in prompt");
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
                for (const char c : std::string("Consider the question.</think>Answer")) {
                    session.feed(common_chat_input(std::string(1, c)));
                }
                const auto msg = session.finish();
                require(msg.reasoning_content == "Consider the question.", "On reasoning was not parsed");
                require(msg.content == "Answer", "On answer was not separated");
            }
        };
        check(json::object(), true, 50);
        for (const char * level : {"none", "off", "minimal", "disabled", "false", " NONE "})
            check({{"reasoning_effort", level}}, true, 50);
        check({{"reasoning_effort", false}}, true, 50);
        check({{"reasoning", {{"effort", "none"}}}}, true, 50);
        check({{"chat_template_kwargs", {{"enable_thinking", false}}}}, true, 50);
        check({{"chat_template_kwargs", {{"reasoning_effort", "none"}}}}, true, 50);
        check({{"reasoning_effort", "none"}, {"reasoning_budget_tokens", 1024}, {"reasoning_budget_message", "override"}}, true, 50);
        check({{"reasoning_effort", "high"}, {"chat_template_kwargs", {{"enable_thinking", false}}}}, true, 50);
        check({{"chat_template_kwargs", {{"thinking", false}}}}, true, 50);
        check({{"reasoning_effort", "high"}, {"chat_template_kwargs", {{"thinking", false}}}}, true, 50);
        check({{"chat_template_kwargs", {{"thinking", false}, {"enable_thinking", false}}}}, true, 50);
        check({{"chat_template_kwargs", {{"thinking", true}, {"enable_thinking", true}}}}, false, 75);
        check({{"reasoning_effort", "low"}}, false, 50);
        check({{"reasoning_effort", "high"}, {"reasoning_budget_tokens", 128}}, false, 75, 128);
        check({{"reasoning_effort", "medium"}}, false, 75);
        check({{"reasoning_effort", "max"}}, false, 100);
        check({{"reasoning_effort", "xhigh"}}, false, 100);
        check({{"reasoning_effort", "maximum"}}, false, 100);
        for (int effort : {1, 50, 75, 100}) {
            check({{"reasoning_effort", effort}}, false, effort);
            check({{"reasoning_effort", std::to_string(effort)}}, false, effort);
            check({{"reasoning", {{"effort", effort}}}}, false, effort);
            check({{"chat_template_kwargs", {{"reasoning_effort", effort}}}}, false, effort);
        }
        check({{"reasoning_effort", " HIGH "}}, false, 75);
        check({{"chat_template_kwargs", {{"thinking_mode", "thinking"}}}}, false, 75);
        check({{"reasoning_effort", "high"}, {"chat_template_kwargs", {{"thinking_mode", "chat"}}}}, true, 50);
        check({{"chat_template_kwargs", {{"thinking_mode", "thinking"}, {"enable_thinking", false}}}}, false, 75);
        check({{"reasoning_effort", "none"}, {"chat_template_kwargs", {{"thinking_mode", "thinking"}}}}, false, 50);
        check({{"chat_template_kwargs", {{"thinking_mode", nullptr}}}}, true, 50);
        auto reject = [&](const json & fields) {
            json body = fields;
            body["messages"] = messages;
            std::vector<raw_buffer> files;
            common_chat_session session;
            bool rejected = false;
            try { oaicompat_chat_params_parse(nullptr, body, opt, files, session); }
            catch (const std::invalid_argument &) { rejected = true; }
            require(rejected, "invalid reasoning option was not rejected by the API");
        };
        for (const auto & effort : json::array({0, 101, -1, 1.5, true, "0", "101", "050", "1.0", "garbage", json::object()})) {
            reject({{"reasoning_effort", effort}});
            reject({{"chat_template_kwargs", {{"reasoning_effort", effort}}}});
        }
        for (const auto & mode : json::array({false, 1, "invalid"})) {
            reject({{"chat_template_kwargs", {{"thinking_mode", mode}}}});
        }
        opt.reasoning_budget = 4096;
        check({{"reasoning_effort", "high"}}, false, 75, 4096);
        check(json::object(), true, 50);
        opt.reasoning_budget = -1;
        opt.enable_thinking = true;
        opt.chat_template_kwargs.clear();
        check({{"reasoning_effort", "none"}}, true, 50);
        check(json::object(), false, 75);
        auto anthropic = server_chat_convert_anthropic_to_oai({{"messages", messages}, {"thinking", {{"type", "disabled"}}}});
        check(anthropic, true, 50);
        opt.enable_thinking = false;
        anthropic = server_chat_convert_anthropic_to_oai({{"messages", messages}, {"thinking", {{"type", "enabled"}, {"budget_tokens", 32}}}});
        check(anthropic, false, 75, 32);
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
            const std::string tag = spaced ? "｜DSML｜ " : "｜DSML｜";
            const std::string block = spaced ? "calls" : "tool_calls";
            for (const char * choice : {"required", "auto"}) {
                for (bool off : {true, false}) {
                    json body = {{"messages", messages}, {"reasoning_effort", off ? "none" : "high"},
                                 {"tools", json::array({tool})}, {"tool_choice", choice}};
                    std::vector<raw_buffer> files; common_chat_session session;
                    oaicompat_chat_params_parse(nullptr, body, opt, files, session);
                    const std::string reasoning = off ? DSV41F_REASONING_BUDGET_MESSAGE : "Use add.";
                    const std::string output = reasoning +
                        "</think>\n\n<" + tag + block + ">\n<" + tag + "invoke name=\"add\">\n<" + tag +
                        "parameter name=\"a\" string=\"false\">2</" + tag + "parameter>\n</" + tag +
                        "invoke>\n</" + tag + block + ">";
                    for (const char c : output) session.feed(common_chat_input(std::string(1, c)));
                    const auto msg = session.finish();
                    require(msg.reasoning_content == reasoning, "tool call lost reasoning");
                    require(msg.tool_calls.size() == 1 && msg.tool_calls[0].name == "add", "tool markup was not parsed");
                    require(json::parse(msg.tool_calls[0].arguments).at("a") == 2, "tool argument was not parsed");
                    require(msg.content.empty(), "tool markup leaked into content");
                }
            }
            // Replay API string arguments and out-of-order results in a multi-turn history.
            json history = json::array({
                {{"role", "developer"}, {"content", "Be concise."}},
                {{"role", "user"}, {"content", json::array({{{"type", "text"}, {"text", "Calculate"}}})}},
                {{"role", "assistant"}, {"content", nullptr}, {"reasoning_content", "Use add."},
                 {"tool_calls", json::array({
                    {{"id", "first"}, {"type", "function"}, {"function", {{"name", "add"}, {"arguments", "{\"a\":2}"}}}},
                    {{"id", "second"}, {"type", "function"}, {"function", {{"name", "add"}, {"arguments", "{\"a\":3}"}}}}})}},
                {{"role", "tool"}, {"tool_call_id", "second"}, {"content", "second-result"}},
                {{"role", "tool"}, {"tool_call_id", "first"}, {"content", "first-result"}}
            });
            json body = {{"messages", history}, {"tools", json::array({tool})}, {"reasoning_effort", "high"}};
            std::vector<raw_buffer> files; common_chat_session session;
            oaicompat_chat_params_parse(nullptr, body, opt, files, session);
            const auto prompt = session.prompt();
            require(prompt.find("Be concise.") != std::string::npos, "developer instructions lost");
            require(prompt.find("<｜User｜>Calculate<｜Assistant｜><think>Use add.</think>") != std::string::npos,
                    "typed content or tool reasoning history lost");
            require(prompt.find("parameter name=\"a\" string=\"false\">2") != std::string::npos,
                    "API tool arguments were not converted to native parameters");
            require(prompt.find("<tool_result>first-result</tool_result>\n\n<tool_result>second-result</tool_result>") != std::string::npos,
                    "tool results were not reordered to match calls");
        }
        opt.tmpls = common_chat_templates_init(nullptr, source.str());
        {
            const json schema = {{"type", "object"}, {"properties", {{"answer", {{"type", "integer"}}}}},
                                 {"required", json::array({"answer"})}};
            json body = {{"messages", messages}, {"reasoning_effort", "high"},
                         {"response_format", {{"type", "json_schema"}, {"json_schema", {{"name", "answer"}, {"schema", schema}}}}}};
            std::vector<raw_buffer> files; common_chat_session session;
            oaicompat_chat_params_parse(nullptr, body, opt, files, session);
            require(session.prompt().find("## Response Format:") != std::string::npos, "JSON schema missing from prompt");
            require(!session.grammar().empty(), "JSON schema grammar was not generated");
            const auto msg = session.finish(common_chat_input("Compute.</think>```json\n{\"answer\":4}\n```"));
            require(json::parse(msg.content).at("answer") == 4, "JSON answer not parsed");
        }
        // An unrelated template must retain the original budget/message behavior.
        opt.tmpls = common_chat_templates_init(nullptr,
            "{% for m in messages %}{{ m.content }}{% endfor %}{% if add_generation_prompt %}<think>{% endif %}");
        json body = {{"messages", messages}, {"reasoning_effort", "none"}};
        std::vector<raw_buffer> files; common_chat_session session;
        auto params = oaicompat_chat_params_parse(nullptr, body, opt, files, session);
        require(params.value("reasoning_budget_message", "") != DSV41F_REASONING_BUDGET_MESSAGE, "other-template behavior changed");
        std::puts("PASS: DeepSeek V4.1 efforts/modes, forced Off decode, streaming DSML tools, history and JSON schema");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
}
