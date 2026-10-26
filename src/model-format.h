#pragma once
#include <filesystem>
#include <string>

enum class dsv4f_model_format { unknown, dsx, ud_q4_k_xl };
inline dsv4f_model_format dsv4f_format(const std::string & path) {
    const auto name=std::filesystem::path(path).filename().string();
    if(name=="DeepSeek-V4.1-Flash-1017-dsx-2.6bit.gguf")return dsv4f_model_format::dsx;
    if(name=="DeepSeek-V4.1-Flash-1017-UD-Q4_K_XL-00001-of-00009.gguf")return dsv4f_model_format::ud_q4_k_xl;
    return dsv4f_model_format::unknown;
}
