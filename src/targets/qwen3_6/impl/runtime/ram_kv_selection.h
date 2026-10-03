#pragma once

#include <ninfer/types.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace ninfer::targets::qwen3_6::detail {

// Training-free lexical prototype, deliberately distinct from KVMem's learned-space Q/K
// summary retrieval. Rank 256-token neighborhoods using document-frequency-weighted query
// overlap. Ties spread over the archive rather than consuming one chronological end.
inline std::vector<std::int32_t> rank_ram_kv_pages(std::span<const TokenId> history,
                                                std::span<const TokenId> query) {
    constexpr std::size_t block_tokens = 256;
    const auto blocks = (history.size() + block_tokens - 1) / block_tokens;
    std::unordered_set<TokenId> terms(query.begin(), query.end());
    std::unordered_map<TokenId, std::size_t> frequency;
    for (std::size_t b = 0; b < blocks; ++b) {
        std::unordered_set<TokenId> seen;
        for (std::size_t i = b * block_tokens;
             i < std::min(history.size(), (b + 1) * block_tokens); ++i) {
            if (terms.contains(history[i])) { seen.insert(history[i]); }
        }
        for (auto id : seen) { ++frequency[id]; }
    }
    std::vector<std::pair<double, std::int32_t>> scores;
    scores.reserve(blocks);
    for (std::size_t b = 0; b < blocks; ++b) {
        std::unordered_set<TokenId> seen;
        double score = 0;
        for (std::size_t i = b * block_tokens;
             i < std::min(history.size(), (b + 1) * block_tokens); ++i) {
            const auto id = history[i];
            if (terms.contains(id) && seen.insert(id).second) {
                score += std::log1p(static_cast<double>(blocks) / frequency[id]);
            }
        }
        scores.emplace_back(score, static_cast<std::int32_t>(b));
    }
    const auto spread = [](std::uint32_t b) {
        b ^= b >> 16; b *= 0x7feb352dU; b ^= b >> 15; b *= 0x846ca68bU;
        return b ^ (b >> 16);
    };
    std::sort(scores.begin(), scores.end(), [&](const auto& a, const auto& b) {
        return a.first != b.first ? a.first > b.first : spread(a.second) < spread(b.second);
    });
    std::vector<std::int32_t> pages;
    pages.reserve(blocks * 4);
    for (const auto& [score, block] : scores) {
        (void)score;
        for (int p = 0; p < 4; ++p) { pages.push_back(block * 4 + p); }
    }
    return pages;
}

} // namespace ninfer::targets::qwen3_6::detail
